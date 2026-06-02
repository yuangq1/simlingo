# TCP_Lang 数据流动过程

## 定位

**SimLingo VLM Encoder + TCP-style GRU Decoder**

- 前半：SimLingo VLM (ViT+LLM) → 从最后一层提取 lang_embed
- 后半：TCP-style autoregressive GRU decoder + route head
- lang_embed + state_embed 在 Join、空间注意力、轨迹GRU每步 3 处注入

---

## 前半：VLM Encoder (和 SimLingo 完全一致)

```
Image [B,3,H,W]                    Instruction "Shift 1 lane left"
    │                                      │
    └── InternVL ViT                       └── Tokenizer
        │                                      │
        image_features [B, 256, 896]          token_embeds [B, L, 896]
        │                                      │
        └──── 替换 <IMG_CONTEXT> ─────────────┘
                        │
                prompt_embeds [B, L_total, 896]
                        │
                InternLM2 + LoRA (r=32, α=64)
                24 层 self-attention
                        │
                hidden_states[-2] [B, L_total, 896]  ← 倒数第二层
                        │
        ┌───────────────┴────────────────┐
        │                                │
  mm_global = mean_pool(hidden)    vis_spatial = image_features
  [B, 896]                         [B, 256, 896]
  VLM 多模态全局特征                  ViT 空间特征
        │                                │
  lang_proj:                         mean_pool over 256 tokens
  Linear(896→256)+ReLU                    │
  +Linear(256→128)                  vis_pooled [B, 896]
        │
  lang_embed [B, 128]
  从 VLM 最后一层提取的语义向量
  (不生成文本, 直接从 hidden_states 池化)
```

---

## 后半：TCP-style Decoder

### State Embedding (TCP measurement branch)

```
speed [B,1] + target_point [B,2] + command [B,6]
        │
  state = concat [B, 9]
        │
  state_encoder: Linear(9→128)+ReLU+Linear(128→128)+ReLU
        │
  state_embed [B, 128]
```

### 位置 1: Join 融合 (lang注入 ①)

```
mm_global [B,896]  +  vis_pooled [B,896]  +  lang_embed [B,128]  +  state_embed [B,128]
        │                     │                      │                      │
        └─────────────────────┼──────────────────────┼──────────────────────┘
                              │
        Join: Linear(2048→896) + ReLU + Linear(896→896) + ReLU
                              │
                        j [B, 896]  初始隐藏状态
```

### 位置 2: 空间注意力 (lang注入 ②)

```
vis_spatial [B,256,896]  +  lang_embed [B,256,128]  +  state_embed [B,256,128]
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   │
        Attn: Linear(1152→256) + ReLU + Linear(256→1) + Softmax over 256 tokens
                                   │
                    att_weights [B, 256]
                                   │
                    att_vis = weighted_sum(vis_spatial) [B, 896]
```

### 位置 3: Trajectory GRU (lang注入 ③, 每步)

```
每步 t = 0..pred_len-1:
    previous_wp [B,2] + target_point [B,2] + lang_embed [B,128] + state_embed [B,128]
            │                │                    │                    │
            └────────────────┼────────────────────┼────────────────────┘
                             │
                  GRUCell(260→896, hidden=896)
                             │
                  hidden [B, 896]
                             │
                  TrajOut: Linear(896→256)+ReLU+Linear(256→2)
                             │
                  delta [B,2] → wp_t = wp_{t-1} + delta

→ pred_wp [B, pred_len, 2]
```

### Route Head

```
j [B,896] + att_vis [B,896] + lang_embed [B,128] + state_embed [B,128]
        │
  RouteHead: Linear(2048→512)+SiLU+Linear(512→256)+SiLU+Linear(256→40)
        │
  pred_route [B, 20, 2]
```

---

## 维度速查

| 步骤 | 输入 | 输出 |
|------|------|------|
| ViT 编码 | image [B,3,H,W] | image_features [B,256,896] |
| Tokenizer | text List[str] | tokens [B,L] |
| LLM 前向 | prompt_embeds [B,N,896] | hidden_states[-1] [B,N,896] |
| mm_global | hidden_states[-1] [B,N,896] | mean_pool → [B,896] |
| lang_embed | mm_global [B,896] | lang_proj → [B,128] |
| vis_pooled | image_features [B,256,896] | mean_pool → [B,896] |
| state_embed | speed(1)+tp(2)+cmd(6) [B,9] | MLP → [B,128] |
| Join | mm(896)+vis_p(896)+lang(128)+state(128) | j [B,896] |
| 空间注意力 | vis_spatial(256,896)+lang(128)+state(128) | att_vis [B,896] |
| GRU 每步 | wp(2)+tp(2)+lang(128)+state(128) | hidden [B,896] |
| Traj 输出 | hidden [B,896] | delta [B,2] |
| Waypoints | pred_len 步累积 | pred_wp [B,10,2] |
| Route | j+att_vis+lang+state [B,2048] | pred_route [B,20,2] |

---

## Loss (完全对齐 SimLingo)

| Loss | 公式 | 权重 |
|------|------|------|
| wp_loss | SmoothL1(pred_wp, gt_waypoints) | 1.0 |
| route_loss | SmoothL1(pred_route, gt_route) | 1.0 |
| language_loss | 0 (不生成文本) | 0.1 |

```
total = wp_loss + route_loss + 0.1 * 0
```

---

## lang_embed 注入位置 (3 处)

| # | 位置 | 方式 | 维度变化 |
|---|------|------|---------|
| ① | Join 融合 | concat | mm(896)+vis_p(896)+lang(128)+state(128)→j(896) |
| ② | 空间注意力 | concat | vis(896)+lang(128)+state(128)→att(256)→1→weighted_sum |
| ③ | GRU 每步 | concat到GRU输入 | wp(2)+tp(2)+lang(128)+state(128)→GRUCell(260→896) |

---

## 可视化

- 每 100 步：控制台打印 loss + GT/Pred waypoints
- 每 1000 步：保存 `visualise_tcp_lang/step_XXXXXX.png`
  - 左上：lang_embed 各维度值 (bar chart)
  - 右上：lang_embed 分布直方图
  - 左下：Waypoints GT vs Pred
  - 右下：Route GT vs Pred

---

## 训练命令

```bash
conda activate simlingo
cd ~/projects/SimLingo/simlingo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_lang
```

5 epoch, batch_size=2, 完全对齐 SimLingo 训练流程。
