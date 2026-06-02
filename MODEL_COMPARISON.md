# TCP 模型版本对比（ResNet-34 主干 + SimLingo 语言辅助）

每个版本用相同数据（simlingo, Town13, 5 epoch），评估口语泛化（0%/50%/100%）。

**核心原则**：所有版本使用 ResNet-34 作为视觉 backbone，SimLingo VLM 仅作为辅助语言编码器。

---

## v0 — 原始 TCP baseline

```
RGB → ResNet-34 → feature_emb(1000) + cnn_feature(512,h,w)
speed(1) + tp(2) + cmd(6) → measurements(9→128) → state_embed
```

| 模块 | 输入 |
|------|------|
| Join | feature_emb + vis_pooled + state_embed |
| Attention | cnn_feature + state_embed |
| GRU | x(2) + tp(2) + state_embed |
| Route | j + att_vis + state_embed |
| 语言 | **无** |

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 2.6321 | 2.7246 | 2.6952 |
| Waypoints FDE | 6.5558 | 6.9017 | 6.7779 |
| Route ADE | 0.6321 | 0.5742 | 0.6093 |
| Route FDE | 0.8943 | 0.8348 | 0.8591 |

---

## v1 — + lang_embed 只注入 GRU

vs v0：仅 GRU 增加 lang_embed(128)，其余模块不变。

| 模块 | 输入 |
|------|------|
| Join | feature_emb + vis_pooled + state_embed |
| Attention | cnn_feature + state_embed |
| GRU | x(2) + tp(2) + state_embed + **lang_embed** |
| Route | j + att_vis + state_embed |

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 0.8625 | 0.8995 | 1.0109 |
| Waypoints FDE | 2.093 | 2.2406 | 2.5115 |
| Route ADE | 0.5829 | 0.5538 | 0.587 |
| Route FDE | 0.835 | 0.7738 | 0.8838 |

---

## v2 — lang_embed 分散注入

vs v1：lang_embed 注入全部 4 个模块。

| 模块 | 输入 |
|------|------|
| Join | feature_emb + vis_pooled + state_embed + **lang_embed** |
| Attention | cnn_feature + state_embed + **lang_embed** |
| GRU | x(2) + tp(2) + state_embed + **lang_embed** |
| Route | j + att_vis + state_embed + **lang_embed** |

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 0.8768 | 0.9119 | 0.8363 |
| Waypoints FDE | 2.2414 | 2.3332 | 2.0965 |
| Route ADE | 0.2914 | 0.2847 | 0.2484 |
| Route FDE | 0.4302 | 0.3993 | 0.3513 |

---

## v3 — 统一融合，保留 command（★主模型）

vs v2：新增 fusion_mlp，感知/规划分离。
- fusion_mlp: lang(128) + state(128) → fused(256)，**只算一次**
- Join 保留原始 lang+state（不预融合）
- GRU/Route 使用 fused（规划）
- Attention 仅用 lang（感知）

| 模块 | 输入 |
|------|------|
| Join | feature_emb + vis_pooled + **lang_embed + state_embed** |
| Attention | cnn_feature + **lang_embed only** |
| GRU | x(2) + tp(2) + **fused_embed** |
| Route | j + att_vis + **fused_embed** |

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 0.8949 | 0.9567 | 0.9444 |
| Waypoints FDE | 2.3084 | 2.3303 | 2.3559 |
| Route ADE | 0.2493 | 0.2506 | 0.2768 |
| Route FDE | 0.4085 | 0.3815 | 0.3997 |

---

## v4 — 统一融合，无 command（语言替代导航意图）

vs v3：去掉 command one-hot，state 仅 speed(1)+tp(2)=3 维。
导航意图完全由 lang_embed 提供。

| 模块 | 输入 |
|------|------|
| Join | feature_emb + vis_pooled + **lang_embed + state_embed** |
| Attention | cnn_feature + **lang_embed only** |
| GRU | x(2) + tp(2) + **fused_embed** |
| Route | j + att_vis + **fused_embed** |

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 0.9460 | 0.9736 | 0.9524 |
| Waypoints FDE | 2.3539 | 2.4631 | 2.3291 |
| Route ADE | 0.2701 | 0.2725 | 0.2687 |
| Route FDE | 0.4006 | 0.4114 | 0.4112 |

| | ResNet-34 | VLM | Command | 语言注入 | 核心问题 |
|---|---|---|---|---|---|
| v0 | ✅ | ❌ | ✅ | — | baseline |
| v1 | ✅ | ✅ | ✅ | GRU only | 语言能否帮助轨迹解码？ |
| v2 | ✅ | ✅ | ✅ | 全部模块 | 全面注入 vs 单点？ |
| v3 | ✅ | ✅ | ✅ | fusion(感知/规划分离) | 统一融合是否更好？ |
| v4 | ✅ | ✅ | ❌ | fusion(语言替代cmd) | 语言能否替代 hard command？ |
| v5-a | ✅ FiLM(layer4) | ✅ | ✅ | FiLM only (visual) | 语言条件化视觉 backbone？ |
| v5-b | ✅ FiLM(layer3+4) | ✅ | ✅ | FiLM only (visual) | 中层 FiLM 是否更好？ |
| v5-c | ✅ FiLM(layer1-4) | ✅ | ✅ | FiLM only (visual) | 全深度 FiLM 上限？ |
| v5-d | ✅ FiLM(layer1-4) | ✅ | ✅ | FiLM + v3 fusion | 视觉+规划联合条件化？ |

---

## v5 系列 — language-conditioned ResNet via FiLM

**核心思路**：不用 VLM 替换 ResNet-34，而是通过 FiLM adapter 用语言 embedding 条件化原始 ResNet-34 的特征提取。

```
RGB → ResNet-34 with FiLM(lang) after each stage → TCP decoder
```

FiLM (Feature-wise Linear Modulation):
- 从 lang_embed 生成 channel-wise gamma/beta
- 调制 feature map: `feat = feat * (1 + gamma) + beta`
- 不改变空间尺寸，参数量可控

### v5-a — FiLM layer4 only

lang_embed → FiLM(layer4: 512ch)，decoder = v0（无语言注入）

| 模块 | 输入 |
|------|------|
| ResNet layer1-3 | 标准 ResNet |
| ResNet layer4 | 标准 ResNet + **FiLM(lang)** |
| Join | feature_emb + vis_pooled + state_embed |
| Attention | cnn_feature + state_embed |
| GRU | x(2) + tp(2) + state_embed |
| Route | j + att_vis + state_embed |

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | — | — | — |
| Waypoints FDE | — | — | — |
| Route ADE | — | — | — |
| Route FDE | — | — | — |

---

### v5-b — FiLM layer3+4

lang_embed → FiLM(layer3: 256ch + layer4: 512ch)，decoder = v0

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | — | — | — |
| Waypoints FDE | — | — | — |
| Route ADE | — | — | — |
| Route FDE | — | — | — |

---

### v5-c — FiLM layer1-4

lang_embed → FiLM(layer1: 64ch + layer2: 128ch + layer3: 256ch + layer4: 512ch)，decoder = v0

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | — | — | — |
| Waypoints FDE | — | — | — |
| Route ADE | — | — | — |
| Route FDE | — | — | — |

---

### v5-d — FiLM layer1-4 + v3 fusion（★全模型语言条件化）

lang_embed → FiLM(layer1-4) + v3 decoder（fusion_mlp, 感知/规划分离）

| 模块 | 输入 |
|------|------|
| ResNet layer1-4 | 标准 ResNet + **FiLM(lang)** after each stage |
| Join | feature_emb + vis_pooled + lang_embed + state_embed |
| Attention | cnn_feature + **lang_embed only** |
| GRU | x(2) + tp(2) + **fused_embed** |
| Route | j + att_vis + **fused_embed** |

### 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | — | — | — |
| Waypoints FDE | — | — | — |
| Route ADE | — | — | — |
| Route FDE | — | — | — |

---

## 实验主线

```
v0 (ResNet-34, 纯视觉 TCP)
 └─ + SimLingo VLM → lang_embed
     └─ v1 (lang → GRU only)
          └─ lang → 全部模块
              └─ v2 (lang 分散注入)
                   └─ + fusion_mlp, 感知/规划分离
                       └─ v3 (融合, 保留 cmd, ★主模型)
                            ├─ - cmd, 语言替代导航意图
                            │   └─ v4 (融合, 无 cmd)
                            └─ + FiLM language-conditioned ResNet
                                ├─ v5-a (FiLM layer4)
                                ├─ v5-b (FiLM layer3+4)
                                ├─ v5-c (FiLM layer1-4)
                                └─ v5-d (FiLM layer1-4 + v3 fusion, ★全模型语言条件化)
```

---

## 训练

```bash
conda activate simlingo && cd ~/projects/SimLingo/simlingo && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v0
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v1
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v2
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v3
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v4
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v5a
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v5b
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v5c
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_v5d
```

## 评估

```bash
PYTHONPATH=$PWD python simlingo_training/eval_compare_lang.py experiment=tcp_v0 checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt
```
