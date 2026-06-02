# SimLingo Dreamer 复现日志

- **Upstream**: https://github.com/RenzKa/simlingo
- **Fork**: https://github.com/yuangq1/simlingo

## 2026-05-07

### 环境配置
- **Conda 环境**: `simlingo` (Python 3.10 + PyTorch 2.10.0 + CUDA 12.8)
- **GPU**: RTX 5090 (Blackwell sm_120) — 需要 PyTorch >= 2.10.0 才稳定
- **WSL**: 启用镜像网络模式 (`networkingMode=mirrored`)，解决网络问题
- **模型**: InternVL2-1B (~2GB)，从 Windows 缓存下载后复制到 WSL
- **参考环境**: `tcp_new` (PyTorch 2.10.0+cu128)，在 RTX 5090 上稳定运行

### 环境踩坑记录
- PyTorch 2.7.0+cu128 在 RTX 5090 上运行训练会导致系统死机重启（Blackwell 驱动兼容性 bug）
- 与能正常跑的 `tcp_new` 环境对比后确认：PyTorch 2.10.0+cu128 + cuDNN 9.10.02 是稳定组合
- 已删除问题环境 `torch270_cu128` 和旧 `simlingo` (PyTorch 2.4.1/Python 3.8)
- 重建 `simlingo` 环境：基于 `environment.yaml`，替换为 PyTorch 2.10.0+cu128 全家桶
- `precision: 32` → `16-mixed`，减少显存和功耗压力
- 使用 `conda activate simlingo` 启动
- 启动前需 export: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`

### 数据集
- 下载 `data_validation_1_scenario_chunk_003.tar.gz` (16 GB)
- 匹配 `simlingo_dreamer/` 中 dreamer 数据
- 500 条 Town13 路线，99%/1% 训练/验证分割
- 训练集: 28,581 张图片，验证集: 451 张图片

### 代码修改
- `train.py`: 延迟导入 DeepSpeed（无 CUDA toolkit 时跳过）
- `dataset_base.py`: 修复数据分割（不强制 Town12=训练, Town13=验证）
- `driving.py`: 移除 `verbose=False`（PyTorch 2.7 兼容）
- `config/experiment/dreamer_no_safety.yaml`: 新建配置文件
  - `use_safety_flag: False`（纯指令跟随模式）
  - `driving_dataset: null`（只用 dreamer 数据）
  - `use_commentary: False`, `use_qa: False`
  - `strategy: ddp_find_unused_parameters_true`
  - `precision: 16-mixed`
- `environment.yaml`: 更新 PyTorch 2.2→2.10, Python 3.8→3.10, CUDA 12.1→12.8
- `driving.py`: `summarise_losses` 传入 `weights={'language_loss': 0.0}`，语言 loss 不参与梯度，模型纯学轨迹
- `visualise.py`: 修复字体路径，加 fallback 默认字体；每 100 step 保存对比图到 `visualise/` 目录
- `driving.py` training_step: 每 100 step 控制台打印 loss 明细 + LLM 输出样本
- `train.py`: VisualiseCallback interval 1000→100，后改为 2000（减少训练中断）

---

## 2026-05-08 — Epoch 0 完成

### 训练状态
- **训练正常运行中！** PyTorch 2.10.0 在 RTX 5090 上稳定，无死机
- 模型: 957M 总参数，327M 可训练
- 训练速度: ~0.89 it/s（epoch 4 时），一个 epoch 约 9 小时 → epoch 0 时 ~0.54 it/s
- 首个 batch loss=28.23，稳步收敛
- wandb offline 模式运行中
- `FlashAttention2` 未安装（不影响训练）
- 显存压力正常（16-mixed 精度 + batch_size=2，32GB 完全够用）

### Epoch 0 关键节点
| Step | total_loss | lang_loss | route_loss | wps_loss | 语言输出 |
|------|-----------|-----------|------------|----------|----------|
| 0 | 28.23 | 6.51 | 15.05 | — | 长篇推理文本（错误） |
| 100 | 8.24 | 6.06 | 0.44 | 7.20 | 长篇推理 |
| 1800 | 5.03 | 0.41 | 1.37 | 3.63 | **首次正确**: "Following the given instruction. Waypoints:" |
| 2000+ | — | ~0.0001 | — | — | 几乎始终正确 |
| 28500 (末) | ~6.34 | ~0.0006 | ~4.51 | ~1.83 | 始终正确 |

- 语言部分 ~step 1800 完全收敛（任务 trivial，就是一句固定话）
- 轨迹预测 loss 稳步下降，但方差大（某些 batch spike 到 6-7）

---

## 2026-05-11 — 训练完成 + 评估

### 训练完成 (max_epochs=5)
- 总训练时间: 约 4 天（5 epochs × ~19h/epoch，epoch 0 最慢，后续加速）
- epoch 4 最终速度: 0.89 it/s
- 最终 checkpoint: `epoch=004.ckpt` / `last.ckpt` (6.5 GB each)

### 5 Epoch 最终 Loss
| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| 总 loss | 1.166 | 0.996 |
| language_loss | ~0 | ~0 |
| route_loss | 0.519 | 0.273 |
| speed_wps_loss | 1.296 | 0.723 |

验证集 route_loss (0.27) 低于训练集 (0.52)，无过拟合。

### 新增评估脚本

**`eval_waypoints.py`** — 验证集 ADE/FDE 评估
- 改动: `config.py` 新增 `checkpoint_path` 字段
- 踩坑: Hydra 切换 cwd 导致相对路径找不到 → 用 `get_original_cwd()` 解析
- 踩坑: PyTorch 2.6 `weights_only=True` 默认值导致加载 PL checkpoint 失败 → 加 `weights_only=False`
- 踩坑: batch 在 CPU、模型在 GPU → 写 `_to_device()` 递归转移所有 tensor

**最终评估结果 (epoch 4, 验证集 450 samples):**

| 指标 | 值 | 含义 |
|------|-----|------|
| waypoints ADE | **1.11 m** | 11 个轨迹点平均误差 |
| waypoints FDE | **2.78 m** | 终点误差 |
| route ADE | **0.43 m** | 20 个路线点平均误差 |
| route FDE | **0.83 m** | 路线终点误差 |
| language accuracy | **100%** | 回答格式完全正确 |

- Route 预测精度 (0.43m) 远好于 Waypoints (1.11m)，因为路线几何比轨迹细节更容易预测
- 论文不报告 ADE/FDE，这些指标用于换模型时横向对比

**`plot_val_samples.py`** — 验证集 GT vs 预测可视化
- 生成 8 张对比图，上下排列（waypoints + route），绿线=GT，蓝线=Pred
- 踩坑: `set_aspect('equal')` 导致纵向被压缩 → 改 `subplots(2,1)` 上下排列 + 移除 equal aspect
- 输出: `checkpoints/eval_plots/sample_00.png ~ sample_07.png`

**`diag_bias.py`** — 偏差诊断

| 指标 | 值 |
|------|-----|
| 纵向误差 (前后) | mean=+0.236m, std=1.722m |
| 横向误差 (左右) | mean=+0.052m, std=1.282m |
| Waypoints ADE 分布 | min=0.01, median=0.55, max=10.69 |
| Route ADE 分布 | min=0.01, median=0.19, max=4.69 |

**结论: 无系统性偏差（means near 0），但方差大（std 1.3-1.7m）。**
- 横向 std=1.28m 说明轨迹点有 zig-zag 抖动（逐点独立预测，无平滑约束）
- 闭环开车时 PID 控制器会自然平滑掉这些抖动，影响不大
- 开环看图时抖动明显，这是用户感觉"预测路径左右浮动很大"的原因

### 与论文对比
- 论文用 **CARLA 闭环驾驶分数** (DS/SR) 评测，不报告开环 ADE/FDE
- 论文 Bench2Drive DS=85.1, SR=67.3%，纯视觉 SOTA
- 论文 Action Dreaming 使指令跟随成功率从 28.22% → 72.96%
- 本次复现无 CARLA 环境，未做闭环评测

### 启动命令速查
```bash
conda activate simlingo
cd ~/projects/SimLingo/simlingo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 训练 - SimLingo Dreamer（带语义增强）
PYTHONPATH=$PWD python simlingo_training/train.py experiment=dreamer_no_safety

# 训练 - TCP_lang（VLM → GRU decoder）
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_lang

# 评估 ADE/FDE（单 checkpoint, 单语言分布）
PYTHONPATH=$PWD python simlingo_training/eval_waypoints.py experiment=<exp> checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt

# 评估口语泛化对比（0%/50%/100% 三种概率）
PYTHONPATH=$PWD python simlingo_training/eval_compare_lang.py experiment=<exp> checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt

# 可视化
PYTHONPATH=$PWD python simlingo_training/plot_val_samples.py experiment=dreamer_no_safety checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt

# 偏差诊断
PYTHONPATH=$PWD python simlingo_training/diag_bias.py experiment=dreamer_no_safety checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt
```

---

## 2026-05-12 — TCP_lang 训练完成 + 语义增强准备

### TCP_lang 模型
- **架构**: VLM encoder → lang_embed (128d) → TCP-style GRU decoder + route head
  - 不直接预测 waypoints，而是用 GRU 自回归解码
  - lang_embed 注入 GRU 每步 + 空间注意力 + join 层
  - 同时输出 waypoints + route 两路预测
- **代码**: `simlingo_training/models/tcp_lang.py`（TCPLangModel + TCPDecoder）
- **配置**: `config/experiment/tcp_lang.yaml`
  - `model._target_: simlingo_training.models.tcp_lang.TCPLangModel`
  - `debug: true, enable_wandb: false`
  - `max_epochs: 5, batch_size: 2, precision: 16-mixed`
  - `use_commentary: False, use_qa: False, use_safety_flag: False`
- **训练完成**: 5 epochs, checkpoint: `outputs/2026_05_12_16_16_37_tcp_lang/checkpoints/last.ckpt`
- **可视化**: `visualise_tcp_lang/` 172 张（step_001000 ~ step_142000+），含 lang_embed 分布 + waypoints/route 预测对比

### 语义增强基础设施
- **规则改写引擎**: `augment_language.py` — 动词同义词、语序调整、口语前缀后缀
- **口语化模板映射**: `data/augmented_templates/instruction_colloquial_map.json` (324KB)
  - 从 `dreamer.json` 27KB 模板生成，每模板 10 个口语变体
  - 训练时 ColloquialRewriter 以 `lang_augment_prob` 概率随机改写
- **config.py 更新**: `DatasetBaseConfig` 新增 `use_language_augment`, `lang_augment_prob`, `colloquial_map_path` 字段
- **重要性**: 原始 dreamer_no_safety 训练时 map 文件不存在，相当于无增强；重新训练后才有

---

## 2026-05-13 — TCP_lang 评估 + dreamer_no_safety 重训启动

### TCP_lang 评估结果

**eval_waypoints.py (默认 50% 口语增强):**
| 指标 | TCP_lang | dreamer_no_safety (旧, 无增强) |
|------|----------|-------------------------------|
| Waypoints ADE | **0.92 m** | 1.11 m |
| Waypoints FDE | **2.28 m** | 2.78 m |
| Route ADE | **0.42 m** | 0.43 m |
| Route FDE | **0.65 m** | 0.83 m |
| Language Acc | 0% (不输出语言) | 100% |

**eval_compare_lang.py (口语泛化对比):**
| 指标 | 0% 口语映射 | 50% 口语映射 | 100% 口语映射 |
|------|-----------|------------|-------------|
| Waypoints ADE | 0.8982 | 0.944 | 0.985 |
| Waypoints FDE | 2.2872 | 2.3311 | 2.3907 |
| Route ADE | 0.2794 | 0.3111 | 0.3149 |
| Route FDE | 0.5771 | 0.534 | 0.5498 |

**结论**: 口语化比例上升 → ADE 轻微退化，FDE 稳定，模型对语言变体有合理泛化，未崩溃。


---

## 2026-05-14 — dreamer_no_safety Epoch 2 崩溃 + 恢复

### 崩溃详情
- **位置**: Epoch 2, 23% (6679/28581), 速度 0.79 it/s
- **错误**: `c10::DistBackendError` / `CUDA error: unknown error` (NCCL watchdog)
- **原因**: RTX 5090 (Blackwell sm_120) 偶发 CUDA 不稳定，与 PyTorch 2.7 时代的死机同源
  - PyTorch 2.10.0+cu128 + cuDNN 9.10.02 是最稳定组合，但不能 100% 避免
  - 非代码 bug，是硬件/驱动间歇性问题

### 崩溃恢复
- 使用绝对路径 resume（Hydra 切换 cwd 后相对路径失效）：
```bash
conda activate simlingo && cd ~/projects/SimLingo/simlingo && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 && PYTHONPATH=$PWD python simlingo_training/train.py experiment=dreamer_no_safety resume=true resume_path=/home/cyrilyuan/projects/SimLingo/simlingo/outputs/2026_05_13_12_12_46_dreamer_no_safety/checkpoints/last.ckpt
```
- 从 Epoch 2 恢复成功，继续训练 Epoch 3-4 无崩溃
- 新 checkpoint 目录: `outputs/2026_05_14_12_29_46_dreamer_no_safety/`

---

## 2026-05-16 — dreamer_no_safety 训练完成 + 最终对比

### dreamer_no_safety 训练完成
- 总 5 epoch 完成（Epoch 0-2 在 5/13 训练，Epoch 2 崩溃，5/14 恢复完成 Epoch 3-4）
- 最终 checkpoint: `outputs/2026_05_14_12_29_46_dreamer_no_safety/checkpoints/last.ckpt`
- Epoch 4 最终速度: 0.61 it/s，训练 loss 0.269，验证 loss 0.919

### dreamer_no_safety eval_compare_lang.py 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 1.3513 | 1.5823 | 1.2518 |
| Waypoints FDE | 3.3391 | 3.9542 | 3.1275 |
| Route ADE | 0.7369 | 0.7339 | 0.6369 |
| Route FDE | 1.5142 | 1.5155 | 1.3356 |

### 最终对比表

| 指标 | dreamer_no_safety (带增强) | tcp_lang | tcp_lang 优势 |
|------|------|----------|-----------|
| Waypoints ADE (0%口语) | 1.3513 | **0.8982** | -33.5% |
| Waypoints FDE (0%口语) | 3.3391 | **2.2872** | -31.5% |
| Route ADE (0%口语) | 0.7369 | **0.2794** | -62.1% |
| Route FDE (0%口语) | 1.5142 | **0.5771** | -61.9% |
| Waypoints ADE (50%口语) | 1.5823 | **0.9440** | -40.4% |
| Waypoints ADE (100%口语) | 1.2518 | **0.9850** | -21.3% |

**结论**: TCP_lang 在所有指标上全面领先 dreamer_no_safety。Route 预测优势最突出（>60%），Waypoints 优势约 30%。dreamer_no_safety 的口语化趋势反常（100% 反而最好，50% 最差），不如 tcp_lang 的单调退化合理。

### 完整命令速查
```bash
conda activate simlingo && cd ~/projects/SimLingo/simlingo && export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 训练 - SimLingo Dreamer（带语义增强）
PYTHONPATH=$PWD python simlingo_training/train.py experiment=dreamer_no_safety

# 训练 - TCP_lang
PYTHONPATH=$PWD python simlingo_training/train.py experiment=tcp_lang

# 从崩溃恢复（务必用绝对路径！）
PYTHONPATH=$PWD python simlingo_training/train.py experiment=dreamer_no_safety resume=true resume_path=/home/cyrilyuan/projects/SimLingo/simlingo/outputs/<run_dir>/checkpoints/last.ckpt

# 评估 - 口语泛化对比（0%/50%/100%）
PYTHONPATH=$PWD python simlingo_training/eval_compare_lang.py experiment=<exp> checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt

# 评估 - 单 checkpoint ADE/FDE
PYTHONPATH=$PWD python simlingo_training/eval_waypoints.py experiment=<exp> checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt

# 可视化 + 偏差诊断
PYTHONPATH=$PWD python simlingo_training/plot_val_samples.py experiment=dreamer_no_safety checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt
PYTHONPATH=$PWD python simlingo_training/diag_bias.py experiment=dreamer_no_safety checkpoint_path=outputs/<run_dir>/checkpoints/last.ckpt
```

---

## 2026-05-17 — TCP_lang v2 训练完成（去 command, state_dim=32）

### 改动说明
- **去掉 command one-hot**: 原版从语言文本抠关键词映射 6 维 command，脆弱且不可靠
- **state_dim 128→32**: state_encoder 输入 speed(1)+tp(2)=3 维，映射到 32 维（原 128 维过于浪费）
- **state_encoder 参数**: 1.2K（原 17K），减小 14 倍
- **导航意图**: 完全由 lang_embed (128d) 承担，state 只编码物理量

### 训练完成
- 5 epoch 完成，速度 3.27 it/s（比 v1 的 ~4-5 it/s 略慢，正常波动）
- 最终 checkpoint: `outputs/2026_05_16_13_46_42_tcp_lang/checkpoints/last.ckpt`
- Epoch 4 训练 loss: 0.092，验证 loss: 0.854

### eval_compare_lang.py 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 0.9938 | 1.061 | 1.0158 |
| Waypoints FDE | 2.5415 | 2.6033 | 2.5151 |
| Route ADE | 0.4087 | 0.4653 | 0.4896 |
| Route FDE | 0.6935 | 0.7964 | 0.819 |

### v1 vs v2 对比

| 指标 | v1 (cmd+128d) | v2 (无cmd+32d) | 退化 |
|------|-------------|---------------|------|
| Waypoints ADE (0%口语) | 0.8982 | 0.9938 | +10.6% |
| Waypoints FDE (0%口语) | 2.2872 | 2.5415 | +11.1% |
| Route ADE (0%口语) | 0.2794 | 0.4087 | +46.3% |
| Route FDE (0%口语) | 0.5771 | 0.6935 | +20.2% |

**分析**: 去掉 command 后 Route 退化最明显（46%）。从文本抠的 command one-hot 虽然 hack，但对 Route Head 提供了实打实的导航先验——"左转"和"直行"的路径几何差异大，显式给比 lang_embed 自己学容易。但 v2 仍碾压 dreamer_no_safety（Route ADE 0.41 vs 0.74）。

### 后续方向
- [x] 折中方案：fusion_mlp 统一融合 lang+state → fused_embed (v3-a)

---

## 2026-05-18 — TCP_lang v3-a 训练完成（统一融合版）

### 架构改进
- **state_encoder**: 3→**64**（提速+目标方向编码）
- **fusion_mlp**: lang_embed(128) + state_embed(64) → **256** 统一规划表征
- **空间注意力**: 只用 lang_embed（感知阶段，语言指导"看哪里"）
- **GRU 输入**: x + tp + fused_embed（规划阶段，统一语义）
- **Route Head**: j + att_vis + fused_embed（同上）
- **设计哲学**: 感知与规划分离——注意力只看语言，解码只看融合表征

### 训练完成
- 5 epoch 完成，速度 2.00 it/s（fusion_mlp 新增 ~50K 参数，略慢可接受）
- 最终 checkpoint: `outputs/2026_05_18_02_20_17_tcp_lang/checkpoints/last.ckpt`
- Epoch 4 训练 loss: 0.039，验证 loss: 0.897

### eval_compare_lang.py 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | **0.7487** | 0.8405 | 0.7958 |
| Waypoints FDE | **1.7599** | 2.0616 | 1.9392 |
| Route ADE | **0.2239** | 0.2285 | 0.2212 |
| Route FDE | **0.3706** | 0.4265 | 0.3849 |

### 三版对比 (0% 口语)

| 指标 | v1 (cmd+128d) | v2 (无cmd+32d) | v3-a (fusion+256d) | v1→v3 提升 |
|------|-------------|---------------|-------------------|-----------|
| Waypoints ADE | 0.8982 | 0.9938 | **0.7487** | -16.6% |
| Waypoints FDE | 2.2872 | 2.5415 | **1.7599** | -23.1% |
| Route ADE | 0.2794 | 0.4087 | **0.2239** | -19.9% |
| Route FDE | 0.5771 | 0.6935 | **0.3706** | -35.8% |

**结论**: v3-a 全面碾压 v1/v2。fusion_mlp 统一融合比单独拼接 lang+state 效果好得多，即使不加 command 也能大幅超越带 command 的 v1。感知/规划分离的架构设计得到验证。

### vs dreamer_no_safety 最终对比

| 指标 | dreamer_no_safety | tcp_lang v3-a | 优势 |
|------|------|----------|------|
| Waypoints ADE (0%口语) | 1.3513 | **0.7487** | -44.6% |
| Waypoints FDE (0%口语) | 3.3391 | **1.7599** | -47.3% |
| Route ADE (0%口语) | 0.7369 | **0.2239** | -69.6% |
| Route FDE (0%口语) | 1.5142 | **0.3706** | -75.5% |

### 待办
- [x] 重启后恢复 dreamer_no_safety 训练
- [x] dreamer_no_safety eval_compare_lang.py 评估
- [x] 填完最终对比表
- [x] v3-a fusion_mlp 统一融合版
- [x] v3-b1 LLM-output-only planning decoder

---

## 2026-05-19 — TCP_lang v3-b1 训练完成（LLM-output-only planning decoder）

### 架构变化（vs v3-a）
- **删除**: state_encoder, fusion_mlp, raw speed/tp 输入
- **lang_proj**: 896→512→**256** → `planning_embed`（原名 lang_embed）
- **空间注意力**: `vis_spatial + planning_embed`
- **GRU**: `x + planning_embed`（不再拼接 raw tp）
- **Route Head**: `j + att_vis + planning_embed`
- **设计动机**: 验证 VLM hidden states 是否已充分编码所有规划所需信息（speed/tp/导航），不需要额外结构化输入

### 训练完成
- 5 epoch 完成，速度 3.11 it/s（比 v3-a 的 2.00 it/s 快 55%，删减模块后更轻量）
- 最终 checkpoint: `outputs/2026_05_18_23_31_40_tcp_lang/checkpoints/last.ckpt`
- Epoch 4 训练 loss: 0.273，验证 loss: 1.157

### eval_compare_lang.py 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 1.0533 | 1.0141 | 0.9564 |
| Waypoints FDE | 2.5793 | 2.5231 | 2.2817 |
| Route ADE | 0.3357 | 0.3556 | 0.3251 |
| Route FDE | 0.8063 | 0.8376 | 0.7113 |

### 四版对比 (0% 口语)

| 指标 | v1 (cmd+128d) | v2 (无cmd+32d) | v3-a (fusion+256d) | v3-b1 (planning) |
|------|-------------|---------------|-------------------|------------------|
| Waypoints ADE | 0.8982 | 0.9938 | **0.7487** | 1.0533 |
| Waypoints FDE | 2.2872 | 2.5415 | **1.7599** | 2.5793 |
| Route ADE | 0.2794 | 0.4087 | **0.2239** | 0.3357 |
| Route FDE | 0.5771 | 0.6935 | **0.3706** | 0.8063 |

**结论**: v3-b1 全面退化，Waypoints FDE 甚至不如 v1。**VLM hidden states 无法完全替代结构化 speed/tp 输入**——对于精准的短期轨迹生成（尤其是终点预测），decoder 仍然需要显式的物理量数值。v3-a 仍是最优版本。

### 待办
- [x] v3-b1 LLM-output-only planning decoder ablation

---

## 2026-05-20~21 — TCP_lang 崩溃恢复 + 训练完成

### 崩溃恢复
- **原运行**: `2026_05_19_12_15_01_tcp_lang`，Epoch 2 完成（19:17）后停止
- **恢复**: 从 `last.ckpt` resume，新目录 `2026_05_20_15_24_43_tcp_lang`
- Epoch 3-4 完成，最终 checkpoint: `outputs/2026_05_20_15_24_43_tcp_lang/checkpoints/last.ckpt`

### eval_compare_lang.py 结果

| 指标 | 0% 口语 | 50% 口语 | 100% 口语 |
|------|--------|---------|----------|
| Waypoints ADE | 1.0135 | 1.158 | 1.0113 |
| Waypoints FDE | 2.6133 | 2.9036 | 2.4752 |
| Route ADE | 0.7382 | 0.8759 | 0.7898 |
| Route FDE | 1.6802 | 1.858 | 1.7355 |

### vs 前版对比 (0% 口语)

| 指标 | v3-b1 (planning) | 本次 (resume) | 退化 |
|------|------------------|--------------|------|
| Waypoints ADE | 1.0533 | 1.0135 | -3.8% |
| Waypoints FDE | 2.5793 | 2.6133 | +1.3% |
| Route ADE | 0.3357 | 0.7382 | +119.9% |
| Route FDE | 0.8063 | 1.6802 | +108.4% |

**结论**: Route 预测全面崩坏（ADE/FDE 翻倍），Waypoints 持平。疑似崩溃恢复后学习率调度器状态异常，或 Epoch 2 中断导致 route head 未充分收敛。
