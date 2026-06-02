# SimLingo Action Dreaming (无安全模式) 复现指南

## 一、项目整体架构

```
SimLingo/
├── simlingo/                          # 主项目代码
│   ├── simlingo_training/             # ★ 完整 SimLingo 训练代码
│   │   ├── train.py                   #    训练入口 (Hydra + PyTorch Lightning + DeepSpeed)
│   │   ├── config.py                  #    配置 dataclass 定义
│   │   ├── eval.py                    #    语言评估入口
│   │   ├── eval_metrics.py            #    评估指标计算
│   │   ├── models/
│   │   │   └── driving.py            #    ★ 主模型 (InternVL2 VLM + LoRA LLM)
│   │   ├── dataloader/
│   │   │   ├── datamodule.py          #    ★ DataModule (50% driving + 50% dreamer)
│   │   │   ├── dataset_dreamer.py     #    ★ Dreamer 数据集加载 + 安全逻辑
│   │   │   ├── dataset_driving.py     #    驾驶数据集加载
│   │   │   └── dataset_base.py        #    基类 (图像/测量/路由加载)
│   │   └── config/
│   │       ├── config.yaml            #    根配置
│   │       ├── experiment/            #    实验配置
│   │       │   ├── debug.yaml         #    调试模式 (1 GPU)
│   │       │   └── simlingo_seed1.yaml#    正式训练 (8 GPU)
│   │       └── data_module/           #    数据模块配置
│   │           └── carla_bucket_v12_dreamer.yaml  # Dreamer 数据分区
│   ├── dataset_generation/
│   │   └── dreamer_data/              # ★ Dreamer 数据生成
│   │       ├── dreamer_generator.py   #    替代轨迹生成 (运动学自行车模型 + PID)
│   │       ├── dreamer_instructions.py#    语言指令 + 安全标签生成
│   │       └── dreamer_utils.py       #    工具函数
│   ├── simlingo_base_training/        # SimLingo-Base (无语言能力的驾驶模型)
│   ├── team_code/                     # CARLA 闭环 agent
│   └── data/
│       └── augmented_templates/
│           └── dreamer.json           # Dreamer 语言指令模板
│
└── simlingo_dreamer/                  # ★ 预生成的 Dreamer 数据 (压缩包)
    ├── process_data.py                #    数据处理/统计脚本
    ├── dreamer_7_category_summary.csv #    7类数据汇总统计
    ├── dreamer_*_training_*.tar.gz    #    训练集 (25个文件)
    └── dreamer_*_validation_*.tar.gz  #    验证集 (15个文件)
```

---

## 二、Action Dreaming 核心原理

### 2.1 什么是 Action Dreaming？

Action Dreaming 是 SimLingo 的核心创新之一。它通过**语言指令 → 替代未来轨迹**的映射来训练模型，使得 VLA 模型能根据自然语言指令调整驾驶行为。

**具体流程：**
1. 从专家驾驶数据集中取一帧（RGB图像 + 当前状态）
2. 使用运动学自行车模型 + PID 控制器，生成多条**替代未来轨迹**（不是专家实际走的）
3. 为每条替代轨迹生成对应的自然语言指令
4. 判断每条指令是否安全（`safe_to_execute`）
5. 训练模型：给定图像 + 指令 → 预测对应的轨迹和文本回答

### 2.2 六种指令模式

| 模式 | 说明 | 示例指令 |
|------|------|----------|
| `target_speed` | 调整到目标速度 | "Drive at 8.3 m/s." |
| `stop` | 停车 | "Stop the vehicle." |
| `faster` | 加速 | "Drive faster." |
| `slower` | 减速 | "Drive slower." |
| `lane_change` | 变道 | "Shift one lane to the left." |
| `crash` | 碰撞（纯 unsafe） | "Crash into the construction cone." |

### 2.3 安全判定逻辑 (`safe_to_execute`)

在 `dreamer_instructions.py` 的 `get_info()` 函数中：

- **Crash 模式** → 永远 `False`
- **碰撞到动态 agent** → `False`
- **超速** (speed > limit) → `False`
- **速度过低** (target_speed 不合理) → `False`
- **行人附近 + 加速** → `False`
- **变道到对向车道/人行道** → `False`
- 其余情况 → `True`

**Dreamer 数据统计（共 ~2100 万条 control samples）：**
- ~55% `safe_to_execute = True`
- ~45% `safe_to_execute = False`

---

## 三、`use_safety_flag` 的关键区别 ★ 重点

这是**你要的核心**。在 `dataset_dreamer.py` 中：

### 3.1 `use_safety_flag = True`（论文原版，考虑安全）

```python
# dataset_dreamer.py 第 53-59 行
if self.use_safety_flag:
    if random.random() < 0.5:
        activate_safety = True    # 50% 概率进入安全模式
    else:
        activate_safety = False   # 50% 概率进入指令跟随模式
```

- **安全模式 `<SAFETY>`**：如果指令是 unsafe 的（`safe_to_execute == False`），模型应拒绝执行，输出 `"Ignore instruction as it leads to a crash. Waypoints:"` 并恢复为专家轨迹
- **指令跟随 `<INSTRUCTION_FOLLOWING>`**：无条件执行指令，即使 unsafe

### 3.2 `use_safety_flag = False`（你要的版本，不考虑安全）

```python
else:
    activate_safety = None  # 无安全/指令跟随区分
```

- **无 `<SAFETY>` / `<INSTRUCTION_FOLLOWING>` 前缀**
- **不检查 `safe_to_execute`**，永远不会拒绝指令
- **模型始终跟随指令**（无论指令安全与否）
- 回答始终是 `"Following the given instruction. Waypoints:"`

**简言之：`use_safety_flag = False` 就是"不管安全，说了就做"的模式。**

### 3.3 配置位置

| 文件 | 配置项 | 默认值 |
|------|--------|--------|
| `simlingo_training/config.py:69` | `use_safety_flag: bool = False` | **默认 False** |
| `config/experiment/debug.yaml:40` | `use_safety_flag: True` | 覆写为 True |
| `config/experiment/simlingo_seed1.yaml:40` | `use_safety_flag: True` | 覆写为 True |
| `config/experiment/simlingo_seed2.yaml:40` | `use_safety_flag: True` | 覆写为 True |

---

## 四、训练流程详解

### 4.1 训练数据组成

训练使用 **50% driving 数据 + 50% dreamer 数据** 混合（`datamodule.py:96-97`）：

```
weights_driving = 0.5
weights_dreamer = 1 - weights_driving  # = 0.5
```

- **Driving 数据**：专家轨迹 + VQA 问答 + Commentary 行为描述
- **Dreamer 数据**：替代轨迹 + 语言指令 + 安全标签

### 4.2 模型架构

```
输入: RGB 前视图 (1024×384) + 语言 Prompt
  │
  ▼
InternVL2-1B Vision Encoder (冻结)
  │
  ▼
视觉特征 + 文本 token → LLM (InternVL2-1B, LoRA微调)
  │
  ├──► Language Adaptor → 文本输出 (回答)
  │
  └──► Driving Adaptor → 轨迹输出 (10个未来waypoints + 路径)
```

### 4.3 损失函数

三个 Loss（在 `driving.py` 中）：
1. **Waypoint ADE Loss**：预测轨迹点与 GT 轨迹点的平均距离误差
2. **Route Loss**：预测路径与 GT 路径的误差
3. **Language NTP Loss**：文本的 Next-Token-Prediction 交叉熵损失

### 4.4 训练超参数

| 参数 | debug 模式 | 正式训练 |
|------|-----------|---------|
| GPU 数量 | 1 | 8 |
| Batch Size | 2 | 6 (per GPU) |
| 学习率 | 3e-5 | 3e-5 |
| 优化器 | AdamW | AdamW |
| 调度器 | OneCycleLR | OneCycleLR |
| 梯度裁剪 | 0.3 | 0.3 |
| 精度 | 16-mixed | 16-mixed |
| 策略 | DeepSpeed Stage 2 | DeepSpeed Stage 2 |
| Epochs | 15 | 15 |
| 验证频率 | 每 1 epoch | 每 2 epoch |
| LoRA alpha | 64 | 64 |
| LoRA r | 32 | 32 |
| 预测长度 | 11 frames (2.75s) | 11 frames |

---

## 五、逐步骤复现指南

### 前提条件
- Linux 系统 (Ubuntu 22.04)
- NVIDIA GPU (至少 24GB 显存，推荐 A100/4090)
- CUDA 11.8+
- Conda 环境管理

### Step 1: 环境搭建

```bash
cd ~/projects/SimLingo/simlingo

# 安装 CARLA 0.9.15
chmod +x setup_carla.sh
./setup_carla.sh

# 创建 conda 环境
conda env create -f environment.yaml
conda activate simlingo

# 安装 PyTorch (确保 CUDA 版本匹配)
pip install torch==2.2.0

# 安装 flash-attention
pip install flash-attn==2.7.0.post2

# 设置环境变量
export CARLA_ROOT=/path/to/CARLA/root
export WORK_DIR=$(pwd)
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}
```

### Step 2: 下载数据集

```bash
# 从 HuggingFace 下载完整数据集
cd ~/projects/SimLingo/simlingo
git clone https://huggingface.co/datasets/RenzKa/simlingo database
cd database
git lfs pull

# 解压 dreaming 数据
mkdir -p database/simlingo
cd ../simlingo_dreamer
for file in *.tar.gz; do
    echo "Extracting $file..."
    tar -xzf "$file" -C ../simlingo/database/simlingo/
done
```

### Step 3: 修改配置（关键 — 关闭安全模式）

编辑 `simlingo_training/config/experiment/debug.yaml`（或创建新配置文件）：

```yaml
# simlingo_training/config/experiment/dreamer_no_safety.yaml
# @package _global_
defaults:
  - /data_module: carla_no_buckets    # 或 carla_bucket_v12_dreamer
  - /model/vision_model: vlm
  - /data_module/base_dataset: dataset

model:
  lr: 3e-5
  predict_route_as_wps: True
  speed_wps_mode: 2d
  language_model:
    variant: 'OpenGVLab/InternVL2-1B'
    lora: True
    lora_alpha: 64
    lora_r: 32
    lora_dropout: 0.1
  vision_model:
    variant: 'OpenGVLab/InternVL2-1B'

data_module:
  batch_size: 2
  num_workers: 4
  base_dataset:
    data_path: database/simlingo
    bucket_path: database/bucketsv2_simlingo
    pred_len: 11
    cut_bottom_quarter: True
    use_commentary: True    # 是否用 commentary 数据
    use_qa: True            # 是否用 VQA 数据
    qa_augmentation: True
    img_shift_augmentation: True
    img_shift_augmentation_prob: 0.5
    hist_len: 1
    route_as: target_point_command
    use_lmdrive_commands: True
    use_old_towns: True
    use_town13: True
    use_safety_flag: False    # ★ 关闭安全模式

max_epochs: 15
val_every_n_epochs: 1
gpus: 1
seed: 42
name: dreamer_no_safety
```

### Step 4: 启动训练

```bash
# 在 simlingo 目录下执行
cd ~/projects/SimLingo/simlingo

# 使用 Hydra 启动训练（会加载 config.yaml → experiment/dreamer_no_safety.yaml）
python simlingo_training/train.py \
  experiment=dreamer_no_safety \
  data_module=carla_bucket_v12_dreamer \
  gpus=1 \
  name=dreamer_no_safety
```

或者直接修改 `config/config.yaml` 的 defaults：
```yaml
defaults:
  - train_base
  - model/language_model: llm
  - experiment: dreamer_no_safety    # 改为你的配置文件
```

### Step 5: 评估 Dreamer 性能

```bash
# 修改 eval.py 中的参数
# eval_mode = "Dreaming"
# load_path = '/path/to/your/checkpoint'

python simlingo_training/eval.py
```

评估指标（在 `driving.py` 的 `on_predict_epoch_end` 中计算）：
- **各模式成功率**：stop, slower, faster, target_speed, lane_change, crash
- **路径 ADE**（平均距离误差）
- 结果保存为 JSON

### Step 6: 查看结果

```bash
# 语言评估指标
python simlingo_training/eval_metrics.py
```

---

## 六、误差分析

### 6.1 Waypoint 预测误差

ADE (Average Displacement Error) 衡量预测轨迹点与 GT 轨迹点的欧氏距离：

```
ADE = mean(||pred_wp[i] - gt_wp[i]||)
```

### 6.2 Dreamer 各模式成功率判定标准

| 模式 | 成功标准 |
|------|----------|
| **stop** | 预测速度在轨迹末端 < 0.1 m/s |
| **slower** | 线性回归斜率 < -0.05 × 当前速度 |
| **faster** | 线性回归斜率 > 0.05 × 当前速度 |
| **target_speed** | 末端速度在目标速度 ±20% 内 |
| **lane_change** | 预测路径末端到指令路径距离 < 到专家路径距离 |
| **crash** | 预测路径接近指令路径（ADE_instruction < ADE_expert） |

### 6.3 Language 评估指标

- **Accuracy**: 精确匹配率
- **GPT Eval**: 使用 GPT-4o 语义相似度评分
- **BLEU / ROUGE_L / METEOR / SPICE / CIDEr**: 标准 NLP 指标

### 6.4 常见问题

1. **数据路径错误**：`data_path` 和 `bucket_path` 必须指向正确位置
2. **显存不足**：减小 `batch_size` 或使用 gradient accumulation
3. **模型未收敛**：检查 `use_safety_flag: False` 是否正确设置，学习率是否合适
4. **Dreamer 数据缺失**：`carla_bucket_v12_dreamer` 中 `train_partitions_dreamer: {all: 1.0}` 需要 dreamer 数据存在

---

## 七、关键代码路径速查

| 你要改/看什么 | 文件路径 |
|-------------|----------|
| **关闭安全模式** | `simlingo_training/config.py:69` (`use_safety_flag: bool = False`) |
| **安全逻辑实现** | `simlingo_training/dataloader/dataset_dreamer.py:53-162` |
| **训练入口** | `simlingo_training/train.py` |
| **模型定义/损失** | `simlingo_training/models/driving.py` |
| **数据混合** | `simlingo_training/dataloader/datamodule.py:96-97` |
| **Dreamer 数据生成** | `dataset_generation/dreamer_data/dreamer_generator.py` |
| **安全标签生成** | `dataset_generation/dreamer_data/dreamer_instructions.py` |
| **指令模板** | `data/augmented_templates/dreamer.json` |
| **评估逻辑** | `simlingo_training/models/driving.py:344-705` |
| **评估入口** | `simlingo_training/eval.py` |
| **评估指标** | `simlingo_training/eval_metrics.py` |

---

## 八、快速启动检查清单

- [ ] Conda 环境 `simlingo` 已创建并激活
- [ ] CARLA 0.9.15 已下载
- [ ] 数据集已解压到 `database/simlingo/`
- [ ] Dreamer 数据已解压到 `database/simlingo/`
- [ ] 配置文件中 `use_safety_flag: False`
- [ ] 数据路径 `data_path` 和 `bucket_path` 正确
- [ ] Wandb 登录（如需要）
- [ ] GPU 可用（`nvidia-smi` 确认）
- [ ] 启动训练并监控 `train/loss` 下降
