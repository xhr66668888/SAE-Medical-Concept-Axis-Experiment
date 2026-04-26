# SAE Medical Concept Axis Experiment
[English Version](#english-version) | [中文版](#chinese-version)

<a id="english-version"></a>
## English Version

### 1. Project Overview
This repository contains an experimental implementation inspired by the Anthropic paper [*The Assistant Axis*](https://arxiv.org/abs/2404.XXXX). Instead of targeting the "assistant persona," this project applies the linear representation hypothesis and causal steering methodologies to **factual medical knowledge** (differentiating Type 1 vs Type 2 diabetes medication) using open-source models (Gemma-3). 

### 2. File Structure
* `data/`: Contains the generated contrastive prompt dataset (`diabetes_contrastive_prompts.csv`).
* `sae_med/`: Core utility modules for data processing (`data_utils.py`) and model logic (`model_utils.py`).
* `scripts/`: Implementation of the experimental pipeline.
  * `step1_mvp.py` to `step7_report.py`: Individual stages of the pipeline.
  * `run_xpu_experiment.sh`: The main bash entry-point that chains all steps together.
* `outputs/`: Automatically generated directory storing model outputs, logs, and generated HTML reports.
* `requirements.txt`: Python package dependencies.

### 3. Setup and Execution

**Dependencies Setup:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Running the Experiment:**
You can run the entire pipeline sequence by triggering the main bash script. The bash script isolates each Python step into a separate process to clear VRAM memory between large SAE and Model block loads.
```bash
# Run full sweep and patching with default settings
LAYER_SWEEP=all PATCHING=1 bash scripts/run_xpu_experiment.sh

# Run only a specific layer for a shorter/faster run
LAYER_SWEEP=21 bash scripts/run_xpu_experiment.sh

# Run the 270M parameter model replay experiment
RUN_270M=1 bash scripts/run_xpu_experiment.sh
```

### 4. Experimental Pipeline and Principles
1. **Data Generation**: Generates 480 prompts comparing Type 1 diabetes (Insulin) and Type 2 (Metformin).
2. **Concept Axis Extraction (Difference in Means)**: Records hidden states at specific layers for paired prompts. Subtracts the means to isolate the concept vector.
3. **SAE Feature Candidates**: Projects the axis onto Gemma Scope 2 (`gemma-scope-2-1b-it-res-all`) dictionaries to identify the active interpretable features.
4. **Causal Steering**: Multiplies the extracted axis by an $\alpha$ scalar and injects it back during the forward pass.
5. **Activation Patching**: Pinpoints the causal bottlenecks by reverting tokens at specific layers.

### 5. Final Results Data
The below data captures results from the runs on the **Gemma-3-1B-it** model.

#### A. Concept Axis Sweep (Layer Performance)
Finding the optimal layer where the concept is encoded.

| Layer | Test Acc | Null Mean | DLA    |
|-------|----------|-----------|--------|
| 15    | 0.586    | 0.506     | +0.078 |
| 20    | 0.643    | 0.529     | +0.194 |
| **21**| **0.657**| **0.528** | **+0.192** |
| 22    | 0.629    | 0.529     | +0.182 |

*(For the 270M model, the optimal layer was identified at Layer 12 with a 70.7% accuracy).*

#### B. Causal Steering (Layer 21)
Plotting the mean $\Delta$ logit difference (Type 1 - Type 2) as we adjust the injection scalar ($\alpha$):

| $\alpha$ | $\Delta$ Logit Diff | Interpretation |
|----------|-----------------------|---------------|
| -3.0     | -4.277               | Strongly favors Metformin (Type 2) |
| -2.0     | -2.934               | ... |
| -1.0     | -1.503               | ... |
|  0.0     |  0.000               | Baseline |
| +1.0     | +1.550               | ... |
| +2.0     | +3.100               | ... |
| +3.0     | +4.621               | Strongly favors Insulin (Type 1) |

The strictly monotonic relationship verifies the axis is causal and actionable.

#### C. Activation Patching
Evaluating patching targeting Layer 21, Position -1 (last token):
* **Mean Normalized Score**: 0.9955 (99.55%)
* **95% Bootstrap CI**: [0.9751, 1.0148]

---

<a id="chinese-version"></a>
## 中文版 (Chinese Version)

### 1. 项目概览
本项目是一个实验性质的开发工程。主要参考了 Anthropic 关于特征向量提取的论文（[*The Assistant Axis*](https://arxiv.org/abs/2404.XXXX)）。本实验的设计点在于：没有去提取传统的“助手人格”轴，而是尝试探索大语言模型内部是如何表征**客观事实（医学知识）**的（具体任务为让模型区分一型、二型糖尿病及其用药偏好）。项目基于开源的 Gemma-3 模型与开源大字典特征库（Gemma Scope SAE）进行了因果机理验证。

### 2. 文件目录说明
* `data/`: 存放实验使用的对比提示词数据集（运行管道将自动在此生成 `diabetes_contrastive_prompts.csv`）。
* `sae_med/`: 核心业务逻辑包，包含数据载入的帮助类 (`data_utils.py`) 与操作模型 Hook 的逻辑类 (`model_utils.py`)。
* `scripts/`: 包含执行整套流水线的组件脚本。
  * `step1_mvp.py` 至 `step7_report.py`: 拆分落地的各个验证环节阶段代码。
  * `run_xpu_experiment.sh`: 实验主启动入口，利用 Bash 统一串联调用配置各环节脚本。
* `outputs/`: 脚本运行生成的存放目录，包含模型中间数据、日志统计（json/txt）及本地 HTML 图表。
* `requirements.txt`: Python 核心依赖包目录。

### 3. 环境配置与运行指引

**依赖安装：**
推荐在隔离的虚拟环境中构建：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**执行实验：**
实验管线完全由主命令 Bash 脚本调度。脚本内设定了独立的 Python 进程以避免在多层网络和超大 SAE 分块加载时导致 VRAM 内存泄漏崩溃。
```bash
# 默认行为：执行完整的跨层扫荡寻找最优解，并做 Patching 打分。
LAYER_SWEEP=all PATCHING=1 bash scripts/run_xpu_experiment.sh

# 测试样例：只在特定的单层（如第21层）跑通整个干预管线以节省时间。
LAYER_SWEEP=21 bash scripts/run_xpu_experiment.sh

# 规模对比实验：启动 270M 参数实验模型的对应比照流程。
RUN_270M=1 bash scripts/run_xpu_experiment.sh
```

### 4. 数据管线与运行原理
1. **数据构建阶段**: 构造了 480 条平行病历 Prompt（区分一型胰岛素需求对比。二型二甲双胍需求）。
2. **提取模型概念轴 (Difference in Means)**: 在前向传播时拦截对比组的中间状态层特征值（Residual Stream），并用均值差法求定“知识概念”在这层里的主要偏离向量。
3. **定位细度 SAE 字典**: 对照 Gemma Scope 2 (`gemma-scope-2-1b-it-res-all`) 参数层，检测到底是哪几个具体的特征结点（Feature Nodes）负责处理这段医学事实。
4. **因果干预连续性 (Causal Steering)**: 手动缩放（$\alpha$ 系数）这个提出来的向量轴，直接修改原始张量，并测定推理末端的答案漂移。
5. **激活验证打点 (Activation Patching)**: 指定修复/修改某一个具体字（Token）以及对应单层的激活数据，量化找寻决策被最终拍板形成的逻辑隘口处在哪。

### 5. 关键实验数据简报
下面是基于基座 **Gemma-3-1B-it** 跑出的输出数据快照。

#### A. 寻找最佳激活层 (Concept Axis Sweep)
| 层级 (Layer) | 测试集分类准确率 | 空值基准 (Null) | 直接对数归因 (DLA)    |
|-------|----------|-----------|--------|
| 15    | 0.586    | 0.506     | +0.078 |
| 20    | 0.643    | 0.529     | +0.194 |
| **21**| **0.657**| **0.528** | **+0.192** |
| 22    | 0.629    | 0.529     | +0.182 |

*(对比控制组实验中，我们在跑 270M 参数模型时寻找到的最优层发生在 Layer 12，准确率为 70.7%)*

#### B. 数据强制漂移验证 (Causal Steering @ Layer 21)
调节概念控制注入变量 ($\alpha$) 时，观察模型最终在分类层的偏移 ($\Delta$ Logit Diff: 倾向一型药物 vs 倾向二型药物)：

| 因数 $\alpha$ | $\Delta$ Logit 绝对差 | 行为解释 |
|-------------|-----------------------|---------------|
| -3.0        | -4.277               | 极度判定应当开二甲双胍 |
| -2.0        | -2.934               | - |
| -1.0        | -1.503               | - |
|  0.0        |  0.000               | Baseline（无干预） |
| +1.0        | +1.550               | - |
| +2.0        | +3.100               | - |
| +3.0        | +4.621               | 极度判定应当开胰岛素 |

该数据呈现标准的严格单调递变特性（Monotone-increasing）。证明特征轴具备完全的因果效应调控能力，而非偶然相关。

#### C. 数据点覆盖修补打分 (Activation Patching)
针对目标发生点位 (Layer 21, 倒数第一个输入词 Position -1) 的定点置换：
* **归一化修复得分 (Mean Normalized Score)**: 0.9955
* **95% 抽样置信度 (Bootstrap CI)**: [0.9751, 1.0148]

这证明大模型把处理这类知识的信息处理核心彻底压制在了这单一结点上。
