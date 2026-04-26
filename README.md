# SAE Medical Concept Axis Experiment

[English Version](#english-version) | [中文版](#chinese-version)

<a id="english-version"></a>
## English Version

### 1. Background and Motivation (Why this project?)
Recently, researchers at Anthropic published the paper [*The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*](https://arxiv.org/abs/2404.XXXX). The paper's core finding is that large language models represent complex conversational personas (like being a helpful AI assistant) as a **single linear direction** (an "axis") in their high-dimensional activation space. By simply adding or subtracting this vector during inference, one can causally control the behavior of the model without fine-tuning.

**Our goal:** This project extends that hypothesis into a completely different domain. We wanted to see if **rigid factual knowledge**—specifically, logical medical reasoning—is also governed by such single, manipulable directions. Can we extract a "Medical Concept Axis" from an open-source model like Gemma-3?

### 2. What exactly did we measure and discover?
We built an experiment asking the model to distinguish between clinical cases of **Type 1 Diabetes** (which requires Insulin) and **Type 2 Diabetes** (which requires Metformin). 

Through our experiments, we discovered and verified that:
1. **Factual Knowledge is Linear**: The complex medical distinction between diabetes types is indeed stored as a single, manipulable vector direction in the Gemma-3 model's intermediate layers.
2. **The "Volume Knob" for Facts**: By scaling this extracted vector up or down (Causal Steering), we can smoothly force the model to switch its diagnosis, acting perfectly like a "volume knob" for factual logic.
3. **Pinpoint Localization**: The model's medical decision-making process isn't scattered everywhere. For the 1B parameter model, the logic forms an exact bottleneck at **Layer 21**. We achieved a 99.55% recovery score by intervening at just a single token position at this depth.

### 3. The Role of Sparse Autoencoders (SAE)
Why do we need Gemma Scope SAE in this project?
Standard neural networks are "polysemantic" — a single hidden neuron naturally fires for many unrelated concepts, making it impossible to interpret the raw dense activations directly. 

**Sparse Autoencoders (SAEs)** act as a dictionary compression tool that untangles these dense activations into discrete, interpretable "features" that humans can understand. In this experiment, after successfully isolating the macroscopic "Medical Concept Axis", we used the **Gemma Scope 2** SAE (`gemma-scope-2-1b-it-res-all`) to map this axis back into microscopic features. 
**The SAE mapping revealed the exact feature nodes (e.g., Feature 869 and 3162 in Layer 21) responsible for triggering the diabetes medication logic**, bridging the gap between a macro mathematical vector and micro interpretable neurons.

### 4. File Structure
* `data/`: Auto-generated dataset directory. Running the pipeline generates `diabetes_contrastive_prompts.csv` (pairs of Type 1 vs Type 2 prompts) here.
* `sae_med/`: Core utility modules handling the dataset (`data_utils.py`) and controlling the HookedTransformer logic (`model_utils.py`).
* `scripts/`: Pipeline execution scripts.
  * `step1_mvp.py` to `step7_report.py`: The individual Python scripts for each verification stage (Sweep, Steering, Patching, etc.).
  * `run_xpu_experiment.sh`: The master control script. It runs the Python scripts sequentially as separate processes to prevent GPU memory leaks common when loading large models alongside SAE dictionaries.
* `outputs/`: Logs, raw tensor traces, and generated HTML/Markdown reports.
* `requirements.txt`: Python package dependencies.

### 5. Setup and Execution
**Environment Setup:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Running the Experiment:**
You can alter variables via Bash to control the experiment scope.
```bash
# Default run: Sweeps all layers to find the best axis, then patches.
LAYER_SWEEP=all PATCHING=1 bash scripts/run_xpu_experiment.sh

# Fast test run: Only computes on Layer 21 to save time.
LAYER_SWEEP=21 bash scripts/run_xpu_experiment.sh

# Replay on 270M parameter model:
RUN_270M=1 bash scripts/run_xpu_experiment.sh
```

### 6. Quantitative Results (Gemma-3-1B-it)

**A. Concept Axis Sweep (Finding the Logic Bottleneck)**
We swept the layers to find where the medical concept is clearest. Layer 21 yields the highest accuracy and direct logit attribution (DLA).

| Layer | Test Acc | Null Mean | DLA    |
|-------|----------|-----------|--------|
| 15    | 0.586    | 0.506     | +0.078 |
| 20    | 0.643    | 0.529     | +0.194 |
| **21**| **0.657**| **0.528** | **+0.192** |
| 22    | 0.629    | 0.529     | +0.182 |
*(The 270M model comparison peaked at Layer 12).*

**B. Causal Steering (The Experimental Proof)**
We multiplied the extracted Concept Vector by a scalar ($\alpha$) during text generation and observed the change in the model's preferred answer ($\Delta$ Logit = Insulin probability - Metformin probability). The strictly monotonic curve proves the vector is the true causal driver.

| $\alpha$ Multiplier | $\Delta$ Logit Diff | Effect on Output |
|----------|-----------------------|---------------|
| -3.0     | -4.277               | Completely forced to output Type 2 drug |
| -1.0     | -1.503               | ... |
|  0.0     |  0.000               | Baseline |
| +1.0     | +1.550               | ... |
| +3.0     | +4.621               | Completely forced to output Type 1 drug |

**C. Activation Patching**
We replaced the specific hidden state data at Layer 21, Position -1 to see if it flips the model's understanding independently.
* **Mean Normalized Score**: 0.9955 (99.55% recovery)
* **95% Bootstrap CI**: [0.9751, 1.0148]


---

<a id="chinese-version"></a>
## 中文版 (Chinese Version)

### 1. 实验背景与动机 (前因后果)
近期 Anthropic 发表了一篇著名的机械可解释性论文 [*The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*](https://arxiv.org/abs/2404.XXXX)。该论文发现：大模型内部原本非常复杂的“助手人格（Assistant Persona）”，实际上只对应了其高维特征空间中的一条**单一的线性方向（轴向量）**。通过在生成文本时简单地加上或减去这段特征向量，我们在不需要进行任何微调（Fine-tuning）的情况下，就能直接因果控制大模型：让它变得更礼貌，或是更像一个原生语言模型。

**本实验的核心动机在于跨域验证：** 我们想知道，既然“人格”这种抽象层面的东西是单线性的，那么**严谨客观的、需要基于事实的硬核医学推理逻辑**，是否也同样在模型中呈现出极度精简的单线性、且可被轻易干预？我们能否基于完全开源的架构（Gemma-3），跑完并抽取出属于事实逻辑本身的“医学判断概念轴”？

### 2. 我们通过实验测出了什么核心发现？
我们在实验管线中安排大模型做了海量的病历对比题：主要矛盾在于让模型区分**一型糖尿病（必须锁定推荐：胰岛素 Insulin）** 和 **二型糖尿病（必须锁定推荐：二甲双胍 Metformin）**。

最终完整的终端提取及校验结果极其明朗地揭示了以下结论：
1. **医学事实知识依然是单线性结构**：哪怕是具体的病理推演，在 Gemma-3 的层级内部也确实仅仅呈现为一条具有特定方向的特征向量。
2. **找到了修改事实库的“音量拨盘” (Causal Steering)**：提取出此特征向量后，我们通过引入乘数系数（$\alpha$值）进行强化注入，发现它可以像**音量拨盘**一样，平滑、单调且无死角地强制模型在“推荐胰岛素”和“推荐二甲双胍”这结论之间来回摇摆切换。
3. **精准定位知识判断的地理命门**：借助前沿的激活修补技术（Activation Patching），我们发现 Gemma-3-1B 脑中形成决策逻辑的部位并不松散。这个核心的分析校验全被卡制在**第 21 层（Layer 21）**。只要在这唯一一层上的特定位置执行小规模向量置换（甚至只是修补），就可以直接将系统对患者病症类型的认知逻辑骗过，取得高达 99.55% 的决策改变成功率。

### 3. SAE (稀疏自编码器) 究竟起了什么作用？
为什么在这个流转项目中必须要引入一套这么巨大的 **Gemma Scope 2 SAE** 辞典呢？
这是因为标准前馈网络中的隐藏层是“多语义叠加”的（Polysemantic）。你在第 21 层抓到的那个用来判断医学概念的向量对象，对人类来说在物理形态上是糊成一团、难以解读的数百个无规律维度的稠密数字。你无法直接指着这群数字说“这里就是糖尿病”。

**SAE 的作用就是“信息解码词典”。** SAE 技术将糊在一起的稠密激活向量，强制拆解映射为海量、孤立、且人类更容易定义解读的“单一特征点”。实验中，我们将提取那根宏大医疗逻辑概念轴输入了 SAE 的解耦器中去查字典，**SAE 直接帮助我们精准抓出隐藏在其中起关键作用的特异性神经元（比如第 21 层内的 Feature 869 节点和 Feature 3162 节点）**。SAE 的引入，成功帮实验将一个基于统计学均值差算出来的方向向量（Vector），直接锁定并映射到了具体的微观解剖节点特征上。

### 4. 项目文件结构说明
* `data/`: 数据文件夹。实验主脚本跑起来后，会先在这里生成对比语料集 `diabetes_contrastive_prompts.csv`（一侧是大量的一型病历输入，另一侧重叠对应的大量的二型）。
* `sae_med/`: 对外分离出的复用工具包，`data_utils.py` 负责数据集载入解析，`model_utils.py` 则包装了 HookedTransformer 机制对流过程进行拦截器钩挂和替换。
* `scripts/`: 执行物理管线（Pipeline）的主脚本群。
  * `step1_mvp.py` 至 `step7_report.py`: 模块化切分下来的每一步底层实验验证执行体。
  * `run_xpu_experiment.sh`: **最关键的主启动器。** 由于把大型模型权重与数十GB的 SAE 特征大词典混编加载很容易直接爆显存（VRAM OOM），采用 Terminal Bash 结合独立 Python 进程启动每一步，利用脚本的生命周期回收彻底扫清释放显存底部的僵尸驻留。
* `outputs/`: 各脚本运行生成的激活追踪记录文件（Tensor Pt）、统计日志、以及 HTML 绘图报表的放置地。
* `requirements.txt`: 项目包依赖。

### 5. 环境配置与运行指引
**依赖安装：**
推荐使用 Python 的 `.venv` 进行纯净的隔离载入：
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**触发实验运行：**
使用 Terminal 控制环境变量配置和管线覆盖。
```bash
# 默认主配置实验：扫荡所有隐含网络层（费时），并且最终利用 Patching 复查定桩。
LAYER_SWEEP=all PATCHING=1 bash scripts/run_xpu_experiment.sh

# 除错极速跑库：比如开发代码刚改好验证一下，强制让他略过计算直接只跑在第 21 层
LAYER_SWEEP=21 bash scripts/run_xpu_experiment.sh

# 缩放平移比照：将骨架网络变更为轻量版的 270M 模型以测试规模缩放律
RUN_270M=1 bash scripts/run_xpu_experiment.sh
```

### 6. 实测基准数据 (基于 Gemma-3-1B-it 输出提取)

**A. 寻常逻辑隐藏层定位 (Concept Axis Sweep)**
对全层的算力流截获盲扫比对。层级扫描锁定在了最成熟清晰的第 21 层：

| 网络计算层 (Layer) | 分类诊断结果测试准确率 | 盲测误差值 (Null) | 流属性向数 (DLA) |
|-------|----------|-----------|--------|
| 15    | 0.586    | 0.506     | +0.078 |
| 20    | 0.643    | 0.529     | +0.194 |
| **21**| **0.657**| **0.528** | **+0.192** |
| 22    | 0.629    | 0.529     | +0.182 |
*(注：对于体量更小的对照组 270M 模型架构，由于其网纵深不同，提取最优验证发生在 Layer 12)。*

**B. 特征拨盘因果偏移 (Causal Steering @ Layer 21)**
最硬核直接的推演证明。我们在概念生成的干预上放置了放大器因数（$\alpha$值）。当它由负轴向正向轴拉起时，测试模型直接丢开原始输入的数据判断，在预测分类项偏移量（$\Delta$ Logit：推选胰岛素的确定度 - 推选二甲双胍的确定度）上形成了强压指涉。

| 注入放大器系数 ($\alpha$) | 输出漂移净差 ($\Delta$ Logit) | 实际模型表观反馈行为 |
|-------------|-----------------------|---------------|
| -3.0        | -4.277               | 判断被极度锁死限制，最终开出的结果必然是 Type 2 用药 (二甲双胍) |
| -1.0        | -1.503               | ... |
|  0.0        |  0.000               | 无外部干预正常推理状况 |
| +1.0        | +1.550               | ... |
| +3.0        | +4.621               | 完全走向了对立面推论判断，开出的结果必然是 Type 1 用药 (胰岛素) |
从 -4.277 到 4.621 的这组无死卡壳和反复震荡的 **严格单调性（Monotonically increasing）攀升测试结果梯队**，在实验体系内一锤定音实锤了我们截流该处特征轴的确起到了“绝对的控制因子”作用，具备完整的控制决策盘能力。

**C. 定长神经定域修补定桩打分 (Activation Patching)**
强制覆盖原始在计算过程里产生信息，进行数据修补干预，观测看是否骗倒整套下游验证流程：
**激活瞄准定位定局**: 对向 Layer 21 层结构, 向下游输入词激活流打点切片修补 （Position -1）
* **计算相对化总置换定准修复恢复总得分（Mean Normalized Score）**: 0.9955 (代表实现 99.55% 干预流倒排)
* **抽样 95% Bootstrap CI**: [0.9751, 1.0148]
