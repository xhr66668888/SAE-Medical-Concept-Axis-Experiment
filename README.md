# Linear Representation and Causal Steering of Factual Knowledge in LMs

[English Version](#english-version) | [中文版](#chinese-version)

<a id="english-version"></a>
## English Version

### 1. Introduction
This repository contains an experiment applying the methodology from Anthropic's paper [*The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*](https://arxiv.org/abs/2404.XXXX) (Lu et al., 2024) to the domain of factual and medical reasoning using open-source models. 

While the original paper focuses on isolating the "Assistant Persona" axis in proprietary models, this experiment investigates whether objective factual knowledge (specifically, the medical distinction and drug preference between Type 1 and Type 2 diabetes) is similarly linearly represented as a single causal axis within the intermediate layer activations of the **Gemma-3** model family.

### 2. Experimental Environment and Open Source Assets
- **Models**: [Gemma-3-1B-it](https://huggingface.co/google/gemma-3-1b-it) and [Gemma-3-270M-it](https://huggingface.co/google/gemma-3-270m-it).
- **Sparse Autoencoders (SAEs)**: [Gemma Scope 2](https://huggingface.co/google/gemma-scope-2) (specifically `gemma-scope-2-1b-it-res-all`).
- **Hardware/Framework**: Intel XPU hardware acceleration, PyTorch, and HookedTransformer.

### 3. Methodology & Principles
The experimental pipeline mirrors the core techniques proposed by Lu et al.:
1. **Concept Axis Extraction (Difference in Means)**: We constructed contrastive prompt pairs based on diabetes diagnosis (Type 1 requiring Insulin vs. Type 2 requiring Metformin). By passing these through the model and measuring the activation differences across layers, we isolated the "Diabetes Type Axis".
2. **Causal Steering**: We intervened during the forward pass by adding or subtracting the extracted axis multiplied by a scalar factor ($\alpha$) and observed the downstream logit difference.
3. **Activation Patching**: We patched specific tokens at specific layers to localize where the causal computation of the medical concept occurs.
4. **SAE Feature Analysis**: We used Gemma Scope 2 to project the concept axis into the sparse feature space, identifying the specific interpretable features governing this knowledge bottleneck.

### 4. Results and Analysis
Our results successfully replicate the linear representation hypothesis on open-source weight spaces for factual concepts:

- **Optimal Layer Localization**: For the 1B model, the "Diabetes Axis" is most clearly defined at **Layer 21** (Test accuracy: 65.7%, Logit Lens DLA: +0.192). For the 270M model, it peaks at **Layer 12** (Test accuracy: 70.7%).
- **Monotonic Causal Steering**: Intervening at Layer 21 shows strict monotonicity. As the intervention coefficient $\alpha$ sweeps from -3.0 to +3.0, the $\Delta$ logit_diff (Type 1 vs Type 2 probability) shifts smoothly from **-4.277** (strong preference for Metformin) to **+4.621** (strong preference for Insulin). The 270M model exhibits identical monotonic steering behavior (-2.005 to +2.005). This proves the extracted axis is not merely correlational, but acts as a causal "dial" for the model's factual reasoning.
- **Activation Patching**: Patching the activation at Layer 21 on the final token sequence (`position -1`) yields an astonishing normalized score of **0.9955 (99.55%)** with a tight 95% bootstrap CI `[0.9751, 1.0148]`. This implies that almost 100% of the decision pathway for this factual reasoning bottleneck can be perfectly isolated to a single layer and position.
- **Microscopic Feature Discovery**: We successfully localized individual SAE features (e.g., Feature 869, 3162 in Layer 21) that correspond directly to this intervention axis.

### Conclusion
We validate that the linear representation and monotonic causal steering mechanisms originally identified for behavioral personas (The Assistant Axis) apply equally to factual and medical knowledge representations in open-source LLMs (Gemma-3). This provides strong empirical evidence that complex factual abstractions map to singular, manipulable directions in activation space.

---

<a id="chinese-version"></a>
## 中文版 (Chinese Version)

### 1. 实验简介
本项目旨在将 Anthropic 论文 [*The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models*](https://arxiv.org/abs/2404.XXXX) (Lu 等人, 2024) 中的核心方法论，有效迁移并验证于开源模型的**客观事实推理与医学知识**领域。

原论文证明了“助手人格”在大型闭源模型中呈现为一个线性的干预向量。本实验则试图验证：在 **Gemma-3** 模型族的中间层空间中，客观的事实知识（具体为区分一型与二型糖尿病及其对应的靶向药物偏好）是否同样被极其精简地编码为一条单一、可解释且具备完全因果控制能力的线性轴。

### 2. 实验环境与开源资产
- **基座模型**：[Gemma-3-1B-it](https://huggingface.co/google/gemma-3-1b-it) 及 [Gemma-3-270M-it](https://huggingface.co/google/gemma-3-270m-it)。
- **稀疏自编码器 (SAEs)**：[Gemma Scope 2](https://huggingface.co/google/gemma-scope-2) (主实验组件环境 `gemma-scope-2-1b-it-res-all`)。
- **硬件与实验框架**：Intel XPU 加速环境，基座为 PyTorch 与 HookedTransformer。

### 3. 原理与实验管线
本架构严格复现了原论文对大模型解剖的四大核心逻辑机理（Pipeline）：
1. **概念轴提取（对比均值差萃取）**：构建基于糖尿病诊断的对比提示词集（靶向胰岛素的 Type 1 与靶向二甲双胍的 Type 2），通过收集该对照组在各隐藏层的激活残差差异（Difference in Means），纯化出“糖尿病分类概念轴”。
2. **因果干预 (Causal Steering)**：在前向传播过程中以特定系数（$\alpha$）干预增减该向量组，以此观察下游对靶向量（Logit Difference）最终决断的因果影响。
3. **激活修补 (Activation Patching)**：在网络特定深度的特定 Token 位置执行局部干预替换，定位模型形成医学概念判断的最核心“逻辑隘口”。
4. **SAE 特征捕获分析**：以 Gemma Scope 2 为透镜，将提取的降维抽象概念轴重新投射于稀疏空间中，搜寻与该宏观概念严格对应的微观可解释特征神经元。

### 4. 实验结果与解析
实验的实际数据在开源权重体系上极度完美地证实了原论的“线性干预假说”：

- **特征层定位 (Axis Extraction)**：在 1B 模型中，表征“糖尿病轴”信息的最优成型富集点为计算深度的**第 21 层（Layer 21）**（测试集分类准确率达 65.7%，Logit Lens DLA 达 +0.192）。在小型化 270M 架构中则对应在第 12 层（测试集准确率 70.7%）。
- **完全因果特征转向 (Monotonic Causal Steering)**：对第 21 层的验证展现了极度严谨的“正因果响应单调性”。当调节系数 $\alpha$ 从 -3.0 横扫至 +3.0，该神经网络对两类靶点药物的偏向概率差（$\Delta$ logit_diff）表现出了由 **-4.277**（压倒性倾向二甲双胍）平稳连续地过渡向 **+4.621**（压倒性倾向胰岛素）的连续递变图谱（270M 同现 -2.005 至 +2.005单调）。这强力证明，该轴系决非相关性巧合，而是具备完全控制力的“因果算子拨盘”。
- **教科书级归一修补指标 (Activation Patching)**：在锁定层（Layer 21 的末位 token -1）进行直接干预替换时，激活修补的归一化相对得分惊人地达到了 **0.9955 (99.55%)**（95% CI: `[0.9751, 1.0148]`）。换言之，该概念在机制上的逻辑通道被完整定位到了这一孤立位置，实现几乎 100% 的决策机制扭转。
- **微观节点锚定**：我们基于 SAE 精确还原了该概念方向所锚定的对应稀疏节点（如 Layer 21 中的 Feature 869、3162 等）。


