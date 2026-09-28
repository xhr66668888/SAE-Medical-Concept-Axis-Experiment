# BRIDGE @ BIBM 2026：实验重设计与论文规划

规划日期：2026-09-22。本文是待执行研究方案，不包含新实验结果；本次只读审计代码和已有结果，没有启动训练、推理或重跑。已知条件：三张 RTX PRO 6000；目前没有认知障碍语料和临床合作者。

## 1. 投稿定位与建议

建议主投 4 页 short paper，集中做“对话上下文的机制审计”。8 页 regular paper 只有在数据审核、外测和第二模型结果都完成后再考虑。计算不是主要瓶颈，人工审核、因果对照和写作时间才是。

该 workshop 聚焦认知变化相关语言与交互指标、照护和负责任的评估。官网与投稿系统当前都标明 2026-09-27 23:59 AoE 截止；4/8 页均包含参考文献与附录，双盲。官网注明日期可能调整，应在投稿前再次核对。[Workshop CFP](https://liulabou.github.io/bridge-bibm-2026/)、[投稿系统](https://wi-lab.com/cyberchair/2026/bibm26/scripts/submit.php?subarea=S48&undisplay_detail=1&wh=/cyberchair/2026/bibm26/scripts/ws_submit.php)

推荐研究问题：**模型能否区分尚未解决的沟通困难与表面相似的合理重复/确认？残差方向和 SAE 特征的干预，改变的是这种上下文判断，还是泛化的“多提问、多帮助”倾向？**

与 workshop 的连接是：如果未来用人机互动支持认知健康研究，首先要知道模型是否正确解释交互证据，以及其回应是否制造额外的重复与修复。本研究检验这一前提，不能声称已得到认知衰退指标、诊断性能或照护效益。

这是对原实验研究对象的实质调整。可复用实现框架，旧 ICD 结果只作 pilot。若希望继续研究疾病概念本身，应保留另一条研究线，寻找更广义的医疗 AI / interpretability venue；这个 workshop 的范围匹配仍会是主要问题。

## 2. 原实验审计：哪些结论需要降级

原实验是 Gemma-3-4B-IT、10 个 CCS 类别对比、34 层、17,360 个模板展开后的提示。实际为 1,736 个不同诊断描述；测试只有 432 个诊断、216 对，各展开 3 个模板。糖尿病测试仅 3 对/6 个诊断/18 行。

| 已有实现或结果 | 对研究结论的影响 | 重跑要求 |
|---|---|---|
| 用 test accuracy 减 random-null mean 选择最佳层 | 报告的 64.4%–99.4% 是选择后的表现 | 独立 train/dev/test，dev 锁层和干预参数 |
| lexical baseline 读取 CCS code，再调用造标签规则 | 100% 是本体映射结果，不是真实文本基线 | 只读输入文本的 TF-IDF/ngram 分类器 |
| 17,360 行来自重复模板；97.0% 测试诊断的三位 ICD 家族在训练出现 | 不能当独立病例数或跨疾病家族泛化 | 按场景家族分割、按家族 bootstrap/permutation |
| 正负描述各自排序后按序号配对 | 没有保持病理、器官、长度等因素不变 | 最小反事实或明确因子设计 |
| SAE 只排名 activation difference × decoder-axis alignment | 没有验证 SAE 特征的因果作用 | feature ablation/transplant 与重建控制 |
| 多数 alpha=+6 的标签 log-prob 差变化约 0.0003–0.0173 | 有方向一致性不等于行为可控性 | 实际决策、自由回应和选择性端点 |
| 伤口轴效应几乎为零；妊娠轴剂量斜率 CI 跨零 | 必须保留负结果 | 完整报告全部预设条件 |
| 自动报告主表仅显示前 8 个轴 | 两个轴的负结果没有完整呈现 | 取消自动截断，逐条件导出 |

证据位置：

- [测试集选层](/Users/xuhaoran/Downloads/SAE-Medical-Concept-Axis-Experiment-main/scripts/fit_axes.py:378)。均值差方向本身确实只用 train，不能把 `fit_splits=train,test` 误读为方向直接用测试集拟合。
- [本体 oracle 基线](/Users/xuhaoran/Downloads/SAE-Medical-Concept-Axis-Experiment-main/scripts/run_lexical_baseline.py:35)。
- [激活采集](/Users/xuhaoran/Downloads/SAE-Medical-Concept-Axis-Experiment-main/medical_axis/runtime.py:183)、[自动报告截断](/Users/xuhaoran/Downloads/SAE-Medical-Concept-Axis-Experiment-main/medical_axis/reporting.py:17)。
- [完整旧结果](/Users/xuhaoran/Downloads/SAE-Medical-Concept-Axis-Experiment-main/runs/gemma3_4b_ccs_icd9_full/outputs/axis/axis_summary.csv)、[剂量结果](/Users/xuhaoran/Downloads/SAE-Medical-Concept-Axis-Experiment-main/runs/gemma3_4b_ccs_icd9_full/outputs/steering/steering_dose_response.csv)。

另外三项重跑前必须处理的技术问题：

1. capture 使用 `hidden_states[layer+1]`，干预挂 decoder-layer output。官方 Gemma3 实现中，最后 hidden state 已经过 final norm，而最后 decoder output 尚未经过；旧结果 layer32→33 的 axis norm 也骤降约 160–178 倍。旧运行版本未锁定，具体影响未动态回验；但必须统一 capture、SAE、patching、steering 的同一 hook site。[官方实现](https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/models/gemma3/modeling_gemma3.py#L705)
2. SAE/patching 的配对字典按 pair_id、side 覆盖模板，实际通常只保留一个模板。应保存 family_id、variant_id、template_id、source_id。
3. readout_baseline 是每行 own-label 减 opposite-label，而 steering 是固定 positive-label 减 negative-label；两者需统一符号。原多 token 标签总 log-prob 还混入长度和标签先验。

## 3. 新颖性应放在哪里

| 近邻 | 已有贡献 | 本研究应增加的证据 |
|---|---|---|
| Assistant Axis | persona 表征、漂移与 activation capping | 模型如何表征用户的交互证据，以及表征对回应的影响 |
| ClarifySAE | SAE 特征发现与澄清行为 steering | 在需要与不需要额外修复的上下文之间，干预是否具有选择性 |
| Ngo et al., LREC 2026 | LLM repair 标注依赖词汇、忽视上下文 | 词汇匹配的反事实、内部表示、受控特征干预的证据链 |
| 医疗 SAE 工作 | 已识别医疗概念并实施 steering | 单纯把 SAE 用到医学或再找一个 concept axis 不足以作为创新 |

来源：[Assistant Axis](https://www.anthropic.com/research/assistant-axis)、[ClarifySAE 作者仓库](https://github.com/sn0rkmaiden/clarifySAE-steering)、[LREC repair 研究](https://aclanthology.org/2026.lrec-1.547/)、[MAIRA-2 SAE](https://arxiv.org/abs/2507.12950)、[CAST 临床文本研究](https://arxiv.org/abs/2608.27397)。这是定向文献核查，不是穷尽式新颖性证明。

应避免“首次发现重复特征”“首次 SAE 医疗应用”“找到 dementia axis”。可以把贡献写成：**在词汇和任务内容受到控制时，检验交互表征的上下文敏感性，并用选择性干预区分可读出信息与被模型用于回应的信息。**

## 4. 数据：只标可观察的交互状态

五天版聚焦两类：重复/确认与修复是否完成；话题维持、指代困难留作扩展。先用英文，避免同时引入跨语言因素。

标签定义为 `unresolved communication problem / resolved or no problem / indeterminate`。正类必须有明确证据，例如请求解释仍未获回答、互相矛盾的任务信息尚未澄清、所需指代仍不明确。单次重复、口吃、简短回答、年龄不能直接作为正类依据。无法确定的例子保留 indeterminate，报告比例，不强行二分。主 A/B 分析只针对预先审核为明确状态的样本；indeterminate 在模型测试前划入单独歧义诊断集，评价不确定性/拒答行为。完整公开纳入流程与数量，不能看完模型结果后删例。

采用“表面重复 × 修复状态”的交叉设计：

| 条件 | 场景构造 | 期望模型识别 |
|---|---|---|
| 有重复，尚未解决 | 用户重问，但系统仍没有回答关键疑问 | 当前还有沟通问题 |
| 有重复，已解决/正常 | 系统要求 read-back，或用户按要求确认信息 | 不能因重复自动判为困难 |
| 无重复，尚未解决 | 没有重复措辞，但必要信息矛盾或指代不明 | 不能因无重复就判为已解决 |
| 无重复，已解决/正常 | 信息完整、一致，任务可以继续 | 正常推进 |

同一场景尽量保留实体、任务目标、轮数和最后一句；必要的上下文修改应由审核者确认会改变交互标签。不要为了完全匹配而制造不自然对话。歧义没有被消除的样本应标为 indeterminate，而不是让生成模型强制裁决。

建议五天版目标为 **300 个独立场景家族 × 4 个条件 = 1,200 个短对话**，每段约 4–8 轮。按预约、购物、交通、做饭、家庭安排等任务分层。这个规模是工作量建议，不是功效保证；人工审核跟不上时缩小主张和样本规模，不能用模板扩增冒充独立数据。

训练/开发/测试为 180/60/60 个家族；同一家族全部条件、改写、生成种子必须在同一 split。留出生成模板家族，并预留一种任务场景作为额外迁移测试；若样本不足，保留一个可信独立测试，不堆多个小测试集。

先写约 30 个家族和标注准则，再扩展。可以用 LLM 起草，但不得让被测模型同时生成全部样本、定全部真值和评价自己的输出。所有测试例子由人复核，记录明确证据跨度。若能找一位非临床同学，独立复标至少 100 个对话并报告一致率/κ；没有第二人就如实报告单人审核，不声称临床标注或多人共识。

长度、重复 n-gram、问号、否定词、礼貌、词汇复杂度和生成来源要做分布诊断。元数据标签不进入模型输入。听力、ASR、分心等因素也可能造成真实修复需求；不能机械地将这些条件当作“无需帮助”的负类，更不能由其反推疾病。

外部验证建议从 [CCPE-M](https://github.com/google-research-datasets/ccpe) 取约 100 个对话片段：这是 502 个众包 Wizard-of-Oz 电影偏好对话组成的公开语料，许可为 CC BY 4.0。重新按上述准则标注，检查合理重复上的误触发。先冻结模型和阈值再读取外测标签。它没有认知状态标注，不能叫“健康对照组”，也不是临床验证。

[DailyDialog++](https://iitmnlp.github.io/DailyDialog-plusplus/) 的词汇重合但语境不合适回应可作为可选上下文外测，标签任务不同，需要单独报告。[DementiaBank](https://talkbank.org/dementia/access.html) 大多数数据需受控申请，不把审批作为五天投稿的前置条件。

## 5. 核心实验矩阵

| 实验 | 必须回答的问题 | 做法 | 主要输出 |
|---|---|---|---|
| E0 实现验收 | 数值变化是否由预期干预造成？ | hook 一致性、零干预、自复制 patch、SAE 重建检查 | 验收记录 |
| E1 行为与捷径 | 模型能否看懂上下文？ | 完整历史、仅最后一轮、打乱/交换历史；文本基线 | paired accuracy、macro-F1、正常重复误报率 |
| E2 表征 | 哪些层包含跨表达的交互状态信息？ | mean-difference、正则线性 probe、SAE sparse probe | 独立测试 AUROC/accuracy、上下文反事实差值 |
| E3 因果选择性 | 特征对判断与回应有特定作用吗？ | axis/feature 消融、双向交换、随机对照 | 目标条件变化与正常条件损伤 |
| E4 外测/复现 | 是否超出合成模板和单一模型？ | CCPE-M + 12B 关键实验 | 泛化落差、误触发、复现效应 |

E0–E3 是主证据链。五天最小交付固定为 4B、一个一致的干预位置、少量候选 feature、独立测试与人工审核。12B、CCPE-M、竞争 clarification 方向、65K 均在核心链完成后再做；12B 只复现锁定的关键条件。E4 完成度决定文章能否超出纯受控 pilot。

E1 的基线包括：多数类、表面重复/长度规则、TF-IDF word+character n-gram logistic regression、同一模型 zero-shot 与少量 few-shot、普通提示词要求先检查上下文。所有方法只读取允许的对话文本，训练数据和开发预算一致。

readout 主用随机交换顺序的 A/B 候选选项，并核查各模型分词；对不同模板和标签映射取配对结果。方向提取优先在纯对话的 user-turn 结束位置完成，避免把答案选项或生成答案的 token 编进方向。再用不同自然语言标签做稳健性检查。

## 6. 表征与因果实验的具体规格

主模型 Gemma-3-4B-IT，复现 Gemma-3-12B-IT；每个模型独立提取自己的方向，不能跨模型直接搬运向量或 feature ID。两者同属 Gemma 家族，因此只能宣称跨规模复现。

对应官方 SAE 为 [4B IT](https://huggingface.co/google/gemma-scope-2-4b-it) 与 [12B IT](https://huggingface.co/google/gemma-scope-2-12b-it)。先沿用 16K SAE 快速验收；优先考察 4B 的 9/17/22/29 层、12B 的 12/24/31/41 层现成 resid_post SAE。若有余力，在 dev 选中层检查 65K 的结论稳定性。实际 release alias、宽度、稀疏度和 hook 以下载配置核对并记录；不默认 SAE 可用于 MedGemma。

不需要从零训练 SAE。域内 SAE 重建如果明显不可靠，先降低 SAE 结论范围，保留 dense-axis 审计，不在截止前追加大型 SAE 训练项目。

训练集估计方向 d = normalize(mean(h_unresolved) − mean(h_resolved))。主比较为 dense mean-difference、L2 logistic probe、稀疏 SAE 特征模型。SAE 候选仅用 train 排名，用 dev 选择 top-k（建议候选 1/4/16）；层、k、剂量、位置均在测试前锁定。feature explanation 的正负例和命名不能来自最终测试挑选。

同时提取一个表面 repetition 方向，以及一个泛化 clarification-output 方向作为竞争解释。两个方向正交化后的敏感性实验可以辅助检验，但正交不等于语义独立。所有 cosine 对比必须在同一模型、同一层、同一 hook site 中进行。

干预剂量以训练集投影标准差标定：h' = h + alpha × sd_train(h·d) × d，初始 alpha 候选为 −2/−1/−0.5/0/0.5/1/2。报告实际扰动范数与 residual 范数之比，并用 dev 排除明显破坏语言质量的剂量。不同层不可直接比较原来任意单位的 alpha=6。

SAE 干预至少包含：

- 将候选 feature 消融或替换为匹配 resolved donor 的值；反向交换用于检验效果的方向性。
- 只修改候选 feature 的 residual 贡献。对线性 decoder，可写成 h' = h + W_dec(z'−z)，保留原始重建误差；如 SAE 有输入缩放，必须在正确的模型激活单位实施。
- 原始前向、SAE 重建替换、保留误差的 feature 修改三种路径分别检查，避免把重建损伤当作特征因果效应。
- 同数量、匹配激活频率/幅度的随机 feature；等扰动范数的随机方向；同标签 donor；非目标 cue 的 feature。
- patch 位置按证据跨度或语义对齐，不直接把不同长度对话的最后第几个 token 视为同一个位置。先做最后 user-turn 边界的固定位置实验，再扩展跨度。

整体 residual patch 只作为粗粒度对照；主因果证据来自目标方向分量或 feature 子集的交换。raw logit/选择概率变化为主指标；旧版除以 clean-corrupt gap 的比值易受小分母影响，只作辅助且预先规定使用条件。

按真实标签选择 donor 或决定干预正负方向，属于使用真值的机制实验，只能支持双向因果效应与选择性结论。不能把其准确率提升当成可部署的纠错效果，也不能据此声称优于 prompt-only。若比较实际任务改善，干预必须固定，或仅由输入与开发集预先确定的规则触发；评估时不能读取测试真值。两类实验分表报告。

统计按场景家族做 paired clustered bootstrap；置换以家族/匹配对为单位。建议 3 个训练/采样种子、95% CI；固定同一测试集时，种子不增加独立测试样本量。报告每个预设 cue、条件和模型的结果；多 feature 筛选只在开发集完成，确认性检验做适当多重比较控制。

## 7. 行为端点：不要只让模型“更爱提问”

主端点先固定为交互状态判断。自由生成作为次端点，在预先抽取的约 100 个测试对话上比较原模型、prompt-only、dense axis、SAE feature 干预。

用明确 rubric 标注：是否回应当前未解决的问题；是否维持任务事实；是否反复索取已给出的信息；是否无依据推断认知疾病；是否产生冗余或难以执行的多重提问。评审隐藏方法名称、交换输出顺序；LLM judge 可辅助整理，不能独自决定真值。

分别报告：有需要条件的正确修复率、已解决条件的额外修复误触发率、事实保持和语言质量。主图画“目标改善—非目标损伤”关系，不仅画平均 helpfulness 或 clarification rate。

支持需求与修复类型不总能一一对应，允许多种合适回应。没有真实用户研究时，不把输出评分称为沟通效率改善、认知负荷降低或照护效益。

一个贯穿全文的扩展问题是：模型给予的回应会改变下一轮用户语言，因此交互指标可能被系统行为影响。本轮若只做固定历史上的一步回应，就只能把它写成动机和限制，不能宣称已测得这种反馈效应。



## 10. 论文故事、图表和标题

叙事顺序：日常人机互动对认知健康研究有潜在价值 → 表面重复/澄清需要上下文解释 → 现有研究已发现模型在 repair 上存在词汇依赖 → 构造词汇与任务内容受控的反事实 → 测试残差和 SAE 中的表征 → 用干预检查选择性 → 说明对于未来自适应沟通评估的意义与临床验证边界。

四页版建议：引言/相关工作约 0.6 页；数据和方法约 1.1 页；结果约 1.3 页；讨论与局限约 0.35 页；参考文献约 0.65 页。篇幅按实际模板调整，参考文献不能被挤出页数限制。

只保留两张核心图与一张主表：

1. 词汇相近、上下文不同的四条件设计，以及层/feature 干预示意。
2. 干预对真实未解决条件和合理重复条件的效应/误触发曲线。
3. 原模型、prompt-only、dense axis、SAE 及随机对照的独立测试与外测结果。

若 SAE 只有相关性且干预无选择性，主文不能把图称为 circuit。跨层连接或 feature 排名图也不等于经过验证的计算路径。

首选标题：**Beyond Repetition: A Causal Audit of Conversational Repair in Language Models**

如果上下文表征清晰但行为因果证据有限：**Context Matters: Probing Conversational Repair in Language Models**

如果发现稳定的读出—干预差距：**Readable but Not Selectively Controllable: Auditing Conversational Cues in Language Models**

如果 SAE 的选择性与回应效应都得到支持：**From Conversational Cues to Supportive Responses: A Sparse-Feature Audit of Language Models**

这些标题均为候选；最终标题随结果选定。“Causal audit”表示使用干预开展审计，不意味着所有被测特征都通过了因果检验。不要用 Dementia Axis、Early Detection 或 Digital Biomarker 作为当前数据可以支持的结论。

摘要可按五句组织：应用动机；上下文歧义问题；受控数据与残差/SAE 干预方法；填写实际结果和 CI；阐明这是模型机制审计，真实认知健康效度仍待验证。现在不预写积极结果。
