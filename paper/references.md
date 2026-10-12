# 文献定位与引用边界

本文件是独立、持续维护的引用登记。方括号大写ID为稳定引用key，正文可按论断引用，后续更新不重编号。每条表格中的核心贡献、条件/限制和本项目关系共同构成登记项。核验状态：除显式PENDING项外，均于本轮打开原始论文/正式论文集确认标题、作者与摘要/相关内容；“等”缩写完整作者以原始链接为准，不代表未核对身份。近期预印本仅核验来源和其报告内容，未独立复现其数值，不声称全文实验覆盖已经审计。

核验日期：2026-10-13。只把原始论文、正式论文集或作者原始预印本作为结论来源。以下是定向检索，不是系统综述；未证明任何“首次”。近期预印本仅按所列版本记录，不把作者摘要中的优越性主张当作本项目复现事实。

## 最直接的竞争与邻近工作

| 文献与已核实身份 | 本文可引用的事实 | 与本项目的关系及边界 |
|---|---|---|
| [DSREG26] **DSReg**，Yujia Zheng、David Klindt、Randall Balestriero、Bernhard Schölkopf，*DSReg: Provably Recovering Individual World Latents without Reconstruction*，arXiv:2610.09457v1，2026-10-07。[原文](https://arxiv.org/html/2610.09457v1) | 在已满足线性可辨识前提的表示上，利用观测依赖足迹的Structural Diversity与functional no-cancellation条件恢复符号/置换意义下的因素；研究训练后旋转。 | 最直接的后处理理论近邻。不能把普通冻结I-JEPA等同LeJEPA理论设定；我们的局部依赖估计/PCA/块约束适配不等于原方法完整复现。OFF等F/m观测别名不是其定理反例：观测已不含唯一参数，先验可辨识前提需先成立。身份先由旧稿书目定位，再由原文核实。 |
| [RATEIDENT26] Tinghe Zhang等，*Frozen in a Frame: The Velocity Blind Spot in JEPA World Models*，arXiv:2610.04585v1，2026-10-03。[原文](https://arxiv.org/abs/2610.04585) | 针对无运动模糊单帧观测的速度盲区，提出RateIdent诊断与TI-JEPA运动分支，并报告规划评价。 | 直接重叠“先检查输入可观测性，再解释读出失败”。我们的多帧质量/隐藏施力等比别名与接触开关不同，但不能称首次发现JEPA可观测性问题。不能把历史回顾位移收益当超过它的未来预测/规划证据。 |
| [CIJEPA26] Mintu Dutta、Ritesh Vyas、Mohendra Roy，*CI-JEPA: A Counterfactual Analysis of Latent Representations in Joint-Embedding Predictive Architectures for Self-Supervised Learning*，arXiv:2610.05043v1，2026-10-04。[原文](https://arxiv.org/abs/2610.05043) | 训练预测原图与改图的表示差，使用花中心遮挡、背景模糊/着色评价选择性响应。 | 直接邻近视觉干预响应诊断；原文也区分表示响应、分类精度与因果发现。我们的机制开关和真实轨迹审计区别于图像编辑；不能只以“干预JEPA”命名新颖性。 |
| [PSGJEPA26] Haodong Yan等，*Is Forward Prediction Enough? Physical State Grounding for JEPA World Models*（PSG-JEPA），arXiv:2608.06799v1，2026-08-07。[原文](https://arxiv.org/abs/2608.06799) | 用本体状态及多时距关节变化训练目标强化表示，报告探针、冻结表示规划和策略评价。 | 已讨论预测目标不足以保证物理状态可读。我们的固定编码器诊断不训练该损失，也没有等价控制收益，不能声称在其任务上胜过它。 |
| [PHYLATENT26] Xi Zeng、Haojie Ren、Ziying Song、Yuanbo Nie、Ross Drummond，*PhyLatent: Learning Dynamics-Relevant Representations for JEPA World Models*，arXiv:2608.05720v2，2026-09-27。[原文](https://arxiv.org/abs/2608.05720v2) | 区分外观不变性、物理状态区分及动作条件未来区分失败，并以训练目标干预，报告视觉控制结果。 | 与本项目“物理差异—表示差异”诊断非常接近。应明确我们的等价轨迹不应该被强制分开；不能用别名场景否定它在可区分状态上的目标。这里采用v2身份与术语，不混用搜索摘要里的v1数据。 |
| [DAS24] Atticus Geiger、Zhengxuan Wu、Christopher Potts、Thomas Icard、Noah Goodman，*Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*（DAS），CLeaR 2024，PMLR236:160–187。[正式原文](https://proceedings.mlr.press/v236/geiger24a.html) | 学习分布式坐标对齐，使用因果抽象框架解释网络，不要求语义变量对应原始神经元坐标。 | 最直接的“旋转/子空间+干预”近邻。我们的因素移植需与这种已有思想比较；仅把编辑向量接近真实干预特征，并未证明原模型下游行为满足因果抽象。 |
| [CAUSALABSTRACT21] Geiger、Hanson Lu、Icard、Potts，*Causal Abstractions of Neural Networks*，2021。[作者预印本](https://arxiv.org/abs/2106.02997) | 用interchange interventions检验神经表示与高层变量的因果角色对应。 | 说明探针可读与行为因果作用不是同一结论。本文目前侧重观测/表示一致性，尚无原生模型行为干预闭环。 |

## 基础表示、因果学习和物理可辨识性

| 文献 | 可支持的定位与限制 |
|---|---|
| [IJEPA23] Mahmoud Assran等，*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*，2023。[原文](https://arxiv.org/abs/2301.08243) | I-JEPA预测同一图像中目标块的表示，是图像自监督方法。对我们冻结I-JEPA逐帧特征的描述应明确，不称它已训练动作条件物理世界模型；不能从本文推广否定整个JEPA家族。 |
| [VJEPA24] Adrien Bardes等，*Revisiting Feature Prediction for Learning Visual Representations from Video*，2024。[原文](https://arxiv.org/abs/2404.08471) | V-JEPA通过视频特征预测学习表征，报告冻结骨干上的运动/外观任务。它与I-JEPA时间输入、预训练数据及目标不同；未实跑V-JEPA就只能引用背景，不能宣称本结果已跨视频JEPA复现。 |
| [LOCATELLO19] Francesco Locatello等，*Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations*，ICML2019，PMLR97:4114–4124。[正式原文](https://proceedings.mlr.press/v97/locatello19a.html) | 无额外归纳偏置时一般无监督解耦不保证可识别。它研究潜变量模型歧义，不等于我们特定动力学输入下参数别名的证明，更不意味着有动作/干预辅助信息时一律不可能。 |
| [CITRIS22] Phillip Lippe等，*CITRIS: Causal Identifiability from Temporal Intervened Sequences*，ICML2022。[正式原文](https://proceedings.mlr.press/v162/lippe22a.html) | 利用时序和已观察干预目标识别标量/多维因素，并可处理预训练autoencoder表示。时序、干预目标等是明确额外假设；知道模拟器标签进行监督读出不等于运行CITRIS或达到其识别保证。 |
| [BISCUIT23] Phillip Lippe等，*BISCUIT: Causal Representation Learning from Binary Interactions*，UAI2023。[正式原文](https://proceedings.mlr.press/v216/lippe23a.html) | 在二值交互机制等条件下研究因果表示学习。与机器人交互强相关，但不能将实验中人为记录的contact_mode开关直接当满足其整套识别条件，更不能把公开CausalWorld重建视作官方BISCUIT完整复现。 |
| [WSCRL22] Johann Brehmer、Pim de Haan、Phillip Lippe、Taco Cohen，*Weakly supervised causal representation learning*，NeurIPS2022。[正式原文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/fa567e2b2c870f8f09a87b6e73370869-Abstract-Conference.html) | 未知随机干预前后配对在其假设下可识别因果表示。我们配对开关是已知机制干预，并非训练该模型；配对数据和因果表示识别已有先例，不能以有配对就声称新理论。 |
| [MECHSPARSE22; PARTIAL26] Sébastien Lachapelle等，*Disentanglement via Mechanism Sparsity Regularization: A New Principle for Nonlinear ICA*，CLeaR2022。[正式原文](https://proceedings.mlr.press/v177/lachapelle22a.html)；同作者线 *Nonparametric Partial Disentanglement via Mechanism Sparsity: Sparse Actions, Interventions and Sparse Temporal Dependencies*，JMLR27(71):1–90，2026。[正式原文](https://www.jmlr.org/beta/papers/v27/24-0771.html) | 稀疏动作/历史依赖和部分解耦已有系统理论。不能把增加历史、动作或允许共享块本身作为新贡献；我们的样本/维度预算对照仅检查具体冻结表示，不验证全部结构假设。 |
| [IMA21] Luigi Gresele、Julius von Kügelgen、Vincent Stimper、Bernhard Schölkopf、Michel Besserve，*Independent mechanism analysis, a new concept?*，NeurIPS2021。[正式原文](https://proceedings.neurips.cc/paper_files/paper/2021/hash/edc27f139c3b4e4bb29d1cdbc45663f9-Abstract.html) | 用混合机制的几何假设约束非线性不可辨识性。提示正交投影/不相交坐标是建模约束，不自动等于物理机制独立；我们的已知干预响应几何应标作诊断。 |
| [SCALING20] Mario Castro、Rob J. de Boer，*Testing structural identifiability by a simple scaling method*，PLOS Computational Biology16(11):e1008248，2020。[正式原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008248) | 参数缩放导致相同可观测输出是既有结构可辨识性问题。F/m缩放不是我们的新发现；本项目增量是接触机制切换下的可复现参数别名案例与表示证据链。 |
| [SCALINGLIMIT21] Alejandro F. Villaverde、Gemma Massonis，*On testing structural identifiability by a simple scaling method: Relying on scaling symmetries can be misleading*，PLOS Computational Biology17(10):e1009032，2021。[正式原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009032) | 只检查缩放不足以证明一般模型可识别，因为还有其他对称性。我们找到具体别名足以否定这些pair的唯一识别，却不能从其余pair分开推出全部参数在任意实验中可识别。 |
| [CAUSALWORLD20] Ossama Ahmed等，*CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning*，2020预印本。[原文](https://arxiv.org/abs/2010.04296) | 可干预的机器人操作环境本身已有。本项目生成的是其公开环境上的定制配对fixture，而不是新物理模拟器；现有结果不等同官方全部迁移基准。 |

## 未完成核验，不能据此作强比较

[VJEPA2CAUSAL_PENDING] 检索命中 *Causal State Variables in V-JEPA 2 Latents: Discovery, Intervention, and Portability* 的[OpenReview原始提交](https://openreview.net/forum?id=CC5xKstwmf)。搜索索引摘要称使用冻结V-JEPA2、子空间与干预，但本轮论坛及PDF均返回浏览器验证页，作者、版本、评审身份和完整方法未核实。它是必须人工补核的高相关线索，不可写成已评审基线，也不引用未经全文核验的具体数值。检索到的二手总结未作为本文事实来源。

## 写作时应区分的三个层次

物理参数可观测性：给定输入、动作是否已知、时间窗、接触模式和环境后，哪些参数组合产生相同观测。表示保留性：不同观测在冻结编码器下是否仍有可区分信号。特定读出/编辑能力：固定PCA、样本预算和回归器能否利用该信号。物理别名、编码压缩和有限模型拟合失败都能造成低探针分数，但不是同一机制。该划分是论文组织建议，不是声称首次提出的新定理。

## 可解释性方法与反例：有限可复用清单

| 稳定ID、文献身份与原始链接 | 核心贡献/假设/限制 | 可引用论断与本项目关系；核验状态 |
|---|---|---|
| [LEACE23] Nora Belrose、David Schneider-Joseph、Shauli Ravfogel、Ryan Cotterell、Edward Raff、Stella Biderman，*LEACE: Perfect linear concept erasure in closed form*，NeurIPS2023。[正式PDF](https://papers.nips.cc/paper_files/paper/2023/file/d066d21c619d0a78c5b557fa3291a8f4-Paper-Conference.pdf) | 闭式线性概念擦除，针对线性可读信息并控制表示扰动；不保证消除非线性信息或分布外重新出现的信号。 | 可把监督线性擦除作为“轴内/残留”诊断基线，不能据线性探针失效就称概念彻底删除。已核实正式论文与作者预印本；G5l已完成有限连续目标平方损失扩展，非原分类benchmark完整复现。OFF matched dev R² .594→.035，held交叉协方差仍有残留，非线性信息未测，不称held完全擦除；见[本地报告](../docs/leace_observable_report_20261013.txt)。 |
| [PROBECONTROL19] John Hewitt、Percy Liang，*Designing and Interpreting Probes with Control Tasks*，EMNLP-IJCNLP2019:2733–2743。[正式原文](https://aclanthology.org/D19-1275/) | 用控制任务区分探针记忆容量与目标表现，实验对象为语言表示。 | 支持有限容量/标签负控的必要性，不直接证明我们episode置换消除了进度捷径。已核实正式论文；迁移到视觉时需另验证控制设计。 |
| [ASAE25] Thomas Fel、Ekdeep Singh Lubana、Jacob S. Prince、Matthew Kowal、Victor Boutin、Isabel Papadimitriou、Binxu Wang、Martin Wattenberg、Demba E. Ba、Talia Konkle，*Archetypal SAE: Adaptive and Stable Dictionary Learning for Concept Extraction in Large Vision Models*，ICML2025，PMLR267:16543–16572。[正式原文](https://proceedings.mlr.press/v267/fel25a.html) | 研究SAE字典稳定性，以数据凸包锚定原子并提出放松变体；含概念合理性与合成混合识别评价。 | 稀疏重构不等于物理因果因素。若试字典，应与其稳定性思路对齐、跨seed匹配和真实干预验证，不能只展示漂亮原子。已核实正式论文，尚无本项目复现。 |
| [PATCHILLUSION24] Aleksandar Makelov、Georg Lange、Atticus Geiger、Neel Nanda，*Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching*，ICLR2024。[正式PDF](https://openreview.net/pdf?id=Ebt7JgMHv1)；[预印本](https://arxiv.org/abs/2311.17030) | 子空间patching成功可能激活旁路，而不是定位原本承载功能的表示；不主张所有patching均无效。 | 直接约束我们的因素移植解释：目标误差低仍需原模型行为/必要性/匹配扰动证据。正式PDF有四作者；早期arXiv索引仅三作者，不能混用版本。已核实正式PDF搜索文本与预印本身份。 |

## 作者公开经验与实务建议（单列证据等级）

[PATCHGUIDE24] Stefan Heimersheim、Neel Nanda，*How to use and interpret activation patching*，arXiv:2404.15255，2024。[作者原始实务文稿](https://arxiv.org/abs/2404.15255)。等级：作者经验性方法建议，不是额外可辨识性定理。讨论patching选项、指标与推断边界；可借鉴保留绝对误差、基线和扰动分布检查，不能据指南宣称本项目已识别因果机制。已核实标题作者摘要，未据此覆盖全文所有建议。

[HEWITTBLOG19] John Hewitt，*Designing and Interpreting Probes*，2019。[作者博客](https://nlp.stanford.edu/~johnhew/interpreting-probes.html)。等级：对本人PROBECONTROL19论文的讲解，不新增独立实验证据；引用技术结论优先正式论文，博客仅用于理解探针容量陷阱。已核实作者站点与对应论文关系。未把社区转载、自动摘要或论坛经验当证明。

## 维护规则

新增来源先核验唯一身份和原始链接，再分配稳定ID；记录版本变化，原结论不静默改成新版本。每次扩展应由待回答的具体审稿问题驱动，不凑条目数。尚未阅读的全文细节、无法访问的提交和未复现的性能分别标注。原始论文的“first”不自动成为本项目的首次主张。当前检索未建立穷尽性，正文禁用“首次”。

## 数据来源、软件与实际基线补充

| 稳定ID | 真实身份、原始来源 | 本项目用法、可引用论断与核验状态 |
|---|---|---|
| [CAUSALWORLD_CODE] | rr-learning作者团队维护的[CausalWorld官方源码](https://github.com/rr-learning/CausalWorld)，对应CAUSALWORLD20论文。 | 引用环境来源；本文修改生成协议而非发明新模拟器。生产PyBullet版本、源码快照/生成脚本hash以本地manifest为准，不以当前main代替历史运行版本。已打开官方仓库核验；与论文条目并列引用。 |
| [LIBERO23] | Bo Liu、Yifeng Zhu、Chongkai Gao、Yihao Feng、Qiang Liu、Yuke Zhu、Peter Stone，*LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning*，NeurIPS2023 Datasets and Benchmarks。[正式PDF](https://papers.nips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf)；[原始预印本](https://arxiv.org/abs/2306.03310)；[官方源码](https://github.com/Lifelong-Robot-Learning/LIBERO)。 | 用于操作demo与任务来源。我们的逐task监督读出和30/10/10划分不是官方lifelong learning整体基准，也不是同一模型跨task迁移。官方项目旧网页标题用in且作者表较短，本登记采用正式论文的for及七作者。已核实论文身份与官方仓库；未声称复现官方策略结果。 |
| [SFA02] | Laurenz Wiskott、Terrence J. Sejnowski，*Slow Feature Analysis: Unsupervised Learning of Invariances*，Neural Computation14:715–770，2002。[作者机构论文页](https://www.ini.rub.de/PEOPLE/wiskott/Projects/LearningInvariances.html)；[作者机构PDF](https://papers.cnl.salk.edu/PDFs/Slow%20Feature%20Analysis_%20Unsupervised%20Learning%20of%20Invariances%202002-3430.pdf)。 | 慢变化特征基线来源；我们在冻结表示上使用的线性/有限维SFA不等于原文全部非线性特征实验，慢也不等于物理因素。作者机构索引/PDF搜索内容已核实标题作者期刊页码，PDF直开本轮内部错误；不引用未读取的细节定理。 |
| [PCA1901] | Karl Pearson，*LIII. On lines and planes of closest fit to systems of points in space*，Philosophical Magazine2(11):559–572，1901。[出版方原始条目](https://www.tandfonline.com/doi/abs/10.1080/14786440109462720)。 | PCA的经典最小平方低维拟合出处；解释本文train-only方差排序压缩，不支持“主成分等于语义因子”。出版方检索核实书目信息；网页2010在线日期不是论文发表年份，不声称审读全文。 |
| [DSREG_CODE] | DSReg作者[项目页](https://dsreg.github.io/)（由DSREG26原文链接）。 | DSREG26已提供理论和方法出处；本项目实际官方API/参数改写须另引本地[适配记录](../docs/intervention_step2b_step3a_report_20261012.txt)与对应代码hash。项目页地址在原文已确认，但本轮没有另核其仓库commit，不据它断言本地是最新版或完整复现。 |

G5l连续目标依据另见[LEACE作者原文v4 §4.3](https://arxiv.org/html/2306.03819v4#S4.SS3)与[官方实现](https://github.com/EleutherAI/concept-erasure)。本地报告记录使用原解析公式与数值阈值，未沿用官方默认收缩等选项；方法引用与实现差别必须同时注明。此项实验状态更新于2026-10-13，旧文献登记中“未运行”状态由有限扩展完成替代，不升级为完整原论文复现。
