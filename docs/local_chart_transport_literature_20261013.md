# G-7h 独立文献与机制登记

2026-10-13；仅本路线使用，未修改paper/references.md。核对原始文献页，已有核回归、局部坐标和变化输运不是本项目原创。

[核回归：E. A. Nadaraya, On Estimating Regression, Theory of Probability and Its Applications 9(1), 141–142, 1964](https://epubs.siam.org/doi/10.1137/1109020)。原始出版记录已核对。本实验采用固定高斯加权训练chart的条件均值，是经典核平滑的矩阵值应用；没有数据驱动带宽最优性或高维一致性结论。

[局部向量场：Amit Singer、Hau-tieng Wu, Vector Diffusion Maps and the Connection Laplacian](https://arxiv.org/abs/1102.0075)。作者原稿摘要核对了对向量场进行插值/回归及在流形采样条件下关联connection Laplacian的目标。本实验没有实现向量扩散谱、估计连接或验证流形采样假设；借鉴的是局部坐标必须有跨区域对齐规则这一问题，不能继承其理论保证。

[变化输运：Hang Shao、Abhishek Kumar、P. Thomas Fletcher, The Riemannian Geometry of Deep Generative Models](https://arxiv.org/abs/1711.08014)。作者原稿核对了生成映射上的几何、沿路径切向量平行移动及类比构造。本实验冻结I-JEPA无生成解码器，没有pullback度量、测地线或Levi-Civita连接，B_r B_d^+只是标签对齐坐标的线性输运，不应称黎曼平行移动。

方法新旧边界：旧L1–5按两参照PCA2选4个互斥正交投影，旧N2只旋转/缩放单个参照方向；本次用完整128维参照描述子对32个训练场景的6列因素坐标连续加权，再在donor/recipient各自共同逆坐标内交换目标两列。它是有监督、条件局部线性模型，连续变化发生在场景条件，不是在每个物体状态上估计非线性切向场。三水平离散数据只支持该有限chart测试，不证明已发现连续生成流形。循环和交换性来自共同左逆的代数结构，不是独立物理机制证据。
