# G-7h 连续条件局部线性坐标与输运冻结协议

2026-10-13；R2-2 / I-4；执行前冻结，CPU、一次有界实验。目的不是证明已发现流形，而是检验连续场景条件、共同逆坐标是否比旧硬路由投影更接近真实因素编辑。旧train32拟合，旧tune8首先评分，再旧ID32/OOD32；全部属于探索，不新增确认。

原L1–5以PCA2描述子硬选四个正交rank16投影，OOD颜色1.333仍劣于noedit。A3是原r0稀疏重构四格，局部坐标没有统一；N2仅用两参照单方向旋转/缩放。本次不重跑这些方法。采用标签对齐的6维非正交局部线性chart，而非更多四聚类或UMAP可视化。

冻结I-JEPA grid5120及旧train拟合PCA64。对三水平因素分别用两个正交Helmert对比，拼接27×6矩阵q，以000/222两行均值q0为原点，C=q−q0。场景s的两参照均值为a_s，训练用全27个已知干预状态拟合B_s=argmin_B ||H_s−a_s−C B^T||²。因素水平标签用于给坐标统一规范，监督强于只知道干预类别；不能称无监督。拟合残差、条件数、主效应/交互能量分别保存。

场景条件c_s=[a_s,H_s(222)−H_s(000)]∈R128。固定高斯核权重w_s(c)∝exp(−||c−c_s||²/(2σ²))，σ为32训练描述子两两非零欧氏距离中位数，不扫描带宽/维数/seed；所有32权重连续，不做近邻截断。B(c)=Σw_sB_s。新scene只读两参照，不读真实factor codes用于反解，且不访问27真值库拟合。伪逆A(c)=B(c)^+的相对阈值预定1e-8；有效秩和六个奇异值逐scene保存。若有效秩<6则仍报告数值结果及失效身份，不许声称坐标可逆；如果秩6但条件数>1e4，标为病态，不偷偷调阈值。保存||A||₂和||B||₂用于噪声放大。

反解u=A(c)(h−a)，编辑第f因素为h′=h+B_f(c_r)(u_d,f−u_r,f)。recipient的chart外残差和PCA补空间保留；donor真实目标codes不输入反解/编辑，只用于构造匹配真值及评价。对照noedit、旧joint0未校正、旧两参照joint0、global6（同6维全局平均B）、连续local6、同容量wrong-condition（训练B_s按scene固定循环移位1格，描述子不变）。local/wrong各存32×64×6，加描述子32×128；global只存64×6，因此不称全局与局部总容量相等。错配控制保留全部容量和核权重。

每域按原固定i与i+n/2双向场景对，单因素采用原conflicting donor集合432 edits/scene/factor；两因素采用两目标均变且第三因素冲突的216 edits/scene/factor-pair。真值在recipient场景27状态中，计算全5120维相对平方误差（分母noedit）、绝对平方误差、目标nearest正确率、非目标保持率、全部正确率。场景先均值再pair等权，tune只有4pair、ID/OOD16pair。主比较local-global和local-wrong全部三因素报告，不以dev选配置。剔除任何输入/目标碰000或222的单编辑结果另存。bootstrap固定seed20261013、5000pair、无多重校正，只描述旧dev探索。

组合交换和循环对chart/正交块可由代数结构保证；必须以独立重算检验实现，但零值不是因果独立或真实合成成功。真实两因素干预误差才检验功能。保存输入哈希、训练系数/核权重/反解/奇异值、每scene metrics及固定抽样编辑预测；另以原5120向量直接计算距离重核低维公式。所有方法原样保留，不因OOD坏而增加中心、带宽或非线性阶数。

文献仅作为机制参照：[Singer与Wu, Vector Diffusion Maps and the Connection Laplacian](https://arxiv.org/abs/1102.0075)研究局部向量场及连接结构；[Shao、Kumar与Fletcher, The Riemannian Geometry of Deep Generative Models](https://arxiv.org/abs/1711.08014)研究生成流形上切向变化的平行移动。本文没有生成解码器/黎曼度量估计，也未实现上述方法的平行输运保证；B_r A_d仅为干预对齐坐标间的局部线性映射，不能借文献名称宣称几何定理。
