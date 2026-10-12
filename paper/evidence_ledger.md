# 论文事实证据账本

更新：2026-10-13。此账本仅核对现有产物，不新增实验，不替代冻结协议。正文应称“冻结 I-JEPA 图像编码器在当前 CausalWorld 仿真装置上的机制审计”。机器人、接触模型、相机、渲染器与采样分布均有限定；没有跨编码器、真实机器人或通用 JEPA 理论证据。代数块互斥、因素可读、响应集中和因果独立是不同命题。

## 证据等级与统计单位

“新场景留出”表示同一装置下按 scene 隔离；“独立复算”表示用替代数值实现核对同一份数据，不等于第二数据集确认。W 的生产 seed 为 20261018，先行两场景审计 seed 为 20261017，二者分开。W 的生成和表示诊断有冻结协议；后来在同一 dev 上追加的诊断应标为复用留出集的探索，不能追溯性称为全新确认。G5j 协议在生产读出前冻结，但仍只是当前装置下的一次 8-scene 留出评价。G5g 已明确复用旧 dev。

40 个 scene 是 W 的独立场景单位，划分 24 train / 8 tune / 8 dev；每 scene 有 9 个质量×外力组合和 ON/OFF 两模式。因此共有 720 个学习分支、6480 张图像；另有 80 个 baseline 重放分支，不进入学习。80 对等比值比较是 40 scene × 2 对，不是 80 个独立场景。时间点、组合、双模式及同场景两对均相关，不能把它们当独立重复扩充样本量。

## W：碰撞配对、别名与表示

| 可写主张 | 原始文件与复核 | 模型、单位及边界 |
|---|---|---|
| ON/OFF 从同一场景 snapshot 开始；初始图像、tool/robot 状态与各 baseline 重放精确一致 | [生产独立核验](../runs/intervention_20261012/contact_paired_production_20261013/independent_audit/validation.json)，[核验报告](../docs/causalworld_contact_paired_production_audit_20261013.txt) | 每 scene 新建环境；只切换机器人各 link 与物体碰撞，机器人仍可见。初始完整 world 信息来自 scene metadata，逐分支保存的 13D fullstates 是物体 pos/quat/线速度/角速度，不可称其单独包含完整 Bullet 世界状态。 |
| OFF 全 121 时刻机器人接触法向力为零；ON 343/360 学习分支出现正机器人接触力；全部 720 分支有正 floor 接触力 | [生产原始目录](../runs/intervention_20261012/contact_paired_production_20261013)，其中 records.json、scene_metadata.json、full_trajectories_with_repeats.npz；[报告](../docs/causalworld_contact_paired_production_audit_20261013.txt) | 正力阈值 >1e-8 N；不能把含零力 contact 项计作有效推力。body 映射来自 runtime getBodyInfo，不能凭 ID 猜。碰撞开关 metadata 是调用记录，OFF 全步力零是生效证据。 |
| OFF 的预定两对等 F/m 参数在全部 40 scene 上产生完全相同的物体轨迹 | [机制结果](../runs/contact_paired_mechanism_20261013/results.json)，[独立归约](../runs/contact_paired_mechanism_20261013/independent_recompute.json)，[报告](../docs/contact_paired_mechanism_report_20261013.txt) | 80/80 对，每对 121×13 物体状态逐值相同；质量单位 kg、力 N、位置 m、速度 m/s。不是关于任意质量/力连续取值或任意接触环境的定理。 |
| 上述 OFF 别名延续至原始图像与冻结 I-JEPA 特征 | [原始特征独立复算](../runs/contact_paired_representation_20261013/independent_audit/validation.json)，[九时刻逐值检查](../runs/contact_paired_representation_20261013/independent_audit/all9_feature_equality.json) | 80/80 对的全部 9 张 RGB、mean1280 和 grid5120 均精确相同；ON 仍有 4/80 对全部图像/特征相同。相同输入的确定性编码不能消除别名；这不是 I-JEPA 特有缺陷。 |
| ON 可以打破部分等比值别名，但不能据此宣布参数唯一可辨 | [表示结果](../runs/contact_paired_representation_20261013/results.json)，[机制汇总](../docs/contact_mechanism_batch_report_20261013.txt) | dev8 的 grid 等比值归一化距离 t30=.4644、t120=.7945、双时刻差分=.7510；OFF 均零。分母为同 anchor 与 7 个不同 ratio 格子的距离中位数，先两对后 scene 等权。t0 分母为零，记 NA，不记“成功的零比率”。 |

物理设置：mass={.01,.02,.04} kg，force={.01,.02,.03} N，friction=.175；时间步 .004 s，前 30 步在瞬时质心施世界 +x 外力，随后停止；记录图像步数 {0,1,5,10,20,30,60,90,120}，物体和机器人状态及接触逐步记录。质量改变时惯量按固定模板绝对缩放，实际值已核验。OFF 从 ON 已稳定的 snapshot 移除碰撞，属于碰撞移除反事实，可能释放初始接触/穿透；头几步保留。ON/OFF 后续机器人自身姿态也变化，因此整图 I-JEPA 差异不能全归因于物体运动。接触冲量尚未闭合，不能写成已验证的质量反演公式。

原始特征核验重算 1280 个距离、160 个 mode×pair 图像/状态比较，最大数值差 1.78e-15。这是程序正确性证据；40-scene 数据没有因此变成两份独立科学证据。编码器和源数据指纹以其 validation.json 的 input_sha256 与生产目录配置为准。

## G5j：六个监督标量读出

[协议](../docs/contact_ratio_readout_protocol_20261013.txt)、[配置](../runs/contact_ratio_readout_20261013/config.json)、[原结果](../runs/contact_ratio_readout_20261013/results.json)、[独立复算](../runs/contact_ratio_readout_20261013/independent_validation.json)。ON/OFF 各拟合三个 ridge，共六模型。输入为 grid 的 concat(z30-z0,z120-z0)，10240D；各模式分别只用 train 拟合未白化 PCA32，目标按 train 标准化。每模式 train/tune/dev 分支数 216/72/72，scene 隔离。alpha 固定 {1e-5,.001,.1,1,10}，tune 场景等权 MSE 选参，dev 只评分。

| 模式 | 目标 | alpha | dev log-MSE | train 标准化 MSE | dev R² |
|---|---|---:|---:|---:|---:|
| ON | log(F/m) | .1 | .452505 | .860181 | .139819 |
| ON | log(mass) | 1e-5 | .277648 | .866831 | .133169 |
| ON | log(force) | 1 | .200594 | .974914 | .025086 |
| OFF | log(F/m) | .1 | .171581 | .326163 | .673837 |
| OFF | log(mass) | .1 | .182154 | .568695 | .431305 |
| OFF | log(force) | .1 | .155434 | .755428 | .244572 |

正式记号应使用 ln(m/kg)、ln(F/N)、ln((F/m)/(1 m/s²))，避免对有量纲数直接取对数的歧义。log-MSE 不是 kg² 或 N²，各目标的 raw log-MSE 不适合直接比较难度；标准化 MSE 与 R² 可描述相对常数基线。平衡九格使本表标准化 MSE=1−R²。

OFF 的可观测组合读出优于单参数读出，与等比值别名相容；这不证明拟合恢复了物理因果变量。known-force 推导 ln(m_hat)=ln(F_true)−predicted ln(F/m) 增加真实外力侧信息，不是第七模型，也不是无标签发现。其质量 R² 为 ON −.412745、OFF .464315；ON 比直接质量读出更差。该估计的 log 误差恰为 ratio 误差的负数，二者同 MSE 是代数恒等式，不是额外确认。替代 SVD ridge 复算六模型，选中 alpha 完全相同，预测最大差 1.33e-15。

## G5g：条件响应子空间的有限正结果

[原结果](../runs/conditional_response_subspace_20261013/results.json)、[独立投影核验](../runs/conditional_response_subspace_20261013/independent_validation.json)、[报告](../docs/conditional_response_subspace_report_20261013.txt)。数据是原 crossed seed20261016 的 ON 装置 24/8/8，dev 已被查看，属于探索。仅 grid5120 的 t30/t120。质量/力主效应需要已知因素标签和配对响应；不是无监督发现。

初始图像的 train-only PCA4 选择 8 个训练近邻；不使用 dev 的后续响应选邻居。每因素未中心化响应 SVD，rank≤4；评分 scene 等权，但拟合高响应能量场景权重大。全局24场景与局部8场景不是总容量完全匹配；random8 固定三种 seed 是邻居数量/秩参照。局部方案对各 query 存储不同映射，总容量更高。

| 响应 | 全局 capture | 局部 capture | 局部减全局 95% scene bootstrap CI |
|---|---:|---:|---|
| t30 mass | .221235 | .285359 | [.01654,.11882] |
| t30 force | .203058 | .213697 | 跨零 |
| t120 mass | .204194 | .223281 | 跨零 |
| t120 force | .239782 | .269267 | [.00314,.07011] |

只有 8 个 dev scene，5000 次 bootstrap，没有多重检验校正或训练重拟合不确定性。最高覆盖仍约28.5%。capture 是把已知真实留出响应投影到训练子空间，不是从初态预测未知响应。dev-oracle 用真实 dev 响应拟合、实际秩2达到100%，因三水平中心化响应和为零，这是代数参照，不是发现结果。192 项投影独立复算最大差5.32e-15。

## 旧静态两参照、OOD 与 DSReg

[两参照协议](../docs/intervention_tworef_protocol_20261012.txt)、[报告](../docs/intervention_tworef_report_20261012.txt)、[ID原结果](../runs/intervention_20261012/tworef_id_eval/evaluation.json)、[OOD原结果](../runs/intervention_20261012/tworef_ood_eval/evaluation.json)。冻结 I-JEPA grid5120，训练拟合 PCA64 和 rank16 joint0 块。每个源/接收 context 使用两个真实参考图像 000/222 校正；这不是零参考编辑。块拟合使用训练干预身份，属于弱监督。

新增 ID32/OOD32 scene 各组成16对独立 scene pair，双向评价，不能写32独立 pair。主结果是颜色、跨 scene、conflicting donor 的全5120维逐编辑相对平方误差，再按预定 pair 汇总；不编辑恒为1。OOD 仅 stage/base 灰度范围从训练 [.55,.95] 改成 [.25,.50]，不是跨对象、跨相机或跨环境泛化。

| 条件 | joint0 | 两参照校正 | 不编辑 |
|---|---:|---:|---:|
| ID | 1.797587 | .742884 | 1 |
| OOD | 3.532064 | 1.397775 | 1 |

四个预注册比较使用16 pair/domain 的联合 bootstrap 同时95%区间。两参照减不编辑：ID −.257116 [−.299999,−.214233]；OOD +.397775 [.262933,.532618]。因此可说“ID改善，灰度OOD仍失败”；不能只展示相对 joint0 改善而省略更差于不编辑。27真值库最近邻是标签检索诊断，不等于生成图像、下游执行或因果编辑成功。该批最初是冻结方案的新场景确认；随后 local/tworef 和误差分解复用它，只能称探索。

DSReg 的来源必须分层。项目[目标说明](../docs/research_objective.txt)明确区分官方合成协议子集、官方 API 已知混合函数复核、真实 I-JEPA 适配，以及高阶/时序类比。不可用其中一类结果替代另一类。本文能直接逐值溯源的静态适配在 [Step2B原结果](../runs/intervention_20261012/step2b/evaluation.json)、[拟合配置](../runs/intervention_20261012/step2b/fit.json)、[报告](../docs/intervention_step2b_step3a_report_20261012.txt)：官方白化/局部解码 Jacobian/旋转 API，PCA64、512训练锚点、16×16 RGB、k96、ridge1e-3、500步、2restart。旋转不用因素标签，但之后按训练干预类别 Hungarian 分配三个16维块，整体评价不能标成纯无监督。此配置只是 pilot 适配，不是完整官方复现；文献原理论与正式复现范围由 references.md 另核验。

在新增24 scene 的 Step2B 确认上，DSReg 目标相对平方误差 .280674、非目标响应能量比 .076835，均优于仅白化 .363681/.102976。联合 joint1 为 .207584/.046419，而 response 为 .155221/.270805：降低泄漏伴随目标重建损失，不能省略取舍。DSReg 斜投影谱范数约2.75–2.86，幂等及跨块零乘积不等于欧氏正交，不能套 Pythagorean 能量分解。Step3A 在相同24 scene 上继续编辑诊断，已属探索。两参照批中的 DSReg 颜色编辑误差 ID2.722934/OOD2.568333 也只评价这一适配，不是 DSReg 理论的反例或整体优劣判决。

## 可直接用于附录的九格上限

令质量行是 (.01,.02,.04)，外力列是 (.01,.02,.03)，单位分别 kg 与 N。九格的 F/m 为

$$\begin{pmatrix}1&2&3\\1/2&1&3/2\\1/4&1/2&3/4\end{pmatrix}\ \mathrm{m/s^2}.$$

恰有七个不同的有理数。两个大小为2的等价类分别是 A={(.01,.01),(.02,.02)} 与 B={(.02,.01),(.04,.02)}，另外五格为单元素类。此计数是精确分数运算，不依赖浮点容差。

假设九格先验等权，且同一 scene 内 A 的两项输入完全相同、B 的两项输入完全相同。输入可以是本实验已验证的 OFF 图像序列、物体状态轨迹或这些图像的确定性特征。对联合九类、质量三类或外力三类任务，两个等价类内的目标标签都不同；单个确定性预测在每个二元类至多答对一个。五个单元素类至多答对五个，因此

$$\mathrm{accuracy}\le (1+1+5)/9=7/9.$$

对随机化预测，二元类两项正确概率之和也至多1，故期望准确率同样受限。此为上限，不保证达到：不同 ratio 之间可能还有别名或有限样本学习误差。证明只需要已验证的两组相同输入，不必宣称完整物理系统严格只依赖 F/m。在每个 scene 平衡九格时可逐 scene 相加，因此混合多个 scene 或按 scene 等权不会解除上限。80对只代表40scene的两组条件，不能把经验相等提升成所有连续参数或所有装置的定理。

此界不适用于给模型提供真实外力/质量/分支编号等能区分等价类的额外信息，不适用于只包含固定外力的三格子任务。t0 的全部九格输入都相同，平衡联合九类上限1/9，质量或外力三类上限1/3。上述是分类准确率界，不能拿7/9解释表中的回归 R²。

若附录讨论平方损失：每个二元类的两个 logmass（或 logforce）标签相差 ln2。同输入输出 a 时两项平方误差的最小和为 (ln2)²/2；两个类、九格平均给出 log-MSE≥(ln2)²/9，其余格的误差只能增加。这也是仅视觉且相同输入条件下的界，真实已知外力侧信息不受此界约束。

## 旧稿定位与编辑边界

已发现同标题“冻结 JEPA 表示中视觉因素与动力学机制的子空间发现及对应”的旧稿：[根目录TeX](../jepa_paper_draft.tex)、[根目录HTML](../jepa_paper_draft.html)、[根目录PDF](../jepa_paper_draft.pdf)，以及 paper/ 下的 [TeX](jepa_paper_draft.tex)、[HTML](jepa_paper_draft.html)、[PDF](jepa_paper_draft.pdf)。本任务仅定位并读标题，未修改其内容，也未将其中尚待实验的表述当证据。论文正文由 root 另行维护。


## G5k：无标签 PCA 截断与监督方向残余

[报告](../docs/observable_direction_report_20261013.txt)、[全部18项结果](../runs/observable_direction_20261013/results.json)、[独立复算](../runs/observable_direction_20261013/independent_validation.json)、[配置及依赖哈希](../runs/observable_direction_20261013/config.json)、[清单](../runs/observable_direction_20261013/manifest.json)。复用 W 的 ON/OFF 各24/8/8及 G5j 的 grid 双差分、train-only 未白化 PCA32；此时 dev 已看过，属于探索。每种 PCA 截断仍拟合有监督 ratio ridge，五个 alpha 仅由 tune MSE 选择，不能把无标签 PCA 与整个任务无监督混为一谈。

| 模式 | PCA1 R² | PCA2 | PCA4 | PCA8 | PCA16 | PCA32 | 移除监督方向后31维 | 场景内打乱训练标签32维 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ON | .000001 | .000403 | .000775 | .001568 | .069573 | .139819 | .001349 | −.011909 |
| OFF | .150796 | .373793 | .595326 | .622950 | .656541 | .673837 | .387594 | .068569 |

OFF 前4个 PCA 方向已有有限样本线性可读性，但 R² 不是信息能量比例，也不是独立物理因素发现。监督一维轴使用历史32维 ridge 权重与 tune 所选 alpha，投影后乘回权重范数，预测与原模型等价；该恒等不意味着一维训练容量或新概念发现。移除系数的欧氏方向并不保证消除与标签的协方差，OFF 残余可读性不支持唯一概念轴。训练标签负控只在各 train scene 的9分支内部用 seed20261013 打乱，ON/OFF同排列，tune/dev仍是真标签；只一个 seed，不是完整置换检验，OFF .068569 必须保留。

18模型经替代 SVD 读出及折叠回原始特征的预测复核；后者最大差6.66e-16。每scene误差和配对区间全存档。所有区间只基于8个dev scene、固定训练模型的5000次bootstrap，无多重检验校正，不包含重训不确定性，不能称新确认或显著发现。该方向诊断只覆盖 PCA32 保留部分，不是原10240维全部信息的检验。

## G5l：有限 LEACE 协方差擦除

[先冻结协议](../docs/leace_observable_protocol_20261013.txt)、[报告](../docs/leace_observable_report_20261013.txt)、[全部4模型与损伤](../runs/leace_observable_20261013/results.json)、[独立核验](../runs/leace_observable_20261013/independent_validation.json)、[12项清单](../runs/leace_observable_20261013/manifest.json)。清单12项已于本条追加前逐项重算 SHA256 一致。LEACE 已实际完成，不再处于建议未运行状态。

依据 [LEACE 原文连续目标扩展§4.3](https://arxiv.org/html/2306.03819v4#S4.SS3)及[官方实现](https://raw.githubusercontent.com/EleutherAI/concept-erasure/main/concept_erasure/leace.py)，采用连续 ln(F/m) 的 float64 解析仿射擦除，无收缩、无 trace 约束，伪逆阈值固定为机器精度规则。保留旧 PCA32，以前16个train scene拟合擦除、后8个train scene训练probe；原表示和擦除表示均用相同后8scene probe、原五alpha/tune8选参、dev8评分，共4模型。PCA使用全部24train无标签统计，是train内部共享统计的半转导拆分，不是dev泄漏。此设计避免同一批样本上消协方差后重训probe必然为常数的循环验证；8scene probe的样本量限制仍在。

| 模式 | 匹配原表示 R² | LEACE后 R² | LEACE减原 log-MSE | 探索性95%配对scene区间 |
|---|---:|---:|---:|---|
| ON | .047878 | .015890 | .016828 | [−.014654,.048837] |
| OFF | .593671 | .035275 | .293749 | [.231300,.348856] |

可写成“OFF 有限样本下的线性 ratio 可读信号明显降低”，不能写完全概念消除。两模式擦除变换均 rank31、幂等，fit16 的交叉协方差近数值零是构造保证；OFF dev 交叉协方差范数仍为原来的 .315767，dev 相对特征平方损伤为 .161453。损伤分母是原特征距 fit 均值的平方范数；没有其他任务保存度或非线性探测证据，不可声称无损擦除。ON 的配对区间跨零。该方法使用 ratio 标签，不是无监督解释，也不解除质量与外力的等比别名。

独立标量闭式对照白化—投影—反白化公式，核验变换rank、幂等与消协方差；四probe以SVD重算，并从保存映射/预测复核指标及两个配对区间。G5k用24scene probe而G5l用8scene probe，因此不能把两批残余分数直接排名；G5l的matched raw对照才是有效比较。旧dev复用、固定模型下8scene bootstrap5000次、无多重校正，均保留为探索性证据。只执行这一轮固定设计，未继续调秩、seed或迭代擦除。
