# G-7i-audit 独立确认审查记录

2026-10-13，mainline_history。状态：旧机制与协议已审读；新评估脚本及正式协议待评估agent交付，真实数据结果未审计。此文件是审查记录，不另立总计划，不据旧结果改新超参数。

## 已核对的旧机制

G7g训练32/tune8：四个拟合候选为orth/oblique×λ0/1；按旧tune误差加泄漏选择两族λ1。PCA64、rank16三个因素块及补块、两参照000/222、cond≤2均已冻结。原空间斜投影误差应使用`||prediction-target||²`，不能套用正交遗漏加干扰恒等。旧ID32/OOD32已查看，均不得进入独立确认样本或重新选择。

新“4模型”需要在正式协议中逐名列出并保存实际矩阵hash，区分它与旧“四个拟合候选”。推荐固定已选择oblique_lambda1、orth_lambda1、joint0_twoanchors和no_edit；模型名单由父协议确定，不由审计者读取新结果后更改。

## 新确认验收条件

64新ID与64背景OOD，每域scene i和i+32组成32个不重叠pair；双向donor/recipient先平均，同一pair内所有编辑共用场景，应在pair层bootstrap，不能把128方向或数十万编辑作独立样本。两域的seed、实际context与图像/状态均需查重，seed不同本身不证明样本独立；不得将合法ID重复因素取值误判为context重复。

正式协议必须精确定义4个主contrast的“域×指标×对照×方向”；四项双侧98.75%区间对应Bonferroni α=.05/4，分位数为.00625与.99375，非.0125/.9875。固定bootstrap种子和重复数；这是近似bootstrap区间的多重校正，不是严格有限样本覆盖证明。其余因素分层、准确率、组合及安全指标若不属于主四项，应标次要描述性；不能看结果后增删主检验。优势/非劣必须预写方向及非劣界，不能用“区间跨0”宣称等效。

测试编辑仅使用recipient、donor、每scene预允许000/222两帧参照及冻结矩阵；完整测试27格只能用于匹配真值与检索评分，不能参与均值校准、拟合、路由或交互修正。目标索引可以由指定干预值构造评分真值，但不能将真值向量送入预测。两参照编码亦需来自新scene，预算明确记录。

单因素/双因素组合应统一原5120空间补残差误差、匹配干预真值、目标恢复和未编辑因素保持。斜投影要保留谱范数/噪声放大说明。组合评价须检查真实两因素joint target，不以P_iP_j=0或交换顺序相等当作可组合语义证据；同一donor提供两因素时仍需明确未编辑因素是否冲突。

冻结文件（模型矩阵、PCA、旧选择记录、旧配置）hash须在新结果前落盘，新脚本不能调用fit或以新场景改参数。评价结束重新核对这些hash。独立复算至少从原空间重建预指定编辑、重算保存预测指标、核对pair映射和CI分位数，且说明完整还是抽样范围。

## 待审

正式协议/评估源码、生成器seed/context查重、新数据特征来源、四主项定义与独立复算结果。以上条目未完成前不写PASS，也不预判科学结论。

## 冻结入口审查与旧数据独立复算完成

已审根正式协议docs/nonorthogonal_newdata_protocol_20261013.md及最终evaluate_frozen_oblique_20261013.py（SHA256 980560f508a8232e4548a5e17f42431690c80c3236a633cf9350d59042c724c4）。四模型已明确为no_edit、joint0_twoanchors、orth_lambda1、oblique_lambda1；四主contrast为ID/OOD各单因素宏相对误差与颜色目标恢复，均为oblique减orth。双因素及joint0比较仅次要。模型/旧selection起止hash核对、模型名单严格断言已加入。

静态审查：无拟合调用；预测只用source/donor及000/222参照；完整网格均值不进入offset。双因素指定两code都改变且第三code冲突，真值一次更新两code；目标向量只进入评分。anchor排除在target组合构造后，recipient/donor/target任一为000/222均排；检索保留27候选。原空间残差项使用一般二次误差，未套正交勾股分解。pair为i与i+half且双向合并，98.75%分位点正确为.00625/.99375。当前未发现阻塞性源码问题。

独立脚本scripts/audit_frozen_oblique_confirmation_20261013.py不导入评估函数：重新枚举所有edit tuple与mask，用W坐标掩码加逆方程求解，而非直接复用保存投影；完整5120维重建固定每scene/task/model前7项并对27候选逐一算距离。全部保存edit指标另重聚合，再以独立bootstrap loop核四主量。最终old_alignment_frozen的旧4scene/2pair通过：672个全空间样本最大误差1.98e-13，预测差1.15e-14，CI差5.55e-17。证据runs/frozen_oblique_entry_20261013/old_alignment_frozen/independent_audit.json。此前old_alignment_final审计也保留，但正式入口以frozen对应源码为准。

新数据到位后执行同脚本 --evaluation 新域目录；新域应有64scene/32pair。此阶段只完成程序和旧数据对齐验收，不是新确认的科学结果。新context去重、生成状态/RGB审计及真实结果复算仍待完成。
