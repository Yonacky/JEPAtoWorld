# G7j-data 尺寸比例与视角变化冻结协议

2026-10-13；G-7j / I-4；mainline_local_fourway。只准备入口，由root完成已授权push后启动远端模拟/编码。旧G7i协议及源文件不修改。

## 两个域及解释范围

shape域固定tool_block尺寸[0.045,0.060,0.065]米、camera0，seed2026101303；原尺寸为[0.065,0.065,0.065]。这同时改变水平尺度与长宽比例，不能把结果归因于单一尺度因素，也不是新物体类别。高度不变，底面支撑与中心z=0.0325保持；水平尺寸均缩小，避免主动引入更大包围盒的穿透风险，实际接触仍必须核验。

view域固定旧立方体，选模拟器原有camera1，seed2026101304；旧数据为camera0。官方本地causalworld.py中camera0位置[.2496,.2458,.58]、四元数[.3760,.8690,-.2918,-.1354]；camera1位置[.0047,-.2834,.58]、四元数[.9655,-.0098,-.0065,-.2603]。不自行计算或修改相机外参，保存运行时三台相机实际pose/view/projection和所选index。camera.py仍128×128、FOV52、near.001/far100、原渲染与翻转流程。视角会同时改变遮挡、透视与像素面积，不能解释为纯旋转等变性。

每域32scene、每scene27格，共1728图；三因素仍x=[-.075,0,.075]、green/tint=[.2,.5,.8]，两参照000/222。y/yaw/9关节/背景范围和随机draw顺序沿原ID，stage_gray[.55,.95]。域条件全域固定，不进入codes/labels；全部split=2。模型、PCA、投影、lambda及提取预处理冻结，不用新域重训或选择候选。各域i/i+16双向配对，16独立pair；指标和参照排除规则沿父G7i冻结评价，两个域都必须报告，不能仅选择通过的域。推断单位为pair而不是单图。

## 实际API和源审计

本地CausalWorld-master/causal_world/envs/scene/objects.py:484的size干预会更新size与体积并reinit_object；Box.createCollisionShape使用halfExtents=size/2，真实collision形状被重建。生成器每scene先单独size setter，再沿旧mass/.02与friction/.175分开setter；每grid直接重新取得当前body ID，避免复用已删除body。这个操作也会重建惯量；静态评估不将其误称质量动力学实验。初始化后旧禁止step机制保留。

robot/observations.py:get_current_camera_observations按camera_indicies返回原归一化图。入口要求indices确为[0,1,2]，只选择index0或1，无新物理调用或额外渲染。现有所有重复渲染、ABA、非目标状态、真实mass/friction、无明显穿透和因素至少5像素变化断言沿用；不删除不可见scene或换seed。失败保留FAILED及源配置，先排查实现，不能以通过筛选重定义样本。

新增domain_geometry.json逐次set_factors记录实际visual/collision完整尺寸、缓存size、AABB、body位置；真实尺寸与预定值一致才继续。camera_metadata.json记录运行时所选相机及全部相机参数。其他dataset/audit_raw/prepared字段与G7i一致，可复用原提取器。独立saved-data核验须沿旧verify_static_confirmation检查所有不变量，再额外核对实际尺寸、AABB与yaw/高度、相机矩阵不同及选定index。新context与旧256加G7i新128及另一域核对无重合；对比context应含实际domain条件，基本context亦另查，不依赖局部scene编号。

## 执行入口

新脚本[scripts/generate_static_domain_shift_20261013.py](../scripts/generate_static_domain_shift_20261013.py)固定复用G7i wrapper SHA5061a96ed626e66059a9358f91f3cff80ba0610d165203278182e881bee89ac4，旧legacy SHA仍f8f4ba66c319ca50c74ee66819f11390ff9725afa5658b746324ca2710b99e81。两域有效源码已在本地prepare-only编译通过，未执行模拟，不提前宣称可见性通过。

```bash
<runtime>/bin/python <scripts>/generate_static_domain_shift_20261013.py \
  --legacy-script <study>/workspace/scripts/generate_tworef_id.py \
  --base-wrapper <scripts>/generate_static_confirmation_20261013.py \
  --output <newroot>/shape_data --domain shape
# view使用 --domain view --output <newroot>/view_data
```

可先在独立输出目录加--scenes 2做每域前缀烟测，检查API/真实几何/渲染，不编码评分、不按效果改参数；其scene属于随后32scene的前缀，不得称额外独立确认。所有输出禁止覆盖。源/config哈希在模拟开始前写入，结束后写manifest。只需上传新脚本和旧G7i wrapper；legacy使用远端原路径。不需要额外模型依赖。源SHA256：930c29423e26c500ed4010ed79d79cf6f9cbc343e32d10708196a728639d413c

## 独立审计调用补充

[verify_static_domain_shift_20261013.py](../scripts/verify_static_domain_shift_20261013.py)已语法编译；用冻结旧verify_static_confirmation_20261013.py先执行共同全部检查，再核对每条actual Bullet记录的body/client/codes与几何记录、27格尺寸恒定、按yaw计算AABB宽度（不是把AABB直接当size）、两相机运行时外参与view/projection以及有效源码的渲染选择。只加载主图像一次做共同审计；后续几何检查只读state数组。实际运行仍待远端生成，不先称PASS。

```bash
<python> verify_static_domain_shift_20261013.py \
 --data <newroot>/shape_data --output <newroot>/shape_independent_audit \
 --base-verifier <scripts>/verify_static_confirmation_20261013.py \
 --old-contexts <旧7份contexts> <G7i>/id_data/contexts.json <G7i>/ood_data/contexts.json \
 --peer-contexts <newroot>/view_data/contexts.json
```

view交换data/output与peer。只保存图像不能独立反推出相机身份，审计明确依赖运行时相机参数与有效源码选择记录。形状/相机不是模型输入的附加标签；view域所有recipient/donor/anchor来自camera1，不借用camera0 anchor。原64scene评价脚本须单独参数化32scene并沿用冻结数学，不可拼域或触发旧对齐路径。
