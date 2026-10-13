# G7i-data 新静态确认数据生成与审计

2026-10-13；所属G-7i/I-4。父冻结协议：[非正交新数据确认](nonorthogonal_newdata_protocol_20261013.md)。本任务只生成/审计数据，不重拟合或选择候选，不连接GPU；root远程执行与提取。

新ID为64scene、seed2026101301、stage_gray范围[.55,.95]；新背景OOD为64scene、seed2026101302、范围[.25,.50]。每scene原27个3×3×3因素状态，合计3456张128×128 RGB、每域5184个无向单因素配对。所有split=2，只用于冻结方案评分。仍是原物体、相机与背景外推，不称跨物体/环境验证。

## 薄包装与冻结源

[generate_static_confirmation_20261013.py](../scripts/generate_static_confirmation_20261013.py)不重写静态模拟逻辑。显式读取原generate_tworef_id.py，要求规范化源码SHA256为f8f4ba66c319ca50c74ee66819f11390ff9725afa5658b746324ca2710b99e81；定稿wrapper SHA256为5061a96ed626e66059a9358f91f3cff80ba0610d165203278182e881bee89ac4。只替换原CLI输出路径/scene数、三个相同seed常量和一处背景范围，并增加持久化审计存储。每处修改要求原字符串唯一存在，否则停止，不静默适配未知旧代码。

原fixture、因素levels、每scene随机draw顺序、一次创建env、所有setter（包括mass/friction分别处理的legacy修正）、相机与像素转换不变。不会新加step、render或随机draw：原第二次render保存为repeat_images，原ABA render保存为aba_images；已有render前后snapshot及实际Bullet dynamics/visual记录保存审计。薄包装执行前写legacy_generator_snapshot.py、effective_generator.py、launch_config_before_render.json及pre_generation_hashes.json；运行失败写FAILED.json。输出目录必须不存在，禁止覆盖旧数据。

新增audit_raw.npz包括逐帧第二次渲染图、每scene返回000图与状态、全部55render/scene的前后状态；actual_bullet.json保存每次set_factors的实际质量/摩擦/惯量/颜色/contact距离。prepared.npz保持原顺序，images/labels/split完全沿dataset，episode=scene、frame=codes@[9,3,1]、episode_length=27、target_names原样，供旧提取器使用。额外审计数组不参与模型输入。

参数化调用：

```bash
<study>/runtime/bin/python <scripts>/generate_static_confirmation_20261013.py \
  --legacy-script <study>/workspace/scripts/generate_tworef_id.py \
  --output <newroot>/id_data --seed 2026101301 --scenes 64 --background 0.55 0.95

<study>/runtime/bin/python <scripts>/generate_static_confirmation_20261013.py \
  --legacy-script <study>/workspace/scripts/generate_tworef_id.py \
  --output <newroot>/ood_data --seed 2026101302 --scenes 64 --background 0.25 0.50
```

不传--prepare-only才执行真实生成。本地准备测试仅编译有效源码，未模拟；其早期准备快照在prepared字段补充前生成，保留历史但不能拿它代替定稿wrapper/实际运行的source hash。

## 旧来源与预检

[preflight.json](../runs/g7i_generator_prepare_20261013/preflight.json)扫描旧study下24份config JSON，没有命中两个新seed；另搜索docs/runs旧配置文件名与内容未命中。按同一独立RNG预计算新128context，与7份旧context共256scene比较，最大坐标差≤1e-12的重复数为0，新ID/OOD互相也无重合。预检不能代替生成后实际context检查。

必须作为--old-contexts输入的七份文件相对于旧study根：pilot/contexts.json（48，包括旧train32/tune8/dev8）、confirmation/contexts.json（24）、step3b_data/contexts.json（32）、repair_data/contexts.json（32）、oat_data/contexts.json（56）、tworef_id_data/contexts.json（32）、tworef_ood_data/contexts.json（32）。旧数据不加入新评分。context比较含y/yaw/stage_gray及完整9关节，不使用新旧局部scene编号判断重合。

## 独立审计入口

[verify_static_confirmation_20261013.py](../scripts/verify_static_confirmation_20261013.py)只读取保存数据，不信producer validation flags。要求64scene各有完整且严格有序27格、两参照000/222存在、labels与codes一致、split全2；独立重造81无向pair/scene与像素改变量，要求每个原单因素pair至少5像素不同；检查非目标state逐项不变、控制factor实际值、robot速度/物体速度0、质量/摩擦、Bullet实际视觉颜色和无明显穿透；逐像素比较全部repeat和ABA，逐状态比较render前后、render记录与主dataset及ABA状态。保存证据支持的是有限静态图像干预，不是动力学。

实际“没有调用step”仍来自源代码中的monkeypatch禁止机制，独立saved-data脚本不能重建完整执行调用历史；报告保留该边界，不把状态相同冒充完整执行轨迹。生成时至少一次初始化可能执行物理，禁止的是初始化之后的匹配渲染阶段。

审计同时从seed与冻结范围重建context RNG序列，并与全部7旧context及另一新域比较；对prepared逐字段逐值验证。每次审核输出目录必须不存在。两域生成完后调用：

```bash
<python> verify_static_confirmation_20261013.py \
  --data <newroot>/id_data --output <newroot>/id_independent_audit \
  --seed 2026101301 --scenes 64 --background 0.55 0.95 \
  --old-contexts <study>/pilot/contexts.json <study>/confirmation/contexts.json \
    <study>/step3b_data/contexts.json <study>/repair_data/contexts.json \
    <study>/oat_data/contexts.json <study>/tworef_id_data/contexts.json \
    <study>/tworef_ood_data/contexts.json \
  --peer-contexts <newroot>/ood_data/contexts.json
```

OOD对应seed2026101302、background .25 .50、data/output换ood、peer换id。读取images/audit只解压一次，prepared逐值核验需单独解压一份副本。父pipeline路径为experiments/separation_confirmation_20261013。提取沿旧extract_confirmation_features.py与e03_fp32_20261011，不改变预处理。

本地入口语法编译、旧源哈希与两配置有效源码编译已通过。root已在远端开始生成，实际完整网格、ABA/状态/可见性等结果待生成后审计，当前不提前标PASS。
