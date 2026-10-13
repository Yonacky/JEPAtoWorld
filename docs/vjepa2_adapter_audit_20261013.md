# G-7b V-JEPA 2 标准视频接口审计

2026-10-13；所属 G-7 / R2-5；mainline_local_fourway。已完成官方源码审计、适配器语法与输入合同核验；本代理没有远程GPU执行。真实权重加载、16帧前向与物理重放由root执行，未以静态代码阅读代替成功运行。

## 可执行选择

固定标准 V-JEPA 2 ViT-L/16、256分辨率、自监督预训练 checkpoint，不用2.1、2-AC或分类微调。官方[README固定版本](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/README.md)模型表给出[vitl.pt](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt)，约300M参数；实际字节数和SHA256由下载后本地文件记录。源码固定 commit `204698b45b3712590f06245fbfba32d3be539812`。

选16个真实RGB时刻 steps=0,8,...,120，dt=.004s，间隔.032s，有效31.25Hz，覆盖0至.48s。不能用原9个非均匀时刻重复/插值凑视频。16帧为本实验短窗设置，不能说复现官方64帧评测协议；模拟器的短时尺度、场景外观与预训练自然视频存在分布差别。

[适配器](../scripts/family_video_adapter.py)直接构造官方 `vit_large`，保留官方 num_frames=64/RoPE配置，但 forward 的实际16帧产生8个tubelet、16×16空间网格，共2048个1024维token。接口输入为 B×3×T×256×256，输出 B×(T/2×256)×1024；没有CLS。使用最后一层LayerNorm后的编码器token，不调用预测器、动作模块或分类头。

## 短窗支持的源码证据

[vision_transformer.py](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/vision_transformer.py)的forward从实际输入取得T，除以tubelet_size，再将实际时间/空间网格传入每个block；[patch_embed.py](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/utils/patch_embed.py)为时间核/步长2和空间16的Conv3d，展开顺序为时间、高度、宽度。RoPE路径没有固定长度可学习pos_embed。故源码上支持偶数16/32帧；32帧预期4096×1024 token，不需要更改权重。

这只是接口支持，不是16/32帧表征质量保证。本轮不增加32帧比较：含0和120两端的32个整数等间隔时刻无法存在，因为120不能被31整除。若未来独立冻结32帧，应明确选择0..124 stride4并延长物理时间，或不同起止/窗口；不能悄悄延长本轮120步协议。奇数31帧会由tubelet卷积丢尾，适配器明确拒绝。

[modules.py](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/models/utils/modules.py)的RoPEAttention用实际网格构建位置，非因果注意力。每个时间token可包含全窗口信息，不能拿其时间索引评价“尚未看到未来”的预测。官方还保留RoPE频率展开的兼容性行为并解释修复会破坏预训练兼容；本次不自行修正。

## 载入陷阱与预处理

固定版本[backbones.py](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/hub/backbones.py)确实将VJEPA_BASE_URL指向localhost:8300，官方公网地址被注释；且以strict=False载入。不可直接调用pretrained=True下载。标准模型默认权重键是target_encoder；适配器显式从root已下载的vitl.pt取该键，不自动回退encoder。只去module./backbone.前缀，以及校验形状后的已知legacy pos_embed；剩余键以strict=True加载，任何未预期缺失、尺寸错误均停止。此过程不创建训练预测器，避免无用显存/内存开销。

按[官方冻结单视图transform](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/evals/video_classification_frozen/utils.py)构建：短边resize为int(256×256/224)=292，bilinear，中心裁256，ClipToTensor转C×T×H×W并除255，RGB均值(.485,.456,.406)、标准差(.229,.224,.225)。无随机crop/flip。适配器直接复用源码transform，传uint8 RGB ndarray序列，固定这条路径；不要将已归一化浮点再送ClipToTensor。hub预处理默认还包一层单元素视图list，适配器直接使用内部确定性Compose避免混淆。

推理eval/inference_mode、FP32，无autocast，禁TF32；设备auto，仅使用root外部CUDA_VISIBLE_DEVICES限制的单卡。依赖原生torch/torchvision/timm/numpy/PIL及官方源码；不需要升级旧transformers，也不调用torchhub网络接口。torch.load使用weights_only=True/mmap，若checkpoint安全加载不兼容，保留失败交root处理，不自动回退任意pickle加载。

## 最小2scene审计与输出

root的[均匀视频协议](family_uniform_video_protocol_20261013.md)和独立生成副本只改变渲染保存时刻，不改积分、动力学、力、碰撞/恢复流程。先复用seed20261017的2scene，每scene18学习分支+2重放。2scene只有接入审计用途，不训练读出；生产seed20261018仍是旧40scene，不能改称新独立确认。

新旧生成必须对每分支比较全部121时刻tool和robot状态逐值相同，共享step0/120图像逐值相同；也检验新16帧baseline重放及OFF两等比对逐像素相同。渲染函数原本验证tool状态未改变，但不足以单独排除robot/solver影响，因此以新旧完整状态逐值比较为验收。仅改变STEPS后还须改学习数据总数断言，2scene应576学习图像，40scene应11520；episode_length=121保留物理长度，实际采样序列另存。

适配器将同episode16帧按frame稳定排序，拒绝缺帧、重复、非均匀采样和分支内scene/split/标签变化；16帧还必须是冻结0:8:120。读取images一次。每clip保存raw像素和实际预处理张量SHA256；预定等F/m对分别记录raw/processed及输出完全相同与最大差，不继承旧9帧的相等结论。首clip同设备重复前向检查数值重复性。完整物理audit由root另行维护。

固定主读出向量为最后层全部时空token均值1024D，键branch_vector。另保存时间均值+固定2×2空间均值grid4096、每tubelet空间均值8×1024供审计，不在看到dev后挑池化。sample_manifest每episode一行，保留scene/split/mode/质量和力levels、物理labels、source_rows和clip_frames；这些标签仅下游监督诊断，不输入编码器。输出config含源码commit、权重SHA、adapter及依赖源文件SHA、精度、形状、全部时间点、实际输入相等性；manifest覆盖所有输出。源码支持32帧接口但本轮科学协议只冻结16帧。

远程调用（root以真实绝对路径替换变量，先只1个episode烟测，再全36个学习clip；两个输出目录必须不同且不存在）：

```bash
CUDA_VISIBLE_DEVICES=<authorized_GPU_UUID> <python> scripts/family_video_adapter.py \
  --source <FAMILY_ROOT>/vendor/vjepa2 --checkpoint <FAMILY_ROOT>/weights/vitl.pt \
  --dataset <uniform_audit2>/dataset.npz --output <fresh_smoke_output> \
  --frames 16 --device auto --max-episodes 1
```

随后去掉--max-episodes，输出另一目录；两次都必须严格加载成功、finite token和预期形状。脚本失败不覆盖旧产物，保留失败目录。首次只做入口检查，不以训练或dev评分选择帧数。

## 同帧I-JEPA对照与解释界限

I-JEPA必须重提取同一16帧，而不是拿原9时刻或G5j双差分当同观测预算对照。固定主聚合为逐帧最后层空间均值再作时间均值；保留逐帧/空间池化原值，以统一的train-only PCA≤32和同五alpha读出比较。I-JEPA与V-JEPA的原始维数不同、预训练数据/容量不同，应列明而非宣称模型公平排名。

仅“同帧”仍不等于同空间视野：V-JEPA有292→256中心裁剪。建议两者使用相同的确定性几何视野，再按各自原生尺寸/归一化编码；若保留I-JEPA历史独立crop，必须单列该预处理差异，不把分数差归因时序建模。以root最终冻结空间策略为准，禁止看dev后选裁剪。另列单帧I-JEPA基线用于区分额外16帧观察信息与编码器差别；mean聚合会丢顺序，不足以代表最佳图像序列模型。

相同RGB喂确定性编码器仍相同，不是视频JEPA失败。真实新问题是非别名输入的组合可读性和子空间组织。全片段参数读出是回顾性任务，不等价未来预测或规划。16帧/0.48s短窗口、机器人运动和单fixture限制继续适用。

## 本地验证记录

family_video_adapter.py通过py_compile，模块导入不触发torch/模型加载。合成合同检验：两个完整均匀16帧episode通过；非均匀时间、重复frame、同episode混split全部拒绝。真实checkpoint尚由root下载并烟测；本报告不将该项标为通过。

执行审计补充：使用共享core/e03/device_utils.py（远程可同目录放device_utils.py）的activate_device，在外部可见设备中选择可用空闲内存最多者；保存实际UUID、名称、总显存及峰值allocated/reserved。首clip重复FP32最大差必须≤1e-6。COMPLETE.json区分PASS_SMOKE_ONLY与PASS_FULL_CLIP_AUDIT，记录episode总数、检查pair数及重复误差；完整输入须覆盖每scene两个OFF对，实际raw/processed相等且输出差≤1e-6，逐值exact另行报告。window_spec.json供G7c显式读取，主向量冻结all-token mean1024。G7c原loader只接受旧9帧数据，其均匀16帧source_rows/clip_frames来源验证仍须对接，不能伪称已全链路跑通。
