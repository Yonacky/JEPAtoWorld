# G-7d 同视频观测窗口 I-JEPA 基线准备

2026-10-13，mainline_history。仅准备代码与CPU合成检查，未运行真实编码器或GPU。入口为`scripts/family_ijepa_video_baseline.py`，不修改历史提取脚本。

固定输入为uniform数据11520帧，720个branch，各16帧step=0,8,…,120。按episode收集、按frame排序，验证完整source_rows覆盖、每branch标签/scene/split/mode/level恒定、scene不跨split且无重复branch；不假定输入文件行序。源step对应回顾性已发生片段，不作未来预测。输出`features.npz`主`branch_vector[720,1280]`是逐帧空间mean再16帧等权时均值；另存时均值grid5120和逐帧mean。该基线丢弃时间顺序，不等于时序编码器，也不同于既有仅step0/30/120的两个差分grid拼接基线。

几何与[锁定V-JEPA2源码](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/datasets/utils/video/transforms.py)一致：ndarray RGB uint8，OpenCV INTER_LINEAR短边292，长边按官方整数截断规则，中心256按Python round取整。全程保留原uint8路径，不做浮点恢复到uint8的额外量化。随后复用`complex_spatial_features.extract(images, model_root, device, batch)`：FP32除255、bilinear antialias到224、[-1,1]归一化、原ViT-H/14严格全键载入与SDPA前向误差检查。输入图像采用惰性batch裁剪，避免完整256帧副本占用约2.3GB内存。

`model_root`要求含`vendor/ijepa-main/src`与`models/ijepa_vith14_1k/model.safetensors`；`root`要求含`core/e03/device_utils.py`。设备由既有activate_device策略选择，根任务设置唯一授权GPU可见性；FP32关闭TF32、无autocast。输出config记录checkpoint、vendor、提取脚本、数据哈希、设备、版本及实际验证结果。

```
python scripts/family_ijepa_video_baseline.py --dataset UNIFORM/dataset.npz --model-root MODEL_ROOT --root PROJECT_ROOT --output NEW_OUTPUT --device auto --batch-size 16
python scripts/family_ratio_readout.py --data UNIFORM/dataset.npz --features NEW_OUTPUT/features.npz --dataset-manifest NEW_OUTPUT/sample_manifest.npz --window-spec NEW_OUTPUT/window_spec.json --input-kind branch_vector --output NEW_READOUT
```

CPU验证入口`scripts/test_family_ijepa_video_baseline.py`；读取固定官方源码的原Resize/CenterCrop类和功能函数，核对六种方形/非方形/奇数边长图像逐像素一致；随机打乱11520行后重建720×16映射；拒绝重复时间、branch内split改变和scene泄漏；时均值与独立float64 reduction比较。结果与源码hash见`runs/family_ijepa_video_baseline_validation_20261013/validation.json`及manifest。检查只覆盖准备逻辑，不预称真实模型运行成功。

此比较只控制观测时刻与几何视野。I-JEPA最终224输入与V-JEPA2的256输入、第二次插值、归一化、预训练、架构和池化均有差异，不能称控制全部变量。需要比较统一下游目标和同split/读出预算，不能比较两模型原始latent距离绝对值。当前40scene仍为探索性旧数据。

## 实际执行后元数据故障与恢复（2026-10-13）

根任务完成11520帧GPU提取并保存features.npz和sample_manifest后，config中显式pooling与展开validation的同名键冲突，抛出TypeError。原合成测试未覆盖真实元数据合并，这是本入口缺陷。已将extract返回值保存在extraction_validation嵌套字段，失败版源码保留于验证目录failure_snapshot。已有特征不能重提取。

独立CPU脚本scripts/recover_ijepa_video_metadata_20261013.py要求原dataset、现有output、model-root、实际original-source与failure-log。核对完整mapping、已有manifest、FP32有限值、时均值与grid一致，以及恢复前后特征SHA；补config/window_spec/COMPLETE/manifest。真实提取未持久化的SDPA误差、设备版本/峰值均不重构数值，明确null与原因；只依据原控制流说明extract返回前存在strict断言，不冒称恢复重新测过。记录实际失败源码、失败日志和已存特征SHA。完整720×16合成恢复通过，修复后的AST结构检查通过；真实恢复由根CPU执行并核验。证据metadata_recovery_validation.json。
