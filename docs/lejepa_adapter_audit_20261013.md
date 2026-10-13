# G-7a LeJEPA 图像适配审计

2026-10-13，mainline_history。本轮锁定 **OK-AI 第三方 LeJEPA 复现敏感性分支**，不称原作者发布模型。未用任何下游结果挑选版本。原作者官方权重缺口保留，官方 V-JEPA 2 是根任务的另一主线。

## 来源与锁定

原论文为 Randall Balestriero、Yann LeCun，2025，[LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics](https://arxiv.org/abs/2511.08544)。[官方仓库](https://github.com/galilai-group/lejepa)在本次检查的 commit `c293d291ca87cd4fddee9d3fffe4e914c7272052` 没有可核验发布权重；releases为空，[issue 3](https://github.com/galilai-group/lejepa/issues/3)仍在请求权重。这是本次检索的发布状态，不是永久不存在的证明。官方代码标注 CC BY-NC 4.0。

采用发布者 [OK-AI/lejepa-vitb16-pretrain-in1k](https://huggingface.co/OK-AI/lejepa-vitb16-pretrain-in1k)，固定 revision `5dd71e2fc9a8e97d84c047f3a8cd5bf19ff199fa`。下载地址：

- [model.safetensors](https://huggingface.co/OK-AI/lejepa-vitb16-pretrain-in1k/resolve/5dd71e2fc9a8e97d84c047f3a8cd5bf19ff199fa/model.safetensors)
- [config.json](https://huggingface.co/OK-AI/lejepa-vitb16-pretrain-in1k/resolve/5dd71e2fc9a8e97d84c047f3a8cd5bf19ff199fa/config.json)
- [preprocessor_config.json](https://huggingface.co/OK-AI/lejepa-vitb16-pretrain-in1k/resolve/5dd71e2fc9a8e97d84c047f3a8cd5bf19ff199fa/preprocessor_config.json)

根任务实测权重 366820152 bytes，SHA256 `5dd54eaa80e3160420a9b1682b2cd2e9505aff6566d1f54b742067fef149f85a`；适配器每次核对。运行只需要前两个文件，预处理文件作为来源存档。第三方卡片标注 Apache-2.0；随附部分未使用工具文件有 DINOv3 许可头，不能把所有混合源码笼统重新许可为 Apache。适配器不执行这些第三方代码，使用 timm 标准实现。

第三方声明 IN1K、300 epochs、2 global224+6 local96、多视图 SIGReg/invariance；这不等于原作者训练流程复现已经独立认证。卡片的通用 teacher 示例与 LeJEPA 仅 student 的说明混杂，因此采用实际配置和源码判定。卡片的预训练图像数量未独立验证，不用于公平数据规模主张。

## 编码器、投影头与预处理

固定 ViT-B/16：12 blocks，768 channels，12 heads，MLP ratio4，224输入，无register、无LayerScale。197 tokens = CLS + 14×14空间tokens；读取最后 LayerNorm(eps1e-6) 后的空间tokens。mean768为空间平均；grid3072为2×2自适应平均并按C、row、column展平；CLS768只作附加输出。无额外L2归一化。原作者官方线性评价拼接末两层CLS的流程与此不同，本实验是固定空间表示适配。

预处理遵循发布者配置：RGB、PIL bicubic短边256、中心裁剪224、除255、ImageNet mean(.485,.456,.406)/std(.229,.224,.225)。不沿用既有I-JEPA的[-1,1]和bilinear处理；模型预处理不同需在跨模型比较中声明。

权重156 tensors均有backbone前缀；其中6个`cva_module_proj.mlp.{0,2,4}.{weight,bias}`是768→2048→2048→64训练投影头。依据[训练源码](https://github.com/Open-Knowledge-AI/lite_ssl/blob/6ad846c891f644f09a255dd75c8ada6fb7247057/lite_ssl/trainer/lejepa_ms_pretrain.py#L104-L107)，投影在编码器返回latent之后单独调用供损失使用。HF固定版本`forward_features`不调用此头。适配器检查这6键及精确shape后显式排除，在metadata列出；剩余150键strict载入，任何其他缺失/多余键报错。这是encoder-only载入，不是完整156键模型载入，也不把投影空间的正则性自动移植到空间tokens。

## 接口与验证

代码 `scripts/family_image_adapter.py`：`FamilyImageAdapter(checkpoint_dir, device='cpu')`；`preprocess(images)`接收uint8 HWC RGB或PIL；`encode(batch)`返回tokens/cls/mean/grid；metadata保存权重、revision、版本、排除键、预处理和输出约定。依赖torch、torchvision、timm1.0.19、safetensors、Pillow，无transformers版本升级要求，无远程自定义代码。

```
python scripts/family_image_adapter.py --checkpoint-dir CHECKPOINT_DIR --device cuda:0 --smoke-output smoke.json
```

设备必须由根任务按授权GPU设置，本代码默认CPU、不自行选卡。FP32、eval、inference_mode，无autocast；根提取runner另记录GPU/TF32设置与源数据hash。

本地CPU已完成源ViTv2到timm1.0.19随机参数的全键strict架构检查和前向比较；可复算入口与JSON在`runs/lejepa_adapter_audit_20261013/validate_architecture.py`、`architecture_validation.json`，阈值1e-5。此项不等于真实checkpoint烟测；真实权重已在根任务远程下载，strict载入/有限输出烟测由根执行并保存，不在此预称通过。源文件hash清单见同目录manifest.json。

## 解释边界

LeJEPA分布正则和理论动机不证明此第三方checkpoint、最后空间token、池化表示满足DSReg结构多样性、无抵消和线性识别等条件。后续DSReg是条件适配诊断，不是定理实例认证。不同架构/训练数据/目标/输出层限制公平排行榜解释；现有40scene仍是旧探索数据。输入等价导致不可辨识的边界也不会因更换模型而消失。
