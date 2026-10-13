# G-7j / RLS-2 依赖与评价schema只读审查

2026-10-13，mainline_history。未修改冻结脚本、git配置、索引或分支。仅使用`git -c safe.directory=F:/2610/CVPR27 ls-files`读取跟踪清单，并以AST和文本审查依赖。本记录只说明审查时状态，根任务后续暂存/提交会改变tracked状态。

## 最小可复现依赖

| 入口 | 发现 | 最小处理建议 |
|---|---|---|
| run_static_confirmation_20261013.py | 通过study/workspace/scripts调用generate_tworef_id.py与extract_confirmation_features.py，二者位于未tracked的旧runs路径 | 原字节保存为核心脚本/明确vendor快照；新runner或部署说明显式指定路径。保留原冻结runner不改写。生成器normalized BASE_SHA必须继续匹配 |
| generate_static_confirmation_20261013.py | 必须由--legacy-script读取旧生成器，再校验hash并编译替换 | 只发布wrapper不足以复现；必须同时提供上述原生成器与CausalWorld固定源码/依赖安装说明 |
| extract_confirmation_features.py旧副本 | 已允许全split={2}；tracked的complex_spatial_features.py原validate只接受{0,1,2} | 不可直接替换后称完全等价；最小是原字节核心化这个test-only提取器，同时保留旧hash。其core/e03/device_utils.py已tracked |
| family_leace_observable.py / family_observable_direction.py | 导入新family_provenance.py；另导入已tracked contact_ratio_readout_20261013.py | 新helper须随入口发布，不仅发布主脚本 |
| family_ijepa_video_baseline.py / recovery入口 | 依赖新family_video_adapter.py及已tracked complex_spatial_features.py | 一并包含adapter，外部V-JEPA2源码与I-JEPA权重/vendor按固定commit/hash单独准备，不把大权重放git |
| extract_family_images_20261013.py | 导入新family_image_adapter.py | 一并发布；标准timm1.0.19、safetensors、torchvision与Pillow依赖须说明 |
| family_anova_blocks / nonorthogonal_edit | 导入fit_target_supervised_reference_20261012.py | helper已tracked，无旧runs动态import；训练数据/冻结fitted.npz属于明确外部实验制品依赖 |
| finish_static_confirmation_20261013.py | 从部署脚本同目录寻找models.npz、selection.json、frozen_config.json | 属部署布局约定，不是干净checkout自足入口；文档应明确这三项来源与hash或新runner显式参数，勿误认为脚本会自行训练生成 |
| test_family_ijepa_video_baseline.py | 从runs验证目录读取固定官方functional.py/transforms.py，另优先该目录deps | CPU测试需小型固定源码fixture；不能依赖本机pip安装残留。保留官方许可头，可核心化测试fixture或提供精确下载/哈希准备步骤 |

本次扫描的新G7脚本多尚未tracked，属于本轮待发布集合，不等于遗漏已发生；其中需额外关注容易漏掉的helper、原生成器与test-only提取器。没有发现ANOVA/非正交/冻结评估在运行时import旧runs训练脚本。CausalWorld、I-JEPA vendor、V-JEPA2 vendor及权重均需明确外部依赖清单，不能仅凭AST无本地导入就断言完全自足。

## G7j域条件与必须保留的schema

数据agent提出shape域tool_block尺寸[.045,.060,.065]米且camera0；view域原.065立方体且camera1。各32scene，seed2026101303/1304，其他context沿原ID。水平缩小、z=.0325不变。它们是两个全域固定context变化，保持原位置/物色/地板三个可独立干预因素，不是新增第四个模型因素，也不把两个域混合成64scene。

需要在scene metadata或等价明确文件新增domain_condition、实际size_xyz、camera_id及实际view/projection/pose；dataset/prepared的codes仍3列、每scene27格，labels仍世界坐标object_x/green/floor，不将camera1屏幕x重新命名为object_x。全部27变体共享域条件，允许控制参照仍仅000/222，并必须来自同域、同camera、同scene。评价器不得读取camera0辅助参照或完整网格平均来修正camera1。

size API重建body后必须刷新每client的实际body id，再核对mass/friction、visual/collision实际几何及orientation/AABB；旋转后的世界AABB不是原size直接拷贝。三因素配对时size、camera、非目标状态应恒定；camera改变只影响渲染，render-before/after及ABA/repeat核验也应针对选定camera1。保存真实形状证据，不能只保存请求参数。若重建改变状态schema长度，需要更新状态offset/key审计，不能静默丢字段。

## 冻结评价适用条件

原G7i evaluator生产硬编码64scene，仅有id/ood名称。G7j各32scene需新的协议和入口副本/参数合同，明确16独立scene pair/domain；不能用old-alignment标志绕过，也不能把shape与view混拼。保持模型、PCA、两参照、anchor排除、单/双因素真值和原空间指标不变，新增域仅评分，不重训。

若仍采用两域×宏误差/颜色恢复四项主比较，应在新结果前明确同样98.75%区间与bootstrap单位；16pair的区间精度弱于G7i32pair，不应用编辑数量掩盖。小样本未显著不等于等效。形状变化只覆盖一个固定尺寸比例、视角变化只覆盖一个现成camera，不扩称跨类别/任意视角泛化。

以上schema建议已发数据agent与根任务。当前不判真实新域数据PASS；实际body/camera日志、源哈希、新入口及结果仍待准备与审计。
