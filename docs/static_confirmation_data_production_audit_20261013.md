# G7i-data 实际生产与独立数据审计补记

2026-10-13；mainline_local_fourway；所属 G-7i / I-4。原[入口交付记录](static_confirmation_data_audit_20261013.md)保留原样，其中“生成中”属于当时历史状态；本补记记录实际完成情况。未重跑模拟器，未修改数据或候选。

新 ID 与背景 OOD 的保存数据审计均为 `PASS_SAVED_DATA_AUDIT`。各域 64 场景、1728 图像、5184 个无向单因素配对，合计 128 场景、3456 图像。每场景完整27格、两参照000/222、全split=2、prepared字段逐值对应均通过。seed分别为2026101301与2026101302。

| 实际审计项 | ID | 背景 OOD |
|---|---:|---:|
| render前后状态最大差 | 0 | 0 |
| render记录与主数据状态最大差 | 0 | 0 |
| ABA状态最大差 | 0 | 0 |
| 非目标状态最大差 | 0 | 0 |
| 全部repeat/ABA图像逐像素相同 | 是 | 是 |
| 三因素最小变化像素数：位置/物体颜色/地板色 | 469 / 241 / 6500 | 472 / 243 / 6504 |
| 实际Bullet参数记录数 | 1792 | 1792 |

两个域各自与7份旧context（共256场景，包含旧train/tune/dev、ID与OOD）及另一个新域比较，容差1e-12内重合数均为0；新ID与新OOD最近context最大坐标差为0.09400679259849298。实际随机context与冻结seed序列核验通过。质量/摩擦与实际颜色等参数按独立审计器规则通过。

[ID独立审计](../runs/separation_confirmation_20261013/id_independent_audit/validation.json)与[OOD独立审计](../runs/separation_confirmation_20261013/ood_independent_audit/validation.json)是实际结果。回存后对两域data与independent_audit的四份manifest合计28项重新计算SHA256，全部匹配，并核对审计脚本源码SHA256与两份结果一致；见[本地收件核验](../runs/separation_confirmation_20261013/data_audit_local_receipt.json)。审计源码SHA256为c7f74c216225f0e80f4d0d8b16168422debedac6b73388155b68f08f9809d518。

这里完成的是保存数据不变量和来源核验，不是独立重演完整执行调用历史。“初始化后未step”仍依赖冻结生成源码的禁止机制；状态相同不能独立证明没有调用step。本批是原模拟器、原物体与相机下新context、原范围背景OOD；支持预先冻结候选的新场景评估，不支持跨物体/跨编码器或真实动力学结论。评分后若据此继续选择方法，该批便不再是后续方案的未见确认集。G-7i总评分与备份由root归档。
