# JEPAtoWorld

冻结视觉表示的时序结构探索：PCA 白化、SFA、联合分块、实 Schur 和双基 SVD，以及低维读出与分块动力学预测。

唯一维护仓库为 `F:\2610\CVPR27`。GitHub 同步核心代码、协议、运行说明和 Markdown 论文写作区及轻量结果摘要；原始数据、模型、完整历史结果和归档保留本地。只有用户明确要求时才执行 push，不自动推送。

当前研究初稿见 [paper/manuscript.md](paper/manuscript.md)，独立参考文献登记见 [paper/references.md](paper/references.md)。

## 计划与协作

[研究总计划](docs/MASTER_PLAN.md)是统一进度入口，按高层到低层排列并保留完成记录。所有智能体开工前读取并遵守[AGENTS.md](AGENTS.md)，执行和收尾时更新负责的计划行。

## 目录

- `core/e03/`：当前实验核心及独立验证器；保留已验证的数值实现。
- `configs/e03.json`：三段固定划分、方法、指标与解释边界。
- `scripts/run_e03.py`：创建独立运行目录及分阶段执行入口。
- `docs/code_origin.json`：核心代码迁移前的来源与哈希。
- `experiments/`：历史运行快照，不改写，不进入 Git。
- `paper/`：本地论文文件；根目录旧文件暂留作迁移备份。
- `tools/legacy/`：从临时目录收回的旧报告生成器，仅供追溯；带旧路径且会修改报告，不直接重跑、不发布。

## 运行

目标环境为 Linux、Python 3.10 和 CUDA GPU。安装 `requirements.txt` 中依赖。模型及第三方源码不随仓库分发，需要遵守其原始许可。`--assets-root` 指向已有资源目录，其中应有 `models/` 和 `vendor/ijepa-main/`；模型内部路径以 `core/e03/features.py` 为准。`--data-root` 下应存在 CITRIS 与 iCITRIS 解压目录，具体文件见 prepare 脚本。

```bash
python scripts/run_e03.py init --run-dir runs/e03_trial
python scripts/run_e03.py test --run-dir runs/e03_trial
python scripts/run_e03.py prepare --run-dir runs/e03_trial --data-root /path/to/datasets
python scripts/run_e03.py features --run-dir runs/e03_trial --assets-root /path/to/assets
python scripts/run_e03.py fit --run-dir runs/e03_trial
python scripts/run_e03.py validate --run-dir runs/e03_trial
```

init 拒绝已有目录，避免覆盖运行。脚本不依赖 `.codex`，不包含服务器口令。数学自检需安装 torch，可自动在 CPU 上运行并记录设备；默认在可见且可分配张量的 CUDA 设备中选择空闲显存最多的一张，否则使用 CPU；使用 `--device cpu` 或 `--device cuda:N` 可手动指定，`CUDA_VISIBLE_DEVICES` 可限制候选卡。需要 float64 的统计计算暂不使用 MPS。特征仅在支持 BF16 的 CUDA 上启用 BF16，其余使用 FP32，实际设备与精度写入记录。当前协议固定为两个数据集、四种表示和三个区段，不要仅修改 JSON 数量就认为循环已泛化。现有空间检查保留约 100 GiB 余量。

## 证据边界

这是静态 I-JEPA 的探索性实验，不是视频 JEPA 或因果因素发现的完成结论。特征默认 BF16 提取，后续 PCA 使用 float64；精度敏感性仍需配对审计。三个时间段不是三个独立环境。Pinball 圆柱激活按连续量处理。实 Schur 为准上三角形式，不能解释为独立动力学块。低维读出中的坐标选择使用训练标签，不能称为完全无监督评估。

历史运行的 16 项数学/数据检查和独立指标核验仅证明对应快照；代码整理后的验证状态见实际执行输出，不将历史通过视作重跑通过。

