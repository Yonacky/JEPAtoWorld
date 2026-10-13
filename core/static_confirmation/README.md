# 静态确认的原始运行依赖

两脚本按实际运行版本原字节保存。generate_static_confirmation_20261013.py会验证生成器规范化源码hash，不应静默修改此文件。提取器允许全评估split={2}；不能无说明替换成只支持train/tune/test齐全的旧入口。

部署既有runner时，将这两文件复制至 `<study>/workspace/scripts/` 下同名路径；`<study>/workspace/core/e03/device_utils.py`对应仓库`core/e03/device_utils.py`。另需CausalWorld运行环境和I-JEPA官方vendor/checkpoint，仓库不含这些大制品。协议与具体参数见docs/nonorthogonal_newdata_protocol_20261013.md。历史运行source/config哈希不随此源码整理改变。
