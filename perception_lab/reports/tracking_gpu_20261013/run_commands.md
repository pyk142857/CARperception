# 已实现的复现命令

项目根目录运行；GPU命令串行，数据/权重/上游源码/环境按来源清单准备，未提交Git。提供轻量阶段入口 `tools/run_tracking_gpu_20261013.py`；下面保留直接执行入口。

```bash
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/tracking_gpu_inputs_20261013.py
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/grae3dmot_2025.py --phase run --input frozen_reference --config perception_lab/configs/tracking_gpu_20261013.json --output-dir reports/tracking_gpu_20261013/native_compare/grae
perception_lab/envs/sparse4d/bin/python perception_lab/tools/grae3dmot_2025.py --phase evaluate --input frozen_reference --config perception_lab/configs/tracking_gpu_20261013.json --output-dir reports/tracking_gpu_20261013/native_compare/grae
perception_lab/envs/sparse4d/bin/python perception_lab/tools/grae3dmot_2025.py --phase audit --input frozen_reference --config perception_lab/configs/tracking_gpu_20261013.json --output-dir reports/tracking_gpu_20261013/native_compare/grae
perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/optipmb_tracking_20261013.py --validate
perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/optipmb_tracking_20261013.py --diagnostic
perception_lab/envs/sparse4d/bin/python perception_lab/tools/tracking_gpu_evaluate_20261013.py --model optipmb
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/tracking_gpu_geometry_20261013.py
```

OptiPMB诊断重放校验结果与未插桩运行完全一致；正式计时包含真实LiDAR读取/变换，排除诊断hash和报告磁盘写入。GRAE原生输出挂有限值检查，最终性能测量另行关闭Hook。完整序列基准单独计时，不复用插桩验证的速度。

## GPU几何与正式计时

```bash
# 已有环境/资源清单为前提；全部从项目根目录运行。
python perception_lab/tools/run_tracking_gpu_20261013.py profile
python perception_lab/tools/run_tracking_gpu_20261013.py kernel
python perception_lab/tools/run_tracking_gpu_20261013.py full_eval
python perception_lab/tools/run_tracking_gpu_20261013.py benchmark
# 仅在B–E结果发布、GPU空闲后运行最后的单一pooling候选
python perception_lab/tools/run_tracking_gpu_20261013.py bev_pooling
```

`kernel` 使用独立TileLang环境，CUDA12.2/C++17编译SM89；已记录CUDA12.2模板兼容补丁，先按[环境说明](environment/README.md)准备。`benchmark`使用原GRAE环境的Torch，ctypes直接加载本进程kernel；不安装或升级原GRAE的Torch。原生复跑可用`native_compare`，输入检查用`inputs`。

原生评价的最终协议绑定入口是 `envs/sparse4d/bin/python tools/tracking_gpu_evaluate_20261013.py --model grae|optipmb|grae_gpu`（从perception_lab目录运行，竖线表示三种可选值）。

单独重测：
```bash
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/tracking_gpu_benchmark_20261013.py --phase full
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/tracking_gpu_benchmark_20261013.py --phase micro
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/tracking_gpu_kernelbench_20261013.py --gpu --trials 3
```

基准不导入GT，30轮每轮分别完整读取39/81帧冻结JSON，按真实scene重置原生predictor。GT仅用于随后官方评估与可视化。

## Completed pooling follow-up

Conditional full-model build and actual shell/Python wrappers are archived under `bev_pooling/execution/`. See `bev_pooling/isolated_full_model_plan.md` for environment/build/evaluation commands. Local benchmark uses `envs/optipmb_20261013`, full model uses `envs/bevfusion`, official evaluation and summarization use `envs/sparse4d`.

```bash
perception_lab/envs/sparse4d/bin/python perception_lab/tools/bev_pooling_preindex_summary_20261013.py
python perception_lab/tools/tracking_gpu_report_20261013.py
python perception_lab/tools/tracking_gpu_acceptance_20261013.py --scope all
```
