# 已实现的复现命令

项目根目录运行；GPU命令串行，数据/权重/上游源码/环境按来源清单准备，未提交Git。当前无通用runner已完成声明，以下是实际入口。

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

OptiPMB诊断重放校验结果与未插桩运行完全一致；正式计时包含真实LiDAR读取/变换，排除诊断hash和报告磁盘写入。GRAE原生输出挂有限值检查，最终性能测量另行关闭Hook。GPU候选、KernelBench、配对基准和pooling命令将在实际实现后补充。
