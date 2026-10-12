# Reference 复现命令

工作目录 `perception_lab/`，依赖原部署已固定的 BEVFusion 环境、数据和权重。该诊断不测性能。

```bash
python3 -m unittest discover -s tests -p test_bevfusion_reference_followup.py
python3 tools/bevfusion_reference_followup.py --select-only
OMP_NUM_THREADS=8 envs/bevfusion/bin/python -u tools/bevfusion_reference_followup.py > runs/deployment_4090_20261012_followup/reference/run.log 2>&1
python3 reports/deployment_4090_20261012_followup/reference/analyze_raster_collisions.py
```

已执行以上流程，GPU 进程退出码 0。五项测试通过；生产 20 次、捕获 20 次，实际命令和代码/扩展文件身份见 `execution.json`。源文件执行快照留在本地 `runs/deployment_4090_20261012_followup/reference/executed_runner.py`；其哈希等于 `identity.json` 的 runner_sha256。

TensorRT 共享回放输入就绪后，独立执行：

```bash
OMP_NUM_THREADS=8 envs/bevfusion/bin/python -u tools/bevfusion_reference_followup.py --replay-dir runs/deployment_4090_20261012_followup/trt/replay > runs/deployment_4090_20261012_followup/reference/replay.log 2>&1
```

最后一条命令是否已执行，须以 `replay/summary.json` 和日志为准，不能仅依据这里列出的命令判断。
