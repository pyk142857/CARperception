# 复现命令

工作目录 `perception_lab/`。先按模块 setup 脚本恢复隔离依赖与官方资产，权重/数据/engine 不入 Git。

```bash
bash tools/bevfusion_setup.sh
bash tools/bevfusion_trt_setup.sh --build-engines
bash tools/sparse4d_setup.sh
bash tools/gaussianworld_setup.sh
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage preflight
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage bevfusion_reference
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage bevfusion_tensorrt
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage bev_pooling
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage sparse4d
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage gaussianworld
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage visualize
python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage summarize
envs/rerun/bin/python tools/deployment_visualize.py --serve
```

各次实际 argv、工作目录、返回码与日志记录在 [runner_status.json](runner_status.json) 和 `commands/`；首次独立执行命令及环境在模块报告中。TensorRT head 必须按配置使用 LayerNorm 插件版本。
