# 复现与运行入口

从仓库根目录运行。环境、数据、权重和上游源码均按 resource_manifest / 各模型环境文件准备；模型大文件未纳入Git。当前缺完整 HARP/Mamba 权重，这两个阶段会写 blocked_resource，退出码2；不能据此声称网络已经跑通。

```bash
# 本轮入口，dry-run只列命令，不执行模型。
python3 perception_lab/tools/run_perception_2025.py --phase all --dry-run
# 实际单阶段：支持 preflight/deimv2/harpnext/mambafusion/grae3dmot/optimize/visualize/summarize。
python3 perception_lab/tools/run_perception_2025.py --phase deimv2
# --resume 只跳过具有完整产物哈希清单且输入/环境/代码相同的 passed 阶段；无清单会重跑。
python3 perception_lab/tools/run_perception_2025.py --phase visualize --resume
python3 perception_lab/tools/run_perception_2025.py --phase summarize
```

本次原生与后端实际命令（GPU命令按顺序执行，不能并发）：

```bash
perception_lab/envs/deimv2_compat_2025/bin/python perception_lab/tools/deimv2_2025.py --phase run
# 作者导出只改demo batch32→1和源码导入路径；实际执行副本和patch有哈希记录。
OMP_NUM_THREADS=8 perception_lab/envs/deimv2_compat_2025/bin/python perception_lab/runs/perception_2025/deimv2/export_onnx_batch1.py -c /home/minglei/Desktop/CAR/perception_lab/third_party/DEIMv2_2025/configs/deimv2/deimv2_dinov3_s_coco.yml -r /home/minglei/Desktop/CAR/perception_lab/checkpoints/perception_2025/deimv2/deimv2_s.pth --check
perception_lab/envs/deimv2_onnx_2025/bin/python perception_lab/tools/deimv2_onnx_2025.py
perception_lab/envs/nuscenes_eval/bin/python perception_lab/tools/deimv2_badcases_2025.py
perception_lab/envs/rerun/bin/python perception_lab/tools/perception_2025_visualize.py
```

各模型真实前置、推理、评价、优化命令见 [GRAE](grae3dmot/report.md)、[HARP](harpnext/report.md)、[Mamba](mambafusion/report.md)、[TensorRT](deimv2/tensorrt/)。固定scene0061为39帧，mini_val为81帧；COCO全部5000张；点语义库存404帧14026208点。原生DEIM运行时修改了工具未执行的optimize分支，原生路径未变；未记录整个工具的启动时hash，不冒充完全不可变的执行快照。上游源码与checkpoint逐文件hash独立固定；后续后端保存执行脚本副本。

最终补充实测/核验：

```bash
python3 perception_lab/tools/deimv2_paired_onnx_2025.py
perception_lab/envs/nuscenes_eval/bin/python perception_lab/tools/deimv2_onnx_geometry_2025.py
perception_lab/envs/deimv2_trt_2025/bin/python perception_lab/tools/deimv2_trt_2025.py --phase validate
perception_lab/envs/deimv2_trt_2025/bin/python perception_lab/tools/deimv2_trt_2025.py --phase benchmark
perception_lab/envs/deimv2_trt_2025/bin/python perception_lab/tools/deimv2_trt_2025.py --phase audit
python perception_lab/tools/perception_2025_rerun_qa.py
perception_lab/envs/grae3dmot_2025/bin/python -m pytest perception_lab/tests/test_*2025*.py -q
python3 perception_lab/tools/perception_2025_artifacts.py
```

ONNX隔离环境的 `vision_overlay.pth` 指向已有vision site-packages以选中GPU ORT，自己site-packages的Pillow12.3优先；旧vision环境没有修改。原生/ORT配对完成后新增的权重/配置/ONNX身份防护仅在比较前检查文件，不改变推理或计时路径；实际文件通过[事后身份核对](deimv2/backend_identity_verification.json)。未来export记录明确输入和ONNX绑定。
