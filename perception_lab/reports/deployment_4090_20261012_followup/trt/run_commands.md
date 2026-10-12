# Actual commands (repository root)

All commands completed with return code 0. `prepare` compiles CPU code; `mark` only edits an independent ONNX copy. GPU operations ran after the coordinator granted exclusive access. The unrelated existing GPU process was not changed.

```bash
source perception_lab/tools/bevfusion_trt_env.sh
perception_lab/envs/tensorrt/bin/python perception_lab/tools/bevfusion_trt_followup.py prepare
perception_lab/envs/tensorrt/bin/python perception_lab/tools/bevfusion_trt_followup.py mark
perception_lab/envs/bevfusion/bin/python perception_lab/tools/bevfusion_trt_followup.py run --library production
perception_lab/envs/bevfusion/bin/python perception_lab/tools/bevfusion_trt_followup.py run --library diagnostic
perception_lab/envs/tensorrt/bin/python perception_lab/tools/bevfusion_trt_followup.py build-head
perception_lab/envs/tensorrt/bin/python perception_lab/tools/bevfusion_trt_followup.py replay
perception_lab/envs/bevfusion/bin/python perception_lab/tools/bevfusion_trt_followup.py cases
python perception_lab/tools/bevfusion_trt_followup.py analyze
```

`prepare` requires the existing successful production build object files and pinned third-party sources; it changes only a copied wrapper source and links to a separate directory. Production source/library/engine files are read-only. Standard ONNX checker does not understand the existing NVIDIA CustomLayerNormalization node; the TensorRT parser/build validated the diagnostic graph successfully.

Local logs: `runs/deployment_4090_20261012_followup/trt/{production,diagnostic,head.diagnostic.build,replay,cases}.log`. Build command and diagnostic engine hash are in `head_diagnostic_build.json`; compile/link commands are in `build.json`. File hashes below cover actual logs, not a claimed performance measurement.

After reference module replay completed, `python perception_lab/tools/bevfusion_trt_followup.py analyze` was rerun to produce the high-score same-input head control. `cases` was rerun on CPU to add neighboring masked-score evidence; GPU captures were reused. No further GPU precision experiment ran.
