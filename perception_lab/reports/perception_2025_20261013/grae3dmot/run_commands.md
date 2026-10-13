# GRAE-3DMOT reproduction

Run from `perception_lab/`. Resources are pinned in `resource_manifest.json`; original data, source, weights, environment and executed source snapshot are excluded from Git.

```bash
OMP_NUM_THREADS=8 envs/grae3dmot_2025/bin/python tools/grae3dmot_2025.py --phase prepare --config configs/perception_2025.yaml
OMP_NUM_THREADS=8 envs/grae3dmot_2025/bin/python tools/grae3dmot_2025.py --phase run --config configs/perception_2025.yaml
OMP_NUM_THREADS=8 envs/grae3dmot_2025/bin/python tools/grae3dmot_2025.py --phase optimize --config configs/perception_2025.yaml
OMP_NUM_THREADS=8 envs/sparse4d/bin/python tools/grae3dmot_2025.py --phase evaluate --config configs/perception_2025.yaml
OMP_NUM_THREADS=8 envs/sparse4d/bin/python tools/grae3dmot_2025.py --phase audit --config configs/perception_2025.yaml
OMP_NUM_THREADS=8 envs/grae3dmot_2025/bin/python tools/grae3dmot_2025.py --phase profile --config configs/perception_2025.yaml
```

All six commands executed successfully; profile used a separately scheduled GPU slot. `--input mambafusion_native` is a separate run and fails explicitly if detector output is absent. No GT annotation enters `prepare` or `run`; evaluation/audit load GT only after inference.
