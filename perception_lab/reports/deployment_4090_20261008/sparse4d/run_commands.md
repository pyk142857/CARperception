# Reproduce from repository root
```bash
bash perception_lab/tools/sparse4d_setup.sh
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_4090.py --packets perception_lab/reports/deployment_4090_20261008/inputs/scene0061_packets.json --out perception_lab/outputs/sparse4d/scene0061 --state-tests
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_4090.py --packets perception_lab/reports/deployment_4090_20261008/inputs/mini_val_packets.json --split mini_val --out perception_lab/outputs/sparse4d/mini_val --state-tests
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_evaluate.py --out perception_lab/outputs/sparse4d/scene0061 --split mini_scene_0061
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_evaluate.py --out perception_lab/outputs/sparse4d/mini_val --split mini_val
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_audit.py --out perception_lab/outputs/sparse4d/scene0061
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_audit.py --out perception_lab/outputs/sparse4d/mini_val
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_cases.py --out perception_lab/outputs/sparse4d/scene0061
perception_lab/envs/sparse4d/bin/python -m pytest perception_lab/tests/test_sparse4d_4090.py -q
perception_lab/envs/sparse4d/bin/python perception_lab/tools/sparse4d_summarize.py
```
All final inference/evaluation commands exited0. Build/run/evaluation logs are in this directory. Initial fixes: Python3.11/3.10 ABI, Torch SM89 whitelist, relative packet path after chdir, yapf compatibility; no network/model layer was skipped. Initial state numerical criterion1e-5 failed as recorded; model atomic forward repeatability has small floating point variation.
