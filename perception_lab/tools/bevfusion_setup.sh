#!/usr/bin/env bash
# Reproduce the isolated tested environment without modifying shared environments.
set -euo pipefail
LAB="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$LAB/third_party/bevfusion_4090"
BASE_PYTHON="${BEVFUSION_BASE_PYTHON:-/home/minglei/anaconda3/envs/autolabel/bin/python}"
if [ ! -d "$SRC/.git" ]; then
  git clone https://github.com/mit-han-lab/bevfusion.git "$SRC"
  git -C "$SRC" checkout db75150717a9462cb60241e36ba28d65f6908607
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = db75150717a9462cb60241e36ba28d65f6908607 ]
if [ ! -x "$LAB/envs/bevfusion/bin/python" ]; then
  "$BASE_PYTHON" -m venv "$LAB/envs/bevfusion"
fi
printf '%s\n' "$LAB/envs/maptr/lib/python3.10/site-packages" > "$LAB/envs/bevfusion/lib/python3.10/site-packages/legacy_torch.pth"
"$LAB/envs/bevfusion/bin/pip" install torchpack==0.3.1 mpi4py
if git -C "$SRC" apply --check "$LAB/tools/bevfusion_torch113_sm89.patch"; then
  git -C "$SRC" apply "$LAB/tools/bevfusion_torch113_sm89.patch"
else
  git -C "$SRC" apply --reverse --check "$LAB/tools/bevfusion_torch113_sm89.patch"
fi
CFG='configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50'
mkdir -p "$SRC/$CFG"
cp "$LAB/third_party/Lidar_AI_Solution/CUDA-BEVFusion/$CFG/"*.yaml "$SRC/$CFG/"
cd "$SRC"
CUDA_HOME=/usr/local/cuda-11.8 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=8.9 "$LAB/envs/bevfusion/bin/python" setup.py build_ext --inplace
