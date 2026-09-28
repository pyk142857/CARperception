#!/usr/bin/env bash
set -euo pipefail
# Run from perception_lab; Python 3.10 and a CUDA 11.8 compiler are required.
cd "$(dirname "$0")/.."
: "${MAPTR_PYTHON:=python3.10}"
: "${CUDA_HOME:=/usr/local/cuda-11.8}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
export MAX_JOBS=2
[ -x envs/maptr/bin/python ] || "$MAPTR_PYTHON" -m venv envs/maptr
envs/maptr/bin/pip install --index-url https://pypi.org/simple 'setuptools<70' wheel
envs/maptr/bin/pip install --index-url https://pypi.org/simple -r configs/maptr_requirements.txt -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
# Upstream's old upper bound predates MMCV 1.7 (needed for a Python 3.10 wheel).
# All actual CUDA ops and strict model weights are verified by the mini run.
envs/maptr/bin/python - <<'PY'
from pathlib import Path
p=Path('third_party/MapTR/mmdetection3d/mmdet3d/__init__.py')
s=p.read_text().replace("mmcv_maximum_version = '1.4.0'", "mmcv_maximum_version = '1.7.1'")
p.write_text(s)
# Prefer this checkout's sparse layers over MMCV 1.7's new same-name registry entries.
p=Path('third_party/MapTR/mmdetection3d/mmdet3d/ops/spconv/conv.py')
p.write_text(p.read_text().replace('@CONV_LAYERS.register_module()', '@CONV_LAYERS.register_module(force=True)'))
# MMDetection 2.28 also includes EfficientNet; keep the plugin implementation.
p=Path('third_party/MapTR/projects/mmdet3d_plugin/models/backbones/efficientnet.py')
p.write_text(p.read_text().replace('@BACKBONES.register_module()', '@BACKBONES.register_module(force=True)'))
# Numba moved this warning class under core.errors.
p=Path('third_party/MapTR/mmdetection3d/mmdet3d/datasets/pipelines/data_augment_utils.py')
p.write_text(p.read_text().replace('from numba.errors import', 'from numba.core.errors import'))
# Removed THC declarations are unused; current stream is already ATen CUDA.
for p in Path('third_party/MapTR/mmdetection3d/mmdet3d/ops').rglob('*.cpp'):
    s=p.read_text()
    fixed=s.replace('#include <THC/THC.h>', '#include <ATen/cuda/CUDAContext.h>').replace('extern THCState *state;', '')
    if fixed != s:p.write_text(fixed)
PY
MAPTR_ENV="$(pwd)/envs/maptr"
(cd third_party/MapTR/mmdetection3d && "$MAPTR_ENV/bin/python" setup.py build_ext --inplace)
(cd third_party/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn && "$MAPTR_ENV/bin/python" setup.py build_ext --inplace && "$MAPTR_ENV/bin/pip" install --no-build-isolation --no-deps .)
envs/maptr/bin/pip freeze > envs/maptr_freeze.txt
