#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Reuse the isolated MapTR Torch 1.13 environment and its compiled legacy SpConv.
# If absent, first run: bash tools/setup_maptr.sh
[ -x envs/maptr/bin/python ] || { echo 'Run tools/setup_maptr.sh first'; exit 1; }
[ -d third_party/Cylinder3D/.git ] || git clone https://github.com/xinge008/Cylinder3D.git third_party/Cylinder3D
[ "$(git -C third_party/Cylinder3D rev-parse HEAD)" = 30a0abb2ca4c657a821a5e9a343934b0789b2365 ] || { echo 'Use Cylinder3D commit 30a0abb2ca4c657a821a5e9a343934b0789b2365'; exit 1; }
[ -x envs/lidarseg_legacy/bin/python ] || envs/maptr/bin/python -m venv envs/lidarseg_legacy
envs/maptr/bin/python - <<'PY'
from pathlib import Path
p=Path('envs/lidarseg_legacy/lib/python3.10/site-packages/car_maptr.pth')
p.write_text(str(Path('envs/maptr/lib/python3.10/site-packages').resolve())+'\n')
PY
envs/lidarseg_legacy/bin/pip install --no-deps 'https://data.pyg.org/whl/torch-1.13.0%2Bcu117/torch_scatter-2.1.1%2Bpt113cu117-cp310-cp310-linux_x86_64.whl'
envs/lidarseg_legacy/bin/python tools/prepare_lidarseg_data.py
envs/lidarseg_legacy/bin/python tools/prepare_cylinder3d.py
envs/lidarseg_legacy/bin/pip freeze > envs/lidarseg_legacy_freeze.txt
