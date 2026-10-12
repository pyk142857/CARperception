#!/usr/bin/env bash
set -euo pipefail
LAB=$(cd "$(dirname "$0")/.." && pwd)
SRC="$LAB/third_party/Sparse4D"
PY="$LAB/envs/sparse4d/bin/python"
mkdir -p "$LAB/checkpoints/sparse4d" "$LAB/outputs/sparse4d"
if [ ! -d "$SRC/.git" ]; then git clone https://github.com/HorizonRobotics/Sparse4D.git "$SRC"; fi
git -C "$SRC" checkout 249ffbb695f4e9db628d953e2bf6d36de04bbb69
if [ ! -x "$PY" ]; then "$LAB/envs/maptr/bin/python" -m venv "$LAB/envs/sparse4d"; fi
"$PY" - "$LAB" <<'PY'
import pathlib,site,sys
lab=pathlib.Path(sys.argv[1]); base=lab/'envs/maptr/lib/python3.10/site-packages'
assert base.is_dir()
pathlib.Path(site.getsitepackages()[0],'maptr_readonly.pth').write_text(str(base)+'\n')
p=lab/'third_party/Sparse4D/projects/mmdet3d_plugin/ops/setup.py'
s=p.read_text()
if 'compute_89' not in s:
    s=s.replace('"-D__CUDA_NO_HALF_OPERATORS__",','"-gencode=arch=compute_89,code=sm_89",\n            "-D__CUDA_NO_HALF_OPERATORS__",')
    p.write_text(s)
PY
"$PY" -m pip install --no-deps motmetrics==1.4.0 nuscenes-devkit==1.2.0 xmltodict==0.14.2 yapf==0.33.0
cd "$SRC/projects/mmdet3d_plugin/ops"
CUDA_HOME=/usr/local/cuda-11.8 CXX=/usr/bin/g++-9 CC=/usr/bin/gcc-9 MAX_JOBS=2 "$PY" setup.py build_ext --inplace
if [ ! -f "$LAB/checkpoints/sparse4d/sparse4dv3_r50.pth" ]; then
    curl -L --fail --retry 2 -o "$LAB/checkpoints/sparse4d/sparse4dv3_r50.pth.partial" https://github.com/HorizonRobotics/Sparse4D/releases/download/v3.0/sparse4dv3_r50.pth
    mv "$LAB/checkpoints/sparse4d/sparse4dv3_r50.pth.partial" "$LAB/checkpoints/sparse4d/sparse4dv3_r50.pth"
fi
cd "$SRC"
"$PY" tools/nuscenes_converter.py --root_path "$LAB/data/nuscenes" --version v1.0-mini --info_prefix "$LAB/outputs/sparse4d/nuscenes-mini"
