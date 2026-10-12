#!/usr/bin/env bash
# Install only into the GaussianWorld venv; base environments remain read-only.
set -euo pipefail
LAB=$(cd "$(dirname "$0")/.." && pwd)
BASE_PYTHON=${GAUSSIANWORLD_BASE_PYTHON:-/home/minglei/anaconda3/envs/autolabel/bin/python}
BASE_SITE=$($BASE_PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
VENV="$LAB/envs/gaussianworld"
REPO="$LAB/third_party/GaussianWorld"
if [[ ! -x "$VENV/bin/python" ]]; then "$BASE_PYTHON" -m venv "$VENV"; fi
SITE=$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cat > "$SITE/base.pth" <<EOF
$LAB/envs/mmdet3d/lib/python3.10/site-packages
$BASE_SITE
$LAB/third_party/mmdetection3d
EOF
"$VENV/bin/pip" install mmsegmentation==1.2.2 timm ninja
if [[ ! -d "$REPO/.git" ]]; then git clone https://github.com/zuosc19/GaussianWorld.git "$REPO"; fi
if [[ $(git -C "$REPO" rev-parse HEAD) != b43629eaecffd5a7cbaac1a55517766e6263e4fc ]]; then
  echo 'Source revision differs; inspect local changes before checking out b43629eaecffd5a7cbaac1a55517766e6263e4fc.' >&2
  exit 1
fi
export CUDA_HOME=${GAUSSIANWORLD_CUDA_HOME:-/usr/local/cuda-11.8}
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4
for OP in model/encoder/gaussian_encoder/ops model/head/localagg; do
  (cd "$REPO/$OP" && "$VENV/bin/python" setup.py build_ext --inplace)
done
mkdir -p "$LAB/checkpoints/gaussianworld"
"$VENV/bin/python" - "$LAB/checkpoints/gaussianworld/ckpt_stream.pth" <<'PY'
import hashlib,re,sys,urllib.request
from pathlib import Path
p=Path(sys.argv[1]);expected='5477081191e6d7fb8ac880bf5b99230b944a7cea76b0d87175c789a51cb97be3'
if not p.exists():
    page=urllib.request.urlopen('https://cloud.tsinghua.edu.cn/f/4939dcc50b2a44c1b98d/',timeout=60).read().decode()
    url=re.search("rawPath: '([^']+)'",page).group(1).replace('\\u002D','-')
    partial=p.with_suffix('.partial');urllib.request.urlretrieve(url,partial);partial.rename(p)
actual=hashlib.sha256(p.read_bytes()).hexdigest()
if actual!=expected:raise SystemExit(f'Checkpoint mismatch: {actual}')
print('Checkpoint SHA256 verified:',actual)
PY
