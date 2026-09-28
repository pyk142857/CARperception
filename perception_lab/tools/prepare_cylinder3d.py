"""Prepare an isolated adapter using original SpConv 1 semantics, not SpConv 2."""
from pathlib import Path
import shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
repo=ROOT/'third_party/Cylinder3D'
adapter=ROOT/'third_party/Cylinder3D_legacy'
for name in ['network','builder']:
    (adapter/name).mkdir(parents=True,exist_ok=True)
    for source in (repo/name).glob('*.py'):
        shutil.copy2(source,adapter/name/source.name)
# Get pristine source even if an earlier compatibility experiment edited it.
s=subprocess.check_output(['git','-C',str(repo),'show','HEAD:network/segmentator_3d_asymm_spconv.py'],text=True)
s=s.replace('import spconv\n','from mmdet3d.ops import spconv\n')
(adapter/'network/segmentator_3d_asymm_spconv.py').write_text(s)
print('Original SpConv 1 kernels and index sharing preserved')
