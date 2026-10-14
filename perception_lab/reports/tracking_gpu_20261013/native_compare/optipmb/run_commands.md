# Reproduction from repository root

Pinned source (third-party checkout is excluded from Git):

```bash
git clone https://github.com/dinggh0817/OptiPMB perception_lab/third_party/OptiPMB
git -C perception_lab/third_party/OptiPMB checkout ed0e0467efb08fd8a96b03647b09169564a74acd
```

An isolated Python 3.10 venv, `perception_lab/envs/optipmb_20261013`, reuses existing maptr and bevfusion site-packages read-only through `reuse.pth`. Its only installed additions are powerboxes 0.2.3, cholupdates 0.0.1a7 and numpyencoder 0.3.2. Exact installed versions are in `environment_freeze.txt`; actual imported versions are in `source_manifest.json`. Existing environments were not upgraded. With the project environment available:

```bash
perception_lab/envs/grae3dmot_2025/bin/python -m venv perception_lab/envs/optipmb_20261013
python - <<'PY'
from pathlib import Path
p=Path('perception_lab/envs/optipmb_20261013/lib/python3.10/site-packages/reuse.pth')
p.write_text('\n'.join(str((Path('perception_lab/envs')/x/'lib/python3.10/site-packages').resolve()) for x in ['maptr','bevfusion'])+'\n')
PY
perception_lab/envs/optipmb_20261013/bin/pip install --no-deps powerboxes==0.2.3 cholupdates==0.0.1a7 numpyencoder==0.3.2
```

Fresh predictions and separate replay evidence:

```bash
perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/optipmb_tracking_20261013.py --split scene0061 --validate
perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/optipmb_tracking_20261013.py --split mini_val --validate
perception_lab/envs/optipmb_20261013/bin/python perception_lab/tools/optipmb_tracking_20261013.py --diagnostic
perception_lab/envs/optipmb_20261013/bin/python -m unittest discover -s perception_lab/tests -p test_optipmb_tracking_20261013.py
```

The adapter sets all CPU numerical thread limits to four and disables visible CUDA devices before importing numerical modules. Real mini point clouds must already exist at `perception_lab/data/nuscenes/`; no point clouds are included in reports. Diagnostic mode expects previous noninstrumented `tracking.json` and verifies exact canonical equality.
