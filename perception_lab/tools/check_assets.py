import argparse,json
from pathlib import Path
from evidence import sha256
p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((root/'checkpoints/manifest.json').read_text());v=manifest[a.module]
if v.get('deserialization')!='passed':raise ValueError('Checkpoint deserialization has not passed')
if sha256(v['path'])!=v['sha256']:raise ValueError('Checkpoint changed or incomplete')
(out/'asset_verification.json').write_text(json.dumps({'status':'passed','module':a.module,'asset':v,'scope':'source-resolved checkpoint bytes and deserialization; model execution status is separate'},indent=2));print(a.module,'checkpoint hash verified')
