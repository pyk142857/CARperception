"""Conservative milestone gates. Checks evidence, not the scientific truth of every output."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence import reusable


def metric_errors(metric):
    errors=[]
    if metric.get('source')!='measured':errors.append('Metric source is not measured')
    value=metric.get('value')
    if not isinstance(value,(int,float)) or not math.isfinite(value):errors.append('Metric is missing or nonfinite')
    if metric.get('scope')!='benchmark':errors.append('Metric is not full benchmark')
    for field in ['protocol','split','sample_count','run_id','unit','direction','raw_evaluator_path']:
        if not metric.get(field):errors.append('Missing metric field '+field)
    raw=metric.get('raw_evaluator_path')
    if not raw or not Path(raw).is_file():errors.append('Missing raw evaluator evidence')
    return errors


def coverage_errors(predictions,expected):
    errors=[]
    tokens=set()
    for p in predictions:
        if p.get('source')!='measured' or p.get('scope')!='replay':
            errors.append('Non-measured or non-replay prediction')
        if p.get('status')!='passed':errors.append('Prediction failed or status missing')
        if p.get('ground_truth') or p.get('source_role')=='GT':errors.append('GT cannot count as prediction')
        tokens.add(p.get('sample_token'))
    if tokens!=set(expected):errors.append('Replay token coverage mismatch')
    return errors


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--milestone',choices=['A','B','C'],required=True);args=p.parse_args()
    plan=json.loads(Path(args.plan).read_text());root=Path(plan['root'])
    status_file=root/'results/status.json'
    status=json.loads(status_file.read_text()) if status_file.exists() else {'modules':{}}
    required=[('M00','smoke'),('M00','replay'),('M14','unified_replay')]
    required += [(f'M{i:02d}',s) for i in range(1,11) for s in ['replay','visualization']]
    required += [('M11','replay_M05'),('M11','replay_M07')]
    if args.milestone in ['B','C']:
        required += [(f'M{i:02d}',s) for i in range(1,11) for s in ['evaluation','benchmark']]
        required += [('M11','evaluation_M05'),('M11','evaluation_M07'),('M11','benchmark')]
        required += [('M12',f'{s}_M{i:02d}') for i in [1,2,3] for s in ['onnx_gpu','tensorrt_fp16','export_validation']]
    if args.milestone=='C':required += [('M13','train_A'),('M13','train_B'),('M13','evaluation'),('M14','report')]
    errors=[]
    for mid,stage in required:
        r=status['modules'].get(mid,{}).get(stage,{})
        if not reusable(r,r.get('fingerprint')):errors.append(f'{mid}/{stage}: no complete valid passed evidence')
    manifest=root/'manifests/replay_val.json'
    if not manifest.is_file():errors.append('Frozen official-val two-scene replay manifest missing')
    else:
        replay=json.loads(manifest.read_text())
        if replay.get('split')!='val' or len(replay.get('scene_tokens',[]))!=2 or not replay.get('sample_tokens'):
            errors.append('Replay scope is not two complete official val scenes')
        for mid in [f'M{i:02d}' for i in range(1,11)]:
            record=status['modules'].get(mid,{}).get('replay',{})
            files=[Path(f) for f in record.get('artifacts',{}) if Path(f).name=='predictions.json']
            if not files:errors.append(mid+': missing replay predictions')
            else:errors += [mid+': '+e for e in coverage_errors(json.loads(files[0].read_text()),replay['sample_tokens'])]
    if args.milestone in ['B','C']:
        for mid in [f'M{i:02d}' for i in range(1,11)]:
            record=status['modules'].get(mid,{}).get('evaluation',{})
            files=[Path(f) for f in record.get('artifacts',{}) if Path(f).name=='metrics.json']
            if not files:errors.append(mid+': missing task metrics')
            for file in files:
                for metric in json.loads(file.read_text()):errors += [mid+': '+e for e in metric_errors(metric)]
    result={'milestone':args.milestone,'status':'failed' if errors else 'passed','errors':errors,
            'limitations':'Structural evidence checks do not by themselves prove scientific authenticity.'}
    (root/'results'/f'verification_{args.milestone}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));return bool(errors)


if __name__=='__main__':sys.exit(main())
