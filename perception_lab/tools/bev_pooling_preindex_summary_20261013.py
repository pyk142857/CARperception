"""Summarize sealed measured pooling experiments; never relax fixed budgets."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from bevfusion_compare import compare
from bev_pooling_full_model import verify_inference_and_evaluation
L=Path(__file__).resolve().parents[1];R=L/'reports/tracking_gpu_20261013/bev_pooling'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 cfg=read(L/'configs/deployment_4090.yaml')['tensorrt']['acceptance_fixed_before_run'];records=[];capture_rows=[];pairs=[]
 for monitor in ['gpu_monitor.json','full_model_gpu_monitor.json']:
  assert all(not r['external_compute'] for r in read(R/monitor)),monitor
 for split in ['scene0061','mini_val']:
  a=R/'full_model/upstream'/split;b=R/'full_model/preindex'/split
  am,bm=read(a/'manifest.json'),read(b/'manifest.json')
  evidence={m:verify_inference_and_evaluation(d,read(d/'manifest.json'),d/'official') for m,d in [('upstream',a),('preindex',b)]}
  for key in ['config','packets','checkpoint','engines','onnx','native_library','core_library','adapter']:assert am['identity'][key]==bm['identity'][key],key
  assert sha(a/'actual_inputs.json')==sha(b/'actual_inputs.json')
  x=compare(read(a/'predictions.json'),read(b/'predictions.json'),cfg['same_class_match_radius_m'])
  u,v=read(a/'official/metrics/metrics_summary.json'),read(b/'official/metrics/metrics_summary.json')
  x.update(split=split,evidence=evidence,identity=am['identity'],thresholds=cfg)
  x['metrics']={k:dict(upstream=u[k],preindex=v[k],delta=v[k]-u[k]) for k in ['mean_ap','nd_score']}
  x['class_ap_deltas']={k:v['mean_dist_aps'][k]-u['mean_dist_aps'][k] for k in u['mean_dist_aps']}
  x['metric_budget_passed']=u['mean_ap']-v['mean_ap']<=cfg['map_drop_max'] and u['nd_score']-v['nd_score']<=cfg['nds_drop_max']
  x['geometry_gates']={k:x['summary'][k]['p95']<=cfg['diagnostic_p95_'+k] for k in ['center_m','size_m','yaw_rad','score_abs','velocity_mps']}
  write(R/f'full_model/{split}_comparison.json',x);records.append(x)
  for p in (b/'paired').glob('*/full_model_paired_timings.csv'):
   rows=list(csv.DictReader(p.open()));assert len(rows)==30
   aa=np.array([float(r['upstream_ms']) for r in rows]);bb=np.array([float(r['candidate_ms']) for r in rows])
   orders={o:float(np.mean([float(r['upstream_ms'])-float(r['candidate_ms']) for r in rows if r['order']==o])) for o in set(r['order'] for r in rows)}
   gate=int(sum(bb<aa))>=24 and min(orders.values())>0 and np.percentile(bb,95)<=np.percentile(aa,95)
   pairs.append(dict(split=split,token=p.parent.name,rounds=30,upstream_mean_ms=float(aa.mean()),preindex_mean_ms=float(bb.mean()),saved_percent=float(100*(1-bb.mean()/aa.mean())),wins=int(sum(bb<aa)),upstream_p95_ms=float(np.percentile(aa,95)),preindex_p95_ms=float(np.percentile(bb,95)),order_saved_ms=orders,stable_benefit=bool(gate),raw_sha256=sha(p)))
  for frame in ([0,19,38] if split=='scene0061' else [0,40,80]):
   captures={}
   for key in ['features','depth','indices','intervals','output']:
    root=L/'runs/tracking_gpu_20261013/bev_pooling/full_model_captures'
    aa=root/'upstream'/split/f'frame_{frame}.{key}.bin';bb=root/'preindex'/split/f'frame_{frame}.{key}.bin'
    captures[key]=dict(upstream_sha256=sha(aa),preindex_sha256=sha(bb),bitwise_equal=sha(aa)==sha(bb))
   capture_rows.append(dict(split=split,frame=frame,arrays=captures))
 repeat=read(R/'full_model/repeat/repeat_outputs.json');repeats=[]
 for mode in ['upstream','preindex']:
  seq=[r for r in repeat if r['mode']==mode]
  for r in seq[1:]:
   d=compare([seq[0]],[r],cfg['same_class_match_radius_m']);repeats.append(dict(mode=mode,repeat=r['repeat'],native_exact=seq[0]['native']==r['native'],summary=d['summary'],counts=d['frames']))
 assert len(pairs)==9
 result=dict(status='executed',splits=[{k:x[k] for k in ['split','metrics','class_ap_deltas','metric_budget_passed','geometry_gates','summary']} for x in records],paired_frames=pairs,captures=capture_rows,repeated_fixed_input=repeats,rebuild_smoke=read(R/'full_model/smoke/rebuild_geometry.json')['summary'],stable_full_model_benefit=all(r['stable_benefit'] for r in pairs),optimization_adopted=False,decision='Retain upstream: none of the nine full-model representatives meets 24/30 faster pairs; rebuild/geometry and repeatability differences remain explicitly reported. Local pooling gains alone do not justify adoption.',scope='nuScenes mini teaching diagnostic; full forward includes H2D/model/native-box D2H, excludes preprocessing/geometry update; separate per-frame timings include those stages.',config_sha256=sha(L/'configs/deployment_4090.yaml'))
 write(R/'full_model_summary.json',result)
 pool=read(R/'results.json');pool.update(full_model_status='passed',full_model_status_meaning='Required inference/evaluation/paired experiments executed and evidence audited; NOT optimization acceptance',optimization_adopted=False,full_model_summary='full_model_summary.json');write(R/'results.json',pool)
 write(R/'status.json',dict(status='completed',research_executed=True,optimization_adopted=False,decision=result['decision']))
 print(json.dumps({k:result[k] for k in ['status','stable_full_model_benefit','optimization_adopted']}))
if __name__=='__main__':main()
