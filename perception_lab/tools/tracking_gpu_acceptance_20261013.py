"""Read-only acceptance evidence audit (does not replace any real experiment)."""
from pathlib import Path
import argparse,hashlib,json
R=Path(__file__).resolve().parents[1]/'reports/tracking_gpu_20261013'
def load(path):return json.loads((R/path).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(value,message):
 if not value:raise AssertionError(message)

def main():
 p=argparse.ArgumentParser();p.add_argument('--scope',choices=['tracking','all'],default='tracking');a=p.parse_args()
 correctness=load('kernel_correctness.json');boundary=load('kernel_boundary_checks.json');kb=load('kernelbench/typed_gpu_validation.json')
 require(correctness['status']=='passed' and correctness['frames']==120,'120 real frames')
 require(all(x['indices_order_types_exact'] and x['model_input_bytes_exact'] for x in correctness['rows']),'NMS/model input exact')
 require(boundary['status']=='passed' and boundary['reference_failures'][0]['status']=='reference_failure','isolated upstream failure')
 require(kb['gpu']['status']=='passed' and kb['gpu']['representative_frames']==9,'typed real KernelBench tasks')
 full=load('benchmark/full.json');micro=load('benchmark/micro.json')
 require(full['status']=='passed' and micro['status']=='passed','performance measured')
 require(len(full['rows'])==60 and len(micro['rows'])==270,'30 pairs per full split and per representative')
 require(len(micro['representatives'])==9,'three scenes first/mid/last')
 for split in ['scene0061','mini_val']:
  rows=[x for x in full['rows'] if x['split']==split]
  require(sorted(x['round'] for x in rows)==list(range(30)),'round coverage')
  require(sum(x['order']=='AB' for x in rows)==15,'balanced orders')
  require(all(x['cpu_output_sha256']==x['gpu_output_sha256'] for x in rows),'every full pair exact')
  cpu=R/'native_compare/grae/frozen_reference'/split/'tracking.json';gpu=R/'grae_gpu/frozen_reference'/split/'tracking.json'
  require(sha(cpu)==sha(gpu),'published trajectories exact')
  x=load(f'native_compare/grae/frozen_reference/{split}/official_eval/metrics_summary.json');y=load(f'grae_gpu/frozen_reference/{split}/official_eval/metrics_summary.json');x.pop('eval_time');y.pop('eval_time')
  require(json.dumps(x,sort_keys=True)==json.dumps(y,sort_keys=True),'official all metrics exact')
 require(full['candidate_adopted']==(all(x['adoption_gate'] for x in full['summary'].values()) and full['combined_summary']['adoption_gate']),'combined adoption gate')
 require(all(not x.get('external_compute') for x in load('benchmark/micro_gpu_monitor.json')),'micro GPU isolation')
 for path in (R/'benchmark').glob('monitor_*.json'):
  # Accepted run currently has no rejected trials; never silently hide rejected records.
  if not full['rejected_trials']:require(all(not x.get('external_compute') for x in json.loads(path.read_text())),'full GPU isolation')
 if a.scope=='all':
  pool=load('bev_pooling/results.json');require(pool['status'] in ['measured','correctness_failed'],'pooling measured')
  require(not pool.get('full_model_required') or pool.get('full_model_status')=='passed','pooling conditional whole model')
  for split in ['', 'mini_val/']:
   vis=load('visualization/'+split+'status.json');require(vis.get('browser_validation')=='passed' and vis['decoded_rrd_audit']['status']=='passed','visualization '+split)
 print(json.dumps({'audit':'passed','scope':a.scope,'real_frames':120,'full_pairs':60,'micro_pairs':270,'gpu_speedup_adopted':full['candidate_adopted']}))
if __name__=='__main__':main()
