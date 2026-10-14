"""Official mini TrackingEval for independently generated frozen-input trackers."""
import argparse,json,hashlib,sys
from pathlib import Path
from tracking_gpu_inputs_20261013 import ROOT,REPORT,SPECS,paths,sha

def evaluate(model):
 from nuscenes.eval.tracking.evaluate import TrackingEval
 from nuscenes.eval.common.config import config_factory
 from importlib.metadata import version
 folder=REPORT/'native_compare'/model
 if model=='grae':folder=folder/'frozen_reference'
 if model=='grae_gpu':folder=REPORT/'grae_gpu'/'frozen_reference'
 rows=[];cfg=config_factory('tracking_nips_2019')
 splitfile=ROOT/'data/nuscenes/v1.0-mini/splits.json'
 custom=json.loads(splitfile.read_text())
 if custom.get('mini_scene_0061')!=['scene-0061']:raise ValueError('Custom scene split mismatch')
 state_path=folder/'evaluation_status.json';state_path.parent.mkdir(parents=True,exist_ok=True)
 state_path.write_text(json.dumps({'status':'running'})+'\n')
 try:
  for split,(count,_,digest,pdigest) in SPECS.items():
   det,packet=paths(split)
   if sha(det)!=digest or sha(packet)!=pdigest:raise ValueError('Frozen inputs altered')
   packets=json.loads(packet.read_text());tokens=[p['sample_token'] for p in packets];pred=folder/split/'tracking.json';data=json.loads(pred.read_text())
   if set(data['results'])!=set(tokens) or len(tokens)!=count:raise ValueError('Tracking coverage mismatch')
   out=folder/split/'official_eval';out.mkdir(parents=True,exist_ok=True)
   protocol=dict(custom_split_sha256=sha(splitfile),custom_scene0061=['scene-0061'],model=model,version='v1.0-mini',devkit=version('nuscenes-devkit'),prediction_sha256=sha(pred),frozen_detection_sha256=digest,packets_sha256=pdigest,tokens=tokens,config=cfg.serialize(),gt_sha256={name:sha(ROOT/'data/nuscenes/v1.0-mini'/name) for name in ['sample_annotation.json','sample.json','scene.json']},command=sys.argv,scope='Official tracking_nips_2019 on mini only; no display score filtering')
   (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
   TrackingEval(cfg,str(pred),'mini_scene_0061' if split=='scene0061' else 'mini_val',str(out),'v1.0-mini',str(ROOT/'data/nuscenes'),verbose=True).main(render_curves=False)
   metrics=out/'metrics_summary.json';values=json.loads(metrics.read_text());rows.append(dict(model=model,split=split,frames=count,tracking_sha256=sha(pred),metrics_sha256=sha(metrics),**{k:values[k] for k in ['amota','amotp','ids','fp','fn','recall','mota']}))
  (folder/'metrics.json').write_text(json.dumps(rows,indent=2)+'\n');state_path.write_text(json.dumps({'status':'passed','rows':rows},indent=2)+'\n');print(json.dumps(rows,indent=2));return rows
 except Exception as exc:
  state_path.write_text(json.dumps({'status':'failed','error':str(exc)},indent=2)+'\n');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['grae','optipmb','grae_gpu'],required=True);evaluate(p.parse_args().model)
