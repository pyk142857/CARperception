"""Reassess sealed BEVFusion backend comparisons after native-center restoration.

No inference, thresholds, scores, rotations, dimensions or frozen inputs changed.
"""
import argparse,copy,csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from bevfusion_compare import compare
L=Path(__file__).resolve().parents[1]
R=L/'reports/center_reassessment_20261019'
SOURCES={'fp32':L/'reports/deployment_4090_20261008/bevfusion_reference',
 'trt':L/'outputs/bevfusion_tensorrt','tile4':L/'outputs/bevfusion_tensorrt_candidate',
 'rebuilt_upstream':L/'reports/tracking_gpu_20261013/bev_pooling/full_model/upstream',
 'preindex':L/'reports/tracking_gpu_20261013/bev_pooling/full_model/preindex'}
PAIRS=[('fp16','fp32','trt'),('tile4','trt','tile4'),('preindex','rebuilt_upstream','preindex')]
SPLITS=['scene0061','mini_val']
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def restore_native_center(records,packets):
    poses={p['sample_token']:np.asarray(p['sensors']['LIDAR_TOP']['T_sensor_to_ego'])[:3,2] for p in packets}
    assert [r['sample_token'] for r in records]==[p['sample_token'] for p in packets]
    corrected=copy.deepcopy(records)
    for record in corrected:
        up=poses[record['sample_token']]
        for b in record['boxes3d']:b['center_xyz']=(np.asarray(b['center_xyz'])-up*b['size_wlh'][2]/2).tolist()
    return corrected

def pairs(a,b,radius):
    out=[]
    for frame,(x,y) in enumerate(zip(a,b)):
        aa,bb=x['boxes3d'],y['boxes3d']
        for cls in sorted(set(v['class_name'] for v in aa+bb)):
            ai=[i for i,v in enumerate(aa) if v['class_name']==cls];bi=[i for i,v in enumerate(bb) if v['class_name']==cls]
            if not ai or not bi:continue
            dist=np.linalg.norm(np.asarray([aa[i]['center_xyz'] for i in ai])[:,None]-np.asarray([bb[i]['center_xyz'] for i in bi])[None,:],axis=2)
            ii,jj=linear_sum_assignment(np.where(dist<=radius,dist,1e6))
            out.extend((frame,ai[i],bi[j]) for i,j in zip(ii,jj) if dist[i,j]<=radius)
    return out

def stats(values):
    a=np.asarray(values);return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),max=float(a.max()))

def prepare():
    cfg=read(L/'configs/deployment_4090.yaml')['tensorrt']['acceptance_fixed_before_run'];receipts=[];comparisons=[]
    for split in SPLITS:
        pp=L/f'reports/deployment_4090_20261008/inputs/{split}_packets.json';packets=read(pp);original={};fixed={}
        for name,base in SOURCES.items():
            folder=base/split;manifest=read(folder/'manifest.json');assert manifest['status']=='passed'
            source=folder/'predictions.json';assert sha(source)==manifest['outputs_sha256']['predictions.json']
            original[name]=read(source);fixed[name]=restore_native_center(original[name],packets)
            out=R/'corrected'/name/split/'predictions.json';write(out,fixed[name])
            receipts.append(dict(model=name,split=split,frames=len(packets),boxes=sum(len(x['boxes3d']) for x in fixed[name]),source=str(source.relative_to(L)),source_sha256=sha(source),source_manifest_sha256=sha(folder/'manifest.json'),corrected_sha256=sha(out),packets_sha256=sha(pp),change='Only standardized ego center_xyz minus lidar-up times h/2; all candidates preserved'))
        for name,left,right in PAIRS:
            old=compare(original[left],original[right],cfg['same_class_match_radius_m']);new=compare(fixed[left],fixed[right],cfg['same_class_match_radius_m'])
            op=pairs(original[left],original[right],cfg['same_class_match_radius_m']);npairs=pairs(fixed[left],fixed[right],cfg['same_class_match_radius_m'])
            errs=[];rows=[]
            for frame,i,j in op:
                x,y=original[left][frame]['boxes3d'][i],original[right][frame]['boxes3d'][j]
                a,b=fixed[left][frame]['boxes3d'][i],fixed[right][frame]['boxes3d'][j]
                before=float(np.linalg.norm(np.asarray(x['center_xyz'])-y['center_xyz']));after=float(np.linalg.norm(np.asarray(a['center_xyz'])-b['center_xyz']))
                assert abs(after-before)<=abs(x['size_wlh'][2]-y['size_wlh'][2])/2+1e-10
                errs.append(after);rows.append(dict(frame=frame,reference_index=i,candidate_index=j,reference_score=x['score'],candidate_score=y['score'],center_before_m=before,center_after_m=after,height_delta_m=y['size_wlh'][2]-x['size_wlh'][2],retained_after_rematching=(frame,i,j) in set(npairs)))
            path=R/'comparisons'/name/split;path.mkdir(parents=True,exist_ok=True)
            with (path/'fixed_pair_errors.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            gates=lambda d:{k:d['summary'][k]['p95']<=cfg['diagnostic_p95_'+k] for k in ['center_m','size_m','yaw_rad','score_abs','velocity_mps']}
            result=dict(comparison=name,split=split,reference=left,candidate=right,thresholds=cfg,old_summary=old['summary'],corrected_summary=new['summary'],fixed_original_pairs_center=stats(errs),old_geometry_gates=gates(old),corrected_geometry_gates=gates(new),old_matched=len(op),corrected_matched=len(npairs),pair_symmetric_difference=len(set(op)^set(npairs)),old_unmatched=sum(x['unmatched_reference']+x['unmatched_candidate'] for x in old['frames']),corrected_unmatched=sum(x['unmatched_reference']+x['unmatched_candidate'] for x in new['frames']))
            for threshold in [.25,.5]:
                selected=[x for x in rows if x['reference_score']>=threshold or x['candidate_score']>=threshold]
                result[f'fixed_pairs_either_score_ge_{threshold}']={'before':stats([x['center_before_m'] for x in selected]),'after':stats([x['center_after_m'] for x in selected])}
            write(path/'summary.json',result);comparisons.append(result)
            print(name,split,'center P95',old['summary']['center_m']['p95'],'->',new['summary']['center_m']['p95'],'fixed-pair',stats(errs)['p95'],flush=True)
    write(R/'source_receipts.json',receipts);write(R/'geometry_summary.json',comparisons)

def evaluate():
    for split in SPLITS:
        for name in SOURCES:
            d=R/'corrected'/name/split
            cmd=[str(L/'envs/sparse4d/bin/python'),str(L/'tools/bevfusion_evaluate.py'),'--predictions',str(d/'predictions.json'),'--dataroot',str(L/'data/nuscenes'),'--out',str(d/'official'),'--eval-set','mini_scene_0061' if split=='scene0061' else 'mini_val']
            with (d/'evaluate.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
            print(name,split,'official evaluation done',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','evaluate']);a=p.parse_args()
    (prepare if a.phase=='prepare' else evaluate)()
