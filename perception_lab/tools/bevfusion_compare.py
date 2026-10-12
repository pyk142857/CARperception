"""Paired measured detection comparison; thresholds come from frozen config."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
from bevfusion_4090 import digest, write_json


def compare(reference, candidate, radius):
    if [r['sample_token'] for r in reference] != [r['sample_token'] for r in candidate]:
        raise ValueError('Paired sample order differs')
    matched, counts = [], []
    for ref, cand in zip(reference, candidate):
        a,b=ref['boxes3d'],cand['boxes3d']
        accepted=[]
        for cls in sorted(set(x['class_name'] for x in a+b)):
            ai=[i for i,x in enumerate(a) if x['class_name']==cls]
            bi=[i for i,x in enumerate(b) if x['class_name']==cls]
            if not ai or not bi: continue
            distance=np.linalg.norm(np.array([a[i]['center_xyz'] for i in ai])[:,None]-np.array([b[i]['center_xyz'] for i in bi])[None,:],axis=2)
            rows,cols=linear_sum_assignment(np.where(distance<=radius,distance,1e6))
            accepted += [(ai[i],bi[j]) for i,j in zip(rows,cols) if distance[i,j]<=radius]
        for i,j in accepted:
            x,y=a[i],b[j]
            rotation=Quaternion(x['rotation_wxyz']).inverse*Quaternion(y['rotation_wxyz'])
            matched.append(dict(sample_token=ref['sample_token'],class_name=x['class_name'],reference_score=x['score'],candidate_score=y['score'],
                center_m=float(np.linalg.norm(np.array(x['center_xyz'])-y['center_xyz'])),
                size_m=float(np.max(np.abs(np.array(x['size_wlh'])-y['size_wlh']))),
                yaw_rad=float(abs(rotation.radians)),score_abs=abs(x['score']-y['score']),
                velocity_mps=float(np.linalg.norm(np.array(x['velocity_xy'])-y['velocity_xy']))))
        counts.append(dict(sample_token=ref['sample_token'],reference=len(a),candidate=len(b),matched=len(accepted),unmatched_reference=len(a)-len(accepted),unmatched_candidate=len(b)-len(accepted)))
    summary={}
    for key in ['center_m','size_m','yaw_rad','score_abs','velocity_mps']:
        v=[r[key] for r in matched]
        summary[key] = dict(mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),max=float(np.max(v))) if v else None
    return dict(matching='class constrained Hungarian 3D centre distance; unmatched reported separately',radius_m=radius,frames=counts,matched=matched,summary=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['reference','candidate','reference-metrics','candidate-metrics','config','out']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())['tensorrt']['acceptance_fixed_before_run']
    result=compare(json.loads(a.reference.read_text()),json.loads(a.candidate.read_text()),cfg['same_class_match_radius_m'])
    rm,cm=json.loads(a.reference_metrics.read_text()),json.loads(a.candidate_metrics.read_text())
    result.update(source='measured',thresholds=cfg,input_hashes={k:digest(getattr(a,k)) for k in ['reference','candidate','reference_metrics','candidate_metrics','config']})
    result['metrics']={k:dict(reference=rm[k],candidate=cm[k],delta=cm[k]-rm[k],drop=rm[k]-cm[k]) for k in ['mean_ap','nd_score']}
    result['class_ap_deltas']={c:cm['mean_dist_aps'][c]-rm['mean_dist_aps'][c] for c in rm['mean_dist_aps']}
    result['metric_budget_passed'] = rm['mean_ap']-cm['mean_ap']<=cfg['map_drop_max'] and rm['nd_score']-cm['nd_score']<=cfg['nds_drop_max']
    result['geometry_diagnostics']={k:bool(result['summary'][k] is not None and result['summary'][k]['p95']<=cfg['diagnostic_p95_'+k]) for k in ['center_m','size_m','yaw_rad','score_abs','velocity_mps']}
    result['status']='passed' if result['metric_budget_passed'] and all(result['geometry_diagnostics'].values()) else 'failed'
    write_json(a.out,result)
    print(json.dumps({k:result[k] for k in ['status','metrics','summary','geometry_diagnostics']},indent=2))

if __name__=='__main__':main()
