"""Export measured standardized ego gravity-centre boxes to official nuScenes JSON."""
import argparse,json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from evidence import sha256
NAMES={'car','truck','bus','trailer','construction_vehicle','pedestrian','motorcycle','bicycle','traffic_cone','barrier'}


def convert_box(box,token,pose):
    def array(key,shape):
        x=np.asarray(box.get(key),dtype=float)
        if x.shape!=shape or not np.isfinite(x).all():raise ValueError('Missing/invalid '+key+' for '+token)
        return x
    center=array('center_xyz',(3,));size=array('size_wlh',(3,));rot=array('rotation_wxyz',(4,));velocity=array('velocity_xy',(2,))
    score=float(box['score'])
    if (size<=0).any() or not np.isclose(np.linalg.norm(rot),1,atol=1e-5) or not np.isfinite(score) or not 0<=score<=1 or box['class_name'] not in NAMES:raise ValueError('Invalid box for '+token)
    q=Quaternion(pose['rotation']);r=q.rotation_matrix
    result=dict(sample_token=token,translation=(r@center+pose['translation']).tolist(),size=size.tolist(),rotation=(q*Quaternion(rot)).elements.tolist(),
                velocity=(r@np.r_[velocity,0])[:2].tolist(),detection_name=box['class_name'],detection_score=score,attribute_name='')
    if not np.isfinite(result['translation']+result['rotation']+result['velocity']).all():raise ValueError('Nonfinite pose')
    return result


def export_records(records,poses,tokens):
    actual=[r['sample_token'] for r in records]
    if len(actual)!=len(set(actual)) or set(actual)!=set(tokens) or len(tokens)!=len(set(tokens)):raise ValueError('Exact unique sample coverage required')
    result=dict(meta=dict(use_camera=False,use_lidar=True,use_radar=False,use_map=False,use_external=False),results={})
    audit=dict(input_frames=len(records),input_boxes=0,output_boxes=0,removed_top500=0,attributes='empty: model does not predict attributes',additional_score_filter=None)
    errors=[];scores=[]
    for rec in records:
        if rec.get('source')!='measured':raise ValueError('Measured predictions required')
        token=rec['sample_token'];pose=poses[token];boxes=rec['boxes3d'];audit['input_boxes']+=len(boxes)
        converted=[convert_box(b,token,pose) for b in boxes]
        if len(converted)>500:
            audit['removed_top500']+=len(converted)-500
            converted=sorted(converted,key=lambda b:-b['detection_score'])[:500]
        result['results'][token]=converted;audit['output_boxes']+=len(converted);scores.extend(b['detection_score'] for b in converted)
        if boxes:
            b=boxes[0];out=convert_box(b,token,pose);q=Quaternion(pose['rotation']);r=q.rotation_matrix
            full_velocity=r@np.r_[b['velocity_xy'],0]
            errors.append(dict(sample_token=token,center=float(np.max(np.abs(r.T@(np.array(out['translation'])-pose['translation'])-b['center_xyz']))),
                               velocity=float(np.max(np.abs(r.T@full_velocity-np.r_[b['velocity_xy'],0]))),
                               rotation=float(np.max(np.abs((q.inverse*Quaternion(out['rotation'])).rotation_matrix-Quaternion(b['rotation_wxyz']).rotation_matrix)))))
    audit['roundtrip_checks']=errors;audit['score_range']=[min(scores),max(scores)] if scores else None
    if any(max(e[k] for k in ['center','velocity','rotation'])>1e-8 for e in errors):raise ValueError('Roundtrip failed')
    return result,audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--dataroot',type=Path,required=True)
    p.add_argument('--version',default='v1.0-mini');p.add_argument('--eval-set',default='mini_scene_0061');p.add_argument('--scene',action='append',help='Merge custom split with these scene names')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import get_scenes_of_split,is_predefined_split
    from nuscenes.eval.common.loaders import get_samples_of_custom_split
    if a.scene:
        if is_predefined_split(a.eval_set):raise ValueError('Cannot redefine predefined split')
        file=a.dataroot/a.version/'splits.json';splits=json.loads(file.read_text()) if file.exists() else {}
        if a.eval_set in splits and splits[a.eval_set]!=a.scene:raise ValueError('Conflicting split definition')
        splits[a.eval_set]=a.scene;file.write_text(json.dumps(splits,indent=2)+'\n')
    nusc=NuScenes(version=a.version,dataroot=str(a.dataroot),verbose=False)
    if is_predefined_split(a.eval_set):
        scenes=set(get_scenes_of_split(a.eval_set,nusc))
        tokens=[s['token'] for s in nusc.sample if nusc.get('scene',s['scene_token'])['name'] in scenes]
    else:
        tokens=get_samples_of_custom_split(a.eval_set,nusc)
    poses={}
    for t in tokens:
        sd=nusc.get('sample_data',nusc.get('sample',t)['data']['LIDAR_TOP']);poses[t]=nusc.get('ego_pose',sd['ego_pose_token'])
    records=json.loads(a.predictions.read_text());result,audit=export_records(records,poses,tokens)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,allow_nan=False))
    audit.update(version=a.version,eval_set=a.eval_set,sample_tokens=tokens,input=str(a.predictions.resolve()),output=str(a.out.resolve()),
                 input_sha256=sha256(a.predictions),output_sha256=sha256(a.out),checkpoint_sha256=sorted({r.get('checkpoint_sha256','') for r in records}),
                 source_config_hash=sorted({r.get('source_config_hash','') for r in records}))
    a.audit.parent.mkdir(parents=True,exist_ok=True);a.audit.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ['roundtrip_checks','sample_tokens']},indent=2))
if __name__=='__main__':main()
