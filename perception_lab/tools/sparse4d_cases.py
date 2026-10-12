"""Fixed-threshold teaching diagnosis; independent from official AP and IDS."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.utils import category_to_detection_name

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'outputs/sparse4d/scene0061');a=p.parse_args()
lab=Path(__file__).resolve().parents[1]
nusc=NuScenes(version='v1.0-mini',dataroot=str(lab/'data/nuscenes'),verbose=False)
records=json.loads((a.out/'predictions.json').read_text());cases=[];counts=[]
for ix,r in enumerate(records):
    sample=nusc.get('sample',r['sample_token']);sd=nusc.get('sample_data',sample['data']['LIDAR_TOP']);pose=nusc.get('ego_pose',sd['ego_pose_token']);rot=Quaternion(pose['rotation']).rotation_matrix
    gt=[]
    for token in sample['anns']:
        ann=nusc.get('sample_annotation',token);cls=category_to_detection_name(ann['category_name'])
        xyz=rot.T@(np.array(ann['translation'])-pose['translation']);distance=float(np.linalg.norm(xyz[:2]))
        ranges={'car':50,'truck':50,'bus':50,'trailer':50,'construction_vehicle':50,'pedestrian':40,'motorcycle':40,'bicycle':40,'barrier':30,'traffic_cone':30}
        if cls not in ranges or distance>ranges[cls] or ann['num_lidar_pts']+ann['num_radar_pts']==0:continue
        gt.append(dict(annotation_token=token,class_name=cls,center_xyz=xyz.tolist(),distance_m=distance))
    pred=[b for b in r['boxes3d'] if b['score']>=.25];matched_gt=set();matched_pred=set()
    for cls in sorted({x['class_name'] for x in gt+pred}):
        gi=[i for i,g in enumerate(gt) if g['class_name']==cls];pi=[i for i,b in enumerate(pred) if b['class_name']==cls]
        if not gi or not pi:continue
        d=np.linalg.norm(np.array([gt[i]['center_xyz'][:2] for i in gi])[:,None]-np.array([pred[i]['center_xyz'][:2] for i in pi])[None],axis=-1)
        rows,cols=linear_sum_assignment(np.where(d<=2,d,1e6))
        for row,col in zip(rows,cols):
            if d[row,col]<=2:matched_gt.add(gi[row]);matched_pred.add(pi[col])
    counts.append(dict(frame_index=ix,sample_token=r['sample_token'],tp=len(matched_gt),fn=len(gt)-len(matched_gt),fp=len(pred)-len(matched_pred)))
    for i,g in enumerate(gt):
        if i in matched_gt:continue
        near=[b for b in r['boxes3d'] if b['class_name']==g['class_name']]
        nearest=min(near,key=lambda b:np.linalg.norm(np.array(b['center_xyz'][:2])-g['center_xyz'][:2])) if near else None
        dist=float(np.linalg.norm(np.array(nearest['center_xyz'][:2])-g['center_xyz'][:2])) if nearest else None
        cases.append(dict(frame_index=ix,sample_token=r['sample_token'],kind='FN',**g,nearest_model_score=nearest['score'] if nearest else None,nearest_same_class_center_distance_m=dist,nearest_prediction_matched_other_gt=(pred.index(nearest) in matched_pred) if nearest in pred else False,
            observation='No same-class predicted box at score >= .25 matched within 2 m.',possible_reason='Hypothesis: low confidence, localization ambiguity or nearby-instance assignment ambiguity; cause not verified.',next_step='Inspect synchronized six-camera projections and adjacent native IDs.'))
cases.sort(key=lambda c:(c['distance_m'],-c['frame_index']))
selected=[];classes=set()
for c in cases:
    if c['class_name'] not in classes: selected.append(c);classes.add(c['class_name'])
    if len(selected)==5:break
(a.out/'failure_cases.json').write_text(json.dumps(dict(protocol='Teaching diagnostic: score >= .25, same-class Hungarian XY distance <= 2 m; GT visibility/range filtered. Not official DetectionEval or TrackingEval IDS.',cases=selected,frame_counts=counts),indent=2)+'\n')
