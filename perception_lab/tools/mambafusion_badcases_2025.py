"""Fixed-threshold teaching diagnostics after official global-box evaluation."""
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from mambafusion_evaluate_2025 import sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.common.loaders import load_gt_of_sample_tokens,load_prediction,add_center_dist,filter_eval_boxes
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=ROOT/'reports/perception_2025_20261013/mambafusion'
    state=json.loads((report/'status.json').read_text())
    if state.get('official_evaluation')!='passed':raise ValueError('Current official evaluation required')
    nusc=NuScenes(version='v1.0-mini',dataroot=str(ROOT/'data/nuscenes'),verbose=False)
    config=config_factory('detection_cvpr_2019')
    for split,eval_set in [('scene0061','mini_scene_0061'),('mini_val','mini_val')]:
        path=report/(split+'_predictions.json')
        if sha(path)!=state['official_metrics'][split]['prediction_sha256']:raise ValueError('Stale predictions')
        packets={p['sample_token']:p for p in json.loads((report.parent/'inputs'/(split+'_packets.json')).read_text())}
        gt=filter_eval_boxes(nusc,add_center_dist(nusc,load_gt_of_sample_tokens(nusc,list(packets),DetectionBox,verbose=False)),config.class_range,verbose=False)
        pred,_=load_prediction(str(path),500,DetectionBox,verbose=False)
        pred=filter_eval_boxes(nusc,add_center_dist(nusc,pred),config.class_range,verbose=False)
        events=[];frames=[];render_rows={}
        for token in pred.sample_tokens:
            targets=gt[token];used=set();rows=[]
            for box in sorted(pred[token],key=lambda b:-b.detection_score):
                if box.detection_score<.25:continue
                choices=[(np.linalg.norm(np.asarray(box.translation[:2])-g.translation[:2]),i) for i,g in enumerate(targets) if i not in used and g.detection_name==box.detection_name]
                distance,index=min(choices,default=(float('inf'),-1))
                kind='TP' if distance<2 else 'FP'
                if kind=='TP':used.add(index)
                rows.append((kind,box))
            rows.extend(('FN',box) for i,box in enumerate(targets) if i not in used)
            counts={kind:sum(k==kind for k,b in rows) for kind in ['TP','FP','FN']}
            frames.append(dict(sample_token=token,**counts));render_rows[token]=rows
            for kind,box in rows:
                if kind=='TP':continue
                events.append(dict(sample_token=token,kind=kind,class_name=box.detection_name,
                    confidence=box.detection_score if kind=='FP' else None,distance_m=box.ego_dist,
                    translation=list(box.translation),size=list(box.size),rotation=list(box.rotation)))
        out=report/split/'badcases';out.mkdir(parents=True,exist_ok=True)
        totals={kind:sum(f[kind] for f in frames) for kind in ['TP','FP','FN']}
        summary=dict(protocol='Official GT/range/points/bicycle-rack filters, display score>=0.25, score-ordered same-class XY distance<2m greedy matching; teaching diagnostic, not AP',
            prediction_sha256=sha(path),totals=totals,frames=frames,events=events)
        (out/'cases.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        for rank,frame in enumerate(sorted(frames,key=lambda f:-(f['FP']+f['FN']))[:3]):
            token=frame['sample_token'];sensor=packets[token]['sensors']['LIDAR_TOP']
            raw=np.fromfile(sensor['path'],dtype=np.float32).reshape(-1,5)[:,:3]
            to_ego=np.asarray(sensor['T_sensor_to_ego']);points=raw@to_ego[:3,:3].T+to_ego[:3,3]
            pose=np.asarray(sensor['T_ego_to_global']);inv=np.linalg.inv(pose)
            fig,ax=plt.subplots(figsize=(9,9));ax.set_facecolor('#111a25');ax.scatter(-points[:,1],points[:,0],s=.1,c='#637082',rasterized=True)
            for kind,box in render_rows[token]:
                center=inv[:3,:3]@np.asarray(box.translation)+inv[:3,3]
                rot=inv[:3,:3]@Quaternion(box.rotation).rotation_matrix
                w,l,h=box.size
                local=np.array([[l/2,w/2,0],[l/2,-w/2,0],[-l/2,-w/2,0],[-l/2,w/2,0],[l/2,w/2,0]])
                corners=local@rot.T+center
                color={'TP':'#44cc88','FP':'#ff5263','FN':'#ffbb44'}[kind]
                ax.plot(-corners[:,1],corners[:,0],color=color,lw=.8 if kind=='TP' else 1.4)
            ax.set(xlim=(-55,55),ylim=(-55,55),aspect='equal',xlabel='-ego y (m)',ylabel='ego x (m)',
                title=f"{split}: TP={frame['TP']} FP={frame['FP']} FN={frame['FN']}\n{token}\nGreen TP / red FP / orange FN; score>=.25, XY<2m")
            fig.tight_layout();fig.savefig(out/f'case_{rank+1}.png',dpi=120);plt.close(fig)
        print(split,totals)


if __name__=='__main__':main()
