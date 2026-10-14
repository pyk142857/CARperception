"""Post-inference teaching case selection; never supplies GT to either tracker."""
import json
from pathlib import Path
import numpy as np
from tracking_gpu_visualize_20261013 import ROOT,REPORT,load_inputs,read_json,sha,ego_boxes,transform_points
from evaluation_utils import match_boxes,identity_events
from deployment_visualize import render,project_cloud


def main():
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import load_gt_of_sample_tokens,load_prediction_of_sample_tokens,add_center_dist,filter_eval_boxes
    from nuscenes.eval.tracking.data_classes import TrackingBox
    nusc=NuScenes(version='v1.0-mini',dataroot=str(ROOT/'data/nuscenes'),verbose=False)
    cfg=config_factory('tracking_nips_2019');out=REPORT/'visualization';result={}
    for split in ['scene0061','mini_val']:
        packets,artifacts,hashes=load_inputs(split);tokens=[p['sample_token'] for p in packets]
        truth=filter_eval_boxes(nusc,add_center_dist(nusc,load_gt_of_sample_tokens(nusc,tokens,TrackingBox,verbose=False)),cfg.class_range,verbose=False)
        rows={};events_by_model={}
        for model in ['grae','optipmb']:
            path=REPORT/'native_compare'/model
            if model=='grae':path=path/'frozen_reference'
            path=path/split/'tracking.json'
            predictions,_=load_prediction_of_sample_tokens(str(path),cfg.max_boxes_per_sample,TrackingBox,sample_tokens=tokens,verbose=False)
            predictions=filter_eval_boxes(nusc,add_center_dist(nusc,predictions),cfg.class_range,verbose=False)
            history={};previous={};last_scene=None;rows[model]=[];events_by_model[model]=[]
            for i,p in enumerate(packets):
                token=p['sample_token'];scene=p['scene_token']
                if scene!=last_scene:previous={};last_scene=scene
                gt=[dict(class_name=b.tracking_name,translation=list(b.translation),instance_token=b.tracking_id) for b in truth[token]]
                pred=[dict(class_name=b.tracking_name,translation=list(b.translation),tracking_id=b.tracking_id,score=b.tracking_score) for b in predictions[token] if b.tracking_score>=.25]
                matches,fn,fp=match_boxes(gt,pred,2.,tracking=True,previous=previous)
                pairs=[(gt[g]['instance_token'],pred[j]['tracking_id']) for g,j,_ in matches]
                events=identity_events(history,scene,i,pairs);previous=dict(pairs)
                rows[model].append(dict(frame=i,sample_token=token,scene_token=scene,tp=len(matches),fp=len(fp),fn=len(fn),id_switch=sum(e['kind']=='id_switch' for e in events),
                    false_positive_predictions=[pred[j] for j in fp],false_negative_truth=[gt[j] for j in fn],identity_events=events))
                events_by_model[model].extend([dict(e,frame=i,sample_token=token) for e in events])
        comparisons=[]
        for g,o in zip(rows['grae'],rows['optipmb']):
            comparisons.append(dict(frame=g['frame'],sample_token=g['sample_token'],grae=g,optipmb=o,
                delta_optipmb_minus_grae={key:o[key]-g[key] for key in ['tp','fp','fn','id_switch']}))
        selected=[]
        # Fixed before inspecting outcomes: FP/FN/IDS changes individually, earliest frame breaks ties.
        for metric in ['fp','fn','id_switch']:
            for direction in ['improvement','degradation']:
                candidates=[r for r in comparisons if (r['delta_optipmb_minus_grae'][metric]<0 if direction=='improvement' else r['delta_optipmb_minus_grae'][metric]>0)]
                if not candidates:continue
                row=sorted(candidates,key=lambda r:(-abs(r['delta_optipmb_minus_grae'][metric]),r['frame']))[0]
                selected.append(dict(reason=metric+'_'+direction,**row))
        for item in selected:
            i=item['frame'];p=packets[i];lidar=p['sensors']['LIDAR_TOP'];pose=np.asarray(lidar['T_ego_to_global'])
            cloud=transform_points(np.fromfile(lidar['path'],dtype=np.float32).reshape(-1,5)[:,:3],np.asarray(lidar['T_sensor_to_ego']))
            item['images']={}
            for model in ['grae','optipmb']:
                name=f'{split}_{item["reason"]}_{model}_{i:02d}.jpg'
                boxes=ego_boxes(artifacts[model][p['sample_token']],pose,model)
                im=render(p,model+' / '+item['reason']+' / teaching only',boxes,cloud,None,[],i,{})
                from PIL import ImageDraw
                draw=ImageDraw.Draw(im)
                draw.rectangle((1340,15,1919,55),fill=(8,14,22))
                draw.text((1340,22),f'frame {i:02d}/{len(packets)-1} | teaching',fill=(180,205,220))
                draw.text((22,55),'BEV red circles: FP; orange crosses: FN (post-inference matching)',fill=(255,200,100))
                for field,color in [('false_positive_predictions',(255,60,60)),('false_negative_truth',(255,180,30))]:
                    for error in item[model][field]:
                        point=transform_points(np.asarray([error['translation']]),np.linalg.inv(pose))
                        x,y=project_cloud(point,'bev')[0];y+=62
                        if not (5<x<635 and 67<y<1000):continue
                        if field=='false_positive_predictions':draw.ellipse((x-6,y-6,x+6,y+6),outline=color,width=3)
                        else:
                            draw.line((x-6,y-6,x+6,y+6),fill=color,width=3);draw.line((x-6,y+6,x+6,y-6),fill=color,width=3)
                im.save(out/name,quality=86);item['images'][model]=name
        result[split]=dict(artifact_sha256=hashes,frames=comparisons,selected=selected,
            totals={m:{k:sum(r[k] for r in rows[m]) for k in ['tp','fp','fn','id_switch']} for m in rows},events=events_by_model)
    report=dict(status='passed',protocol='Post-inference only. Official devkit GT/pred class-range/point/rack filters; display score>=.25; same-class XY<2m; preserve previous valid match then Hungarian; raw tracks without interpolation. Teaching IDSW are not official IDS.',
        selection_rule='For each split and FP/FN/IDSW separately, choose greatest strictly negative (improvement) and strictly positive (degradation) Opti-minus-GRAE delta; earliest frame tie break. Omit absent directions. No tuning.',splits=result)
    (out/'teaching_cases.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:dict(totals=v['totals'],selected=[(x['reason'],x['frame']) for x in v['selected']]) for s,v in result.items()}))

if __name__=='__main__':main()
