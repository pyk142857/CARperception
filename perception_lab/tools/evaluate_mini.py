"""Evaluate measured mini detections/tracks and export traceable teaching failure cases."""
import argparse
import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import points_in_box
from evaluation_utils import match_boxes, identity_events
from rerun_mini import measured_results
from evidence import sha256
from tracking_overlay import bev_outline

ROOT=Path(__file__).resolve().parents[1]
TRACK_CLASSES={'bicycle','bus','car','motorcycle','pedestrian','trailer','truck'}


def write_csv(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def aggregate(rows):
    tp=sum(r['tp'] for r in rows);fp=sum(r['fp'] for r in rows);fn=sum(r['fn'] for r in rows)
    errors=[v for r in rows for v in r.get('errors',[])]
    return dict(tp=tp,fp=fp,fn=fn,gt=tp+fn,predictions=tp+fp,
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,
                mean_center_error_m=float(np.mean(errors)) if errors else None,
                median_center_error_m=float(np.median(errors)) if errors else None)


def load_frames(packets,detections,tracks):
    nusc=NuScenes(version='v1.0-mini',dataroot=str(ROOT/'data/nuscenes'),verbose=False)
    ranges=config_factory('detection_cvpr_2019').class_range
    frames=[];excluded=Counter()
    for packet,det,track in zip(packets,detections,tracks):
        pose=np.array(packet['sensors']['LIDAR_TOP']['T_ego_to_global']);inv=np.linalg.inv(pose)
        ann=[nusc.get('sample_annotation',t) for t in nusc.get('sample',packet['sample_token'])['anns']]
        racks=[Box(a['translation'],a['size'],Quaternion(a['rotation'])) for a in ann if a['category_name']=='static_object.bicycle_rack']
        def accept(box,group):
            cls=box['class_name']
            if cls not in ranges:excluded[group+'_unsupported_class']+=1;return False
            if np.linalg.norm(np.array(box['translation'][:2])-pose[:2,3])>=ranges[cls]:
                excluded[group+'_outside_range']+=1;return False
            if box.get('num_pts',-1)==0:excluded[group+'_zero_points']+=1;return False
            if cls in {'bicycle','motorcycle'} and any(points_in_box(r,np.array(box['translation']).reshape(3,1))[0] for r in racks):
                excluded[group+'_bike_rack']+=1;return False
            return True
        gt=[]
        for a in ann:
            cls=category_to_detection_name(a['category_name'])
            box=dict(class_name=cls,translation=a['translation'],center_xyz=(inv[:3,:3]@a['translation']+inv[:3,3]).tolist(),
                     size_wlh=a['size'],rotation_wxyz=(Quaternion(matrix=inv[:3,:3])*Quaternion(a['rotation'])).elements.tolist(),
                     instance_token=a['instance_token'],annotation_token=a['token'],num_pts=a['num_lidar_pts']+a['num_radar_pts'],
                     num_lidar_pts=a['num_lidar_pts'],visibility_token=a['visibility_token'])
            if accept(box,'gt'):gt.append(box)
        def predictions(boxes,group):
            out=[]
            for index,b in enumerate(boxes):
                box=dict(b,prediction_index=index,translation=(pose[:3,:3]@np.array(b['center_xyz'])+pose[:3,3]).tolist())
                if not np.isfinite(box['translation']).all() or not np.isfinite(box['score']):raise ValueError('Non-finite prediction')
                if accept(box,group):out.append(box)
            return out
        frames.append(dict(gt=gt,detection=predictions(det['boxes3d'],'detection'),tracking=predictions(track['tracks3d'],'tracking'),pose=pose.tolist()))
    return frames,ranges,dict(excluded)


def evaluate(frames,packets,score,gate):
    totals={};cases=[];frame_rows=[];class_rows=[];details={};hist={};previous={};last_scene=None
    for module in ['detection','tracking']:
        rows=[];by_class=defaultdict(list);module_events=Counter()
        for i,(data,packet) in enumerate(zip(frames,packets)):
            scene=packet['scene_token']
            if scene!=last_scene:previous={};last_scene=scene
            gt=[b for b in data['gt'] if module=='detection' or b['class_name'] in TRACK_CLASSES]
            pred=[b for b in data[module] if b['score']>=score]
            if module=='tracking' and len({b['tracking_id'] for b in pred})!=len(pred):raise ValueError('Duplicate tracking IDs in frame')
            matched,fn,fp=match_boxes(gt,pred,gate,tracking=module=='tracking',previous=previous)
            errors=[d for g,p,d in matched]
            row=dict(module=module,frame=i,sample_token=packet['sample_token'],tp=len(matched),fn=len(fn),fp=len(fp),errors=errors)
            rows.append(row)
            details[(module,i)]=(gt,pred,matched,fn,fp)
            base=dict(module=module,frame=i,sample_token=packet['sample_token'],timestamp_us=packet['timestamp_us'],
                      elapsed_seconds=(packet['timestamp_us']-packets[0]['timestamp_us'])/1e6)
            def case(kind,box,**extra):
                pose=np.array(data['pose']);distance=float(np.linalg.norm(np.array(box['translation'][:2])-pose[:2,3]))
                cases.append(dict(base,kind=kind,class_name=box['class_name'],instance_token=box.get('instance_token'),
                                  tracking_id=box.get('tracking_id'),prediction_index=box.get('prediction_index'),
                                  score=box.get('score'),distance_m=distance,num_lidar_pts=box.get('num_lidar_pts'),
                                  visibility_token=box.get('visibility_token'),center_ego=box['center_xyz'],**extra))
            for g in fn:
                same=[np.linalg.norm(np.array(gt[g]['translation'][:2])-p['translation'][:2]) for p in pred if p['class_name']==gt[g]['class_name']]
                case('false_negative',gt[g],nearest_same_class_m=float(min(same)) if same else None)
            for p in fp:
                same=[np.linalg.norm(np.array(pred[p]['translation'][:2])-g['translation'][:2]) for g in gt if g['class_name']==pred[p]['class_name']]
                case('false_positive',pred[p],nearest_same_class_m=float(min(same)) if same else None)
            for g,p,d in matched:
                if d>1.:case('center_error_over_1m',gt[g],matched_prediction_index=pred[p]['prediction_index'],center_error_m=d)
            for cls in sorted(set(b['class_name'] for b in gt+pred)):
                cm=[(g,p,d) for g,p,d in matched if gt[g]['class_name']==cls]
                by_class[cls].append(dict(tp=len(cm),fn=sum(gt[g]['class_name']==cls for g in fn),
                                         fp=sum(pred[p]['class_name']==cls for p in fp),errors=[d for g,p,d in cm]))
            event_counts=Counter()
            if module=='tracking':
                pairs=[(gt[g]['instance_token'],pred[p]['tracking_id']) for g,p,d in matched]
                events=identity_events(hist,scene,i,pairs)
                for event in events:
                    box=next(b for b in gt if b['instance_token']==event['instance_token'])
                    event=dict(event);kind=event.pop('kind');event.pop('instance_token')
                    # The GT has no tracking_id; avoid duplicate keyword from case().
                    target=event.pop('tracking_id')
                    case(kind,dict(box,tracking_id=target),**event)
                    event_counts[kind]+=1;module_events[kind]+=1
                previous=dict(pairs)
            frame_rows.append(dict(base,tp=len(matched),fn=len(fn),fp=len(fp),
                                   mean_center_error_m=float(np.mean(errors)) if errors else None,**event_counts))
        totals[module]=aggregate(rows)
        totals[module]['events']=dict(module_events)
        class_rows.extend(dict(module=module,class_name=cls,**aggregate(rs)) for cls,rs in sorted(by_class.items()))
    return totals,frame_rows,class_rows,cases,details


def render_cases(out,frames,packets,rows,cases,details):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[]
    for kind in ['id_switch','gap_id_change','false_negative','false_positive','center_error_over_1m']:
        if kind in {'id_switch','gap_id_change'}:
            candidates=[c for c in cases if c['kind']==kind]
            candidates=sorted(candidates,key=lambda c:(c.get('missed_frames',0),-c['frame']),reverse=True)
            keys=[(c['module'],c['frame']) for c in candidates]
        else:
            field={'false_negative':'fn','false_positive':'fp'}.get(kind)
            keys=[(r['module'],r['frame']) for r in sorted(rows,key=lambda r:r[field],reverse=True) if r['module']=='detection'] if field else [(c['module'],c['frame']) for c in sorted(cases,key=lambda c:c.get('center_error_m',0),reverse=True) if c['kind']==kind]
        added=0
        for key in keys:
            if key not in [(a,b) for a,b,k in chosen]:chosen.append((*key,kind));added+=1
            if added==2:break
    gallery=[]
    for module,i,reason in chosen:
        gt,pred,matched,fn,fp=details[(module,i)];packet=packets[i]
        sensor=packet['sensors']['LIDAR_TOP'];points=np.fromfile(sensor['path'],np.float32).reshape(-1,5)[::8,:3]
        t=np.array(sensor['T_sensor_to_ego']);points=points@t[:3,:3].T+t[:3,3]
        fig,ax=plt.subplots(figsize=(10,10));ax.scatter(-points[:,1],points[:,0],s=.4,c='#cccccc')
        # Plot forward upward; this is [-ego_y, ego_x], unlike image-coordinate Rerun BEV.
        def draw(box,color,label=None,style='-'):
            xy=bev_outline(box).copy();xy[:,1]*=-1
            ax.plot(xy[:,0],xy[:,1],style,color=color,linewidth=1.2)
            if label:ax.text(xy[0,0],xy[0,1],label,fontsize=6,color=color)
        for g,p,d in matched:draw(gt[g],'#16803a');draw(pred[p],'#1675c1',style='--')
        for g in fn:draw(gt[g],'#d02020',gt[g]['class_name'])
        for p in fp:draw(pred[p],'#d000ba',pred[p]['class_name'])
        current=[c for c in cases if c['module']==module and c['frame']==i and c['kind'] in {'id_switch','gap_id_change'}]
        for c in current:
            box=next(g for g in gt if g['instance_token']==c['instance_token'])
            draw(box,'#e59000',f"ID {c['previous_tracking_id']}->{c['tracking_id']}")
        ax.scatter([0],[0],marker='^',s=90,c='black');ax.set(xlim=(-55,55),ylim=(-55,55),xlabel='right (m)',ylabel='forward (m)',
            title=f'{module} | frame {i} | {reason}\nTP={len(matched)} FN={len(fn)} FP={len(fp)}\nGreen=matched GT; blue=prediction; red=miss; magenta=false positive; orange=ID change')
        ax.set_aspect('equal');ax.grid(alpha=.2);fig.tight_layout()
        name=f'{module}_{i:03d}_{reason}.png';fig.savefig(out/name,dpi=130);plt.close(fig)
        gallery.append(dict(module=module,frame=i,reason=reason,image=name,sample_token=packet['sample_token']))
    return gallery


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--score',type=float,default=.25)
    parser.add_argument('--distance',type=float,default=2.);parser.add_argument('--out',type=Path,default=ROOT/'reports/mini_evaluation')
    args=parser.parse_args()
    if not 0<=args.score<=1 or not np.isfinite(args.distance) or args.distance<=0:parser.error('Invalid thresholds')
    packets=json.loads((ROOT/'manifests/frame_packets.json').read_text());status=json.loads((ROOT/'results/status.json').read_text())
    det,detpath=measured_results(status,'M05','mini_scene',packets);tracks,trackpath=measured_results(status,'M11','mini_tracking_M05',packets)
    assert all(a['timestamp_us']<b['timestamp_us'] for a,b in zip(packets,packets[1:]))
    frames,ranges,excluded=load_frames(packets,det,tracks)
    totals,rows,classes,cases,details=evaluate(frames,packets,args.score,args.distance)
    out=args.out;out.mkdir(parents=True,exist_ok=True)
    for i,c in enumerate(cases):c['case_id']=f'case_{i:05d}'
    sweep=[]
    for score in sorted({.1,.25,.5,args.score}):
        values,*_=evaluate(frames,packets,score,args.distance)
        sweep.extend(dict(module=m,score=score,**v) for m,v in values.items())
    files=[detpath,trackpath,ROOT/'manifests/frame_packets.json',Path(__file__),ROOT/'tools/evaluation_utils.py']
    files.extend((ROOT/'data/nuscenes/v1.0-mini').glob('*.json'))
    summary=dict(scope='scene-0061 fixed-threshold teaching diagnostic; not official mAP/NDS/AMOTA',frames=len(frames),
                 score=args.score,distance_gate_m=args.distance,class_range_m=ranges,excluded_before_score=excluded,
                 matching={'detection':'same class, score-ordered greedy, global XY distance < gate',
                           'tracking':'preserve valid previous-frame pairs, then gated Hungarian; same class/global XY',
                           'id_switch':'GT matched in consecutive frames to different prediction IDs',
                           'gap_id_change':'GT reacquired with different ID after one or more unmatched/unobserved frames',
                           'gap_recovery':'same ID reacquired after gap; diagnostic event, not necessarily a failure'},
                 totals=totals,case_counts=dict(Counter(c['kind'] for c in cases)),source_sha256={str(p):sha256(p) for p in files})
    gallery=render_cases(out,frames,packets,rows,cases,details)
    breakdown=[]
    counts=Counter((c['module'],c['kind'],c['class_name']) for c in cases)
    for (module,kind,cls),count in sorted(counts.items()):
        breakdown.append(dict(module=module,kind=kind,class_name=cls,count=count))
    summary['failure_breakdown']=breakdown
    summary['gallery']=gallery
    write_csv(out/'failure_breakdown.csv',breakdown)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    (out/'cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False))
    write_csv(out/'cases.csv',cases);write_csv(out/'frames.csv',rows);write_csv(out/'classes.csv',classes);write_csv(out/'thresholds.csv',sweep)
    def fmt(v):return '—' if v is None else f'{v:.3f}'
    text=['# mini 检测与跟踪诊断', '',f'场景 scene-0061，{len(frames)} 帧；score ≥ {args.score}，同类别地面中心距离 < {args.distance} m。',
          '', '**仅为固定阈值教学诊断，不是官方 mAP、NDS 或 AMOTA；mini 可能与预训练数据重叠。**', '',
          '| 分支 | TP | FP | FN | 精确率 | 召回率 | 匹配中心平均误差/m |','|---|---:|---:|---:|---:|---:|---:|']
    for module,v in totals.items():text.append(f"| {module} | {v['tp']} | {v['fp']} | {v['fn']} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['mean_center_error_m'])} |")
    text+=['','## 统计口径','','检测含 10 类，跟踪含 7 类；两分支的 TP/FP/FN 不能直接相加。数量为逐帧目标次数，不是独立物体数。',
           '沿用官方类别映射、类别距离范围、零 LiDAR+radar 点真值过滤与自行车架过滤。中心误差仅在成功匹配目标上统计，不衡量尺寸或朝向。',
           '检测按分数从高到低一对一匹配；跟踪优先保留有效的前帧配对，再做门限内匈牙利匹配。阈值或匹配策略改变会影响结果。',
           'id_switch 是连续帧换 ID；gap_id_change 是间隔后换 ID，间隔可能来自漏检、过滤或离开评估范围；gap_recovery 表示同 ID 恢复，不直接认定为失败。',
           'center_error_over_1m 是已匹配目标的额外误差标记，不额外计入 FP/FN。误检、漏检的具体成因需要结合图片人工确认。',
           '', '## 跟踪身份事件','',json.dumps(totals['tracking']['events'],ensure_ascii=False), '', '## 分类结果','',
           '| 分支 | 类别 | TP | FP | FN | 精确率 | 召回率 |','|---|---|---:|---:|---:|---:|---:|']
    for r in classes:text.append(f"| {r['module']} | {r['class_name']} | {r['tp']} | {r['fp']} | {r['fn']} | {fmt(r['precision'])} | {fmt(r['recall'])} |")
    text+=['','## 分数阈值对比（同一场景，不用于选择泛化最优阈值）','','| 分支 | score | 精确率 | 召回率 | FP | FN |','|---|---:|---:|---:|---:|---:|']
    for r in sweep:text.append(f"| {r['module']} | {r['score']} | {fmt(r['precision'])} | {fmt(r['recall'])} | {r['fp']} | {r['fn']} |")
    text+=['','## 失败统计摘要','',
           '| 分支 | 事件 | 类别 | 次数 |','|---|---|---|---:|']
    for r in breakdown:
        text.append(f"| {r['module']} | {r['kind']} | {r['class_name']} | {r['count']} |")
    text+=['','## 失败案例入口','','[本地图片画廊](index.html) · [逐帧统计](frames.csv) · [完整案例清单](cases.csv) · [来源哈希与口径](summary.json)',
           '', 'frame 从 0 开始，与 Rerun 的 frame 时间轴一致；sample_token 可精确定位原数据。以下图片选择错误较多帧及身份变化帧，不代表随机样本。']
    for g in gallery:text+=['',f"### {g['module']} / frame {g['frame']} / {g['reason']}",'',f"sample_token: `{g['sample_token']}`",'',f"![BEV failure case]({g['image']})"]
    text+=['','## 复现','','```bash','cd perception_lab',f'envs/mmdet3d/bin/python tools/evaluate_mini.py --score {args.score} --distance {args.distance}','```','',
           '过滤规则参考：[nuScenes detection](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/README.md)。本脚本未运行官方整套评估。']
    (out/'report.md').write_text('\n'.join(text)+'\n')
    cards=''.join(f'<article><h2>{html.escape(g["module"])} / frame {g["frame"]} / {g["reason"]}</h2><p>{g["sample_token"]}</p><img src="{g["image"]}" loading="lazy"></article>' for g in gallery)
    (out/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>Mini evaluation failure cases</title><style>body{font:16px sans-serif;margin:30px auto;max-width:1200px;background:#fafafa}img{max-width:100%}article{border-top:1px solid #bbb;margin-top:30px}pre{white-space:pre-wrap}</style><h1>mini 检测与跟踪失败案例</h1><p>39 帧教学诊断；红色=漏检，紫色=误检，橙色=ID变化。frame 对应 Rerun 时间轴。完整口径见 report.md。</p><p><a href="report.md">完整报告</a> · <a href="cases.csv">案例 CSV</a> · <a href="frames.csv">逐帧统计</a></p><pre>'+html.escape(json.dumps(totals,indent=2,ensure_ascii=False))+'</pre>'+cards+'</html>')
    print(json.dumps(totals,indent=2));print('Report:',out/'report.md')

if __name__=='__main__':main()
