"""Pinned GRAE-3DMOT mini inference; GT-free detection reader and strict weights.

Uses the author's model and predictor unchanged for native association. SimpleTrack
is imported solely for the author's NMS dependency. No training or full-val data.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import sys
import textwrap
import time
import traceback

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'third_party/GRAE-3DMOT'
DEPENDENCY = ROOT / 'third_party/SimpleTrack_grae'
CHECKPOINT = ROOT / 'checkpoints/grae3dmot_2025/model.pth'
OUT = ROOT / 'reports/perception_2025_20261013/grae3dmot'
COMMIT = '63def8bde5e199a4e77fdf4fab76a4b3511fe132'
CLASSES = ['car','pedestrian','bicycle','bus','motorcycle','trailer','truck']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def normalize_weights(weights):
    prefixes=[k.startswith('module.') for k in weights]
    if any(prefixes) and not all(prefixes):raise ValueError('Mixed module prefix in checkpoint')
    return {k[7:]:v for k,v in weights.items()} if all(prefixes) else weights


def checkpoint_name_adapter(weights):
    result={}
    for key,value in weights.items():
        target=key
        for old,new in [('spatial_mfl.','spatial_dffl.'),('temporal_mfl.','temporal_dffl.')]:
            if key.startswith(old):target=new+key[len(old):]
        if target in result:raise ValueError('Checkpoint rename collision '+target)
        result[target]=value
    return result


def frame_arrays(boxes,packet,nms=True):
    """Allowlist only predicted global boxes and timestamp. Never copy annotation IDs."""
    chosen=[b for b in boxes if b['detection_name'] in CLASSES]
    shapes={'translation':3,'size':3,'rotation':4,'velocity':2}
    result={k:np.asarray([b[k] for b in chosen],dtype=np.float64).reshape(-1,width) for k,width in shapes.items()}
    for k,v in result.items():
        if not np.isfinite(v).all():raise ValueError('Nonfinite predicted '+k)
    if (result['size']<=0).any():raise ValueError('Nonpositive box size')
    norm=np.linalg.norm(result['rotation'],axis=1)
    if not np.allclose(norm,1,atol=1e-5):raise ValueError('Nonunit quaternion')
    q=result['rotation']
    result['yaw']=np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
    result['score']=np.asarray([b['detection_score'] for b in chosen],dtype=np.float64)
    result['classes']=np.asarray([CLASSES.index(b['detection_name']) for b in chosen],dtype=np.int64)
    if not np.isfinite(result['score']).all():raise ValueError('Nonfinite score')
    if nms and chosen:
        sys.path.insert(0,str(DEPENDENCY))
        from mot_3d.preprocessing import nms as official_nms
        from mot_3d.data_protos import BBox
        from pyquaternion import Quaternion
        bboxes=[BBox(x=b['translation'][0],y=b['translation'][1],z=b['translation'][2],
                     w=b['size'][0],l=b['size'][1],h=b['size'][2],o=Quaternion(b['rotation']).yaw_pitch_roll[0]) for b in chosen]
        for bbox,b in zip(bboxes,chosen):bbox.s=b['detection_score']
        indices,_=official_nms(bboxes,result['classes'],0.1)
        result={k:v[indices] for k,v in result.items()}
    # This is exactly the val reader's filtering boundary, separate from evaluation.
    mask=result['score']>=.1
    result={k:v[mask] for k,v in result.items()}
    result['tracking_id']=np.full(len(result['score']),-1,dtype=np.int64)
    result.update(sample_token=packet['sample_token'],scene_token=packet['scene_token'],timestamp=packet['timestamp_us']/1e6)
    return result


def make_instance(frame,device,sentinel=-1):
    import torch
    from models.structures import Instances
    inst=Instances([1,1],{k:frame[k] for k in ['sample_token','scene_token','timestamp']})
    for k in ['translation','size','yaw','rotation','velocity','score','classes','tracking_id']:
        value=torch.from_numpy(frame[k].copy())
        if k=='tracking_id':value.fill_(sentinel)
        if k in ['yaw','score','classes','tracking_id']:value=value.unsqueeze(-1)
        inst.set(k,value)
    inst.set('times',torch.full((len(inst),1),frame['timestamp'],dtype=torch.float64))
    inst.set('instance_inds',torch.full((len(inst),1),-1,dtype=torch.int64))
    return inst.to(device)


def new_predictor(model,device,batched=False):
    import trainer.predictor as original
    predictor=original.BasePredictor.__new__(original.BasePredictor)
    predictor.model=model;predictor.device=device
    predictor.alpha=.24;predictor.conf_th=.12;predictor.age=12
    predictor.class_names=CLASSES;predictor.save_notmatched_track=False
    if batched:
        source=textwrap.dedent(inspect.getsource(original.BasePredictor._test_mini_seq))
        for obj in ['dets','draw_track']:
            loop='for idx in range(len('+obj+')):'
            # Preserve all association/state math, only batch the final output transfers.
            lines=source.splitlines();changed=[]
            for line in lines:
                if line.strip()==loop:
                    indent=line[:len(line)-len(line.lstrip())]
                    changed.append(indent+"_export = {k: getattr("+obj+", k).detach().cpu().numpy() for k in ['translation','size','rotation','velocity','instance_inds','classes','score']}")
                changed.append(line)
            source='\n'.join(changed)+'\n'
            for field in ['translation','size','rotation','velocity','instance_inds','score']:
                source=source.replace(obj+'.'+field+'[idx].cpu().numpy()',"_export['"+field+"'][idx]")
            source=source.replace('int('+obj+'.classes[idx])',"int(_export['classes'][idx])")
        namespace=dict(original.__dict__)
        exec(compile(source,'<GRAE batched output transfer>','exec'),namespace)
        import types
        predictor._test_mini_seq=types.MethodType(namespace['_test_mini_seq'],predictor)
    return predictor


def run_sequence(model,frames,device,batched=False,sentinel=-1):
    import torch
    p=new_predictor(model,device,batched)
    instances=[make_instance(f,device,sentinel) for f in frames]
    for inst in instances:
        if len(inst):
            p.setup_instance(inst)
            inst.time=torch.full((len(inst),1),inst._img_meta['timestamp'],dtype=torch.float64,device=device)
    with torch.inference_mode():result=p._test_mini_seq(instances)
    return result


def load_model(device):
    import torch
    from models import GRAE
    if not CHECKPOINT.exists():raise FileNotFoundError('Official complete checkpoint is not locally materialized: '+str(CHECKPOINT))
    cfg=json.loads((SOURCE/'config/val.json').read_text())
    arch=dict(cfg['arch']['args']);arch['device']=device
    checkpoint=torch.load(CHECKPOINT,map_location='cpu')
    weights=normalize_weights(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    original_weights=weights
    weights=checkpoint_name_adapter(weights)
    model=GRAE(**arch)
    incompatible=model.load_state_dict(weights,strict=True)
    loaded=model.state_dict()
    mapping=[]
    for key,tensor in original_weights.items():
        target=next(iter(checkpoint_name_adapter({key:tensor})))
        if not torch.equal(tensor,loaded[target]):raise ValueError('Strict load tensor changed '+key)
        if target!=key:
            mapping.append({'original':key,'target':target,'shape':list(tensor.shape),
                'sha256':hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),'loaded_bytes_equal':True})
    write(OUT/'checkpoint_key_mapping.json',{'scope':'exact module attribute prefix only; no permutation, cast, dropped or invented tensor',
        'pairs':mapping,'all_loaded_tensors_equal_source':True,'tensor_count':len(weights),
        'source_module_type':'Both target named spatial_dffl/temporal_dffl are DFFL(in_channels128,dist1,out128), each with _layer/reduce/gamma/beta',
        'first_unadapted_strict_failure':'strict_load_initial_failure.log'})
    write(OUT/'strict_load.json',{'checkpoint_sha256':sha(CHECKPOINT),'config':arch,
        'missing':list(incompatible.missing_keys),'unexpected':list(incompatible.unexpected_keys),
        'loaded_tensors':len(weights),'parameters':sum(p.numel() for p in model.parameters()),'strict':True})
    return model.to(device).eval()


def sources(config,input_name):
    if input_name=='frozen_reference':
        return {split:ROOT/'reports/deployment_4090_20261008/bevfusion_reference'/split/'evaluation/official_predictions.json' for split in ['scene0061','mini_val']}
    configured=config.get('models',{}).get('grae3dmot',{}).get('mamba_inputs',{})
    return {split:ROOT/configured.get(split,'reports/perception_2025_20261013/mambafusion/'+split+'_predictions.json') for split in ['scene0061','mini_val']}


def prepare(config,input_name):
    if input_name=='mambafusion_native':
        folder=ROOT/config.get('report','reports/perception_2025_20261013')/'mambafusion'
        state=json.loads((folder/'status.json').read_text())
        if state.get('status') not in ('passed','partial') or state.get('official_evaluation')!='passed':
            raise ValueError('Current completed Mamba native inference and evaluation required')
        for split,path in sources(config,input_name).items():
            bound=(state.get('official_metrics') or {}).get(split,{})
            if not path.is_file() or sha(path)!=bound.get('prediction_sha256'):
                raise ValueError('Mamba input differs from evaluated native output')
    files=sources(config,input_name);sequences={};manifest=[]
    for split,path in files.items():
        if not path.exists():raise FileNotFoundError('Detection input unavailable: '+str(path))
        detections=json.loads(path.read_text());packets=json.loads((ROOT/config['splits'][split]).read_text())
        tokens=[p['sample_token'] for p in packets]
        if set(detections['results'])!=set(tokens):raise ValueError('Detection/sample coverage mismatch '+split)
        groups={};before=after=0;times=[];preprocess_times=[]
        for packet in packets:
            before+=len(detections['results'][packet['sample_token']])
            started=time.perf_counter()
            f=frame_arrays(detections['results'][packet['sample_token']],packet)
            preprocess_times.append({'sample_token':packet['sample_token'],'nms_reader_ms':(time.perf_counter()-started)*1000})
            after+=len(f['score']);groups.setdefault(f['scene_token'],[]).append(f)
        for scene,frames in groups.items():
            ts=[f['timestamp'] for f in frames];delta=np.diff(ts)
            if not (delta>0).all():raise ValueError('Nonmonotonic scene timestamps')
            times.extend(delta.tolist())
        sequences[split]=groups
        manifest.append({'split':split,'detection_path':str(path),'sha256':sha(path),'frames':len(packets),
                         'detections_before_class_nms_score_filter':before,'inference_detections':after,
                         'scene_count':len(groups),'timestamp_deltas_seconds':times,
                         'packets_sha256':sha(ROOT/config['splits'][split]),'cpu_reader_nms_timings':preprocess_times})
    write(OUT/input_name/'input_manifest.json',{'input_family':input_name,'origin':'reused_archive' if input_name=='frozen_reference' else 'new_measured',
        'reader_fields':'predicted global translation/wlh/quaternion/xy_velocity/score/class and scene/sample/timestamp only',
        'gt_tracking_id':'sentinel -1; future/GT fields excluded','nms':'author SimpleTrack IoU=.1, seven tracking classes; author val score>=.1',
        'splits':manifest})
    return sequences


def environment():
    import torch,platform,importlib.metadata
    return {'python':platform.python_version(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'dependencies':{k:importlib.metadata.version(k) for k in ['lap','fvcore','numpy','scipy','nuscenes-devkit']},
            'environment_scope':'isolated envs/grae3dmot_2025 read-only overlay of existing bevfusion/maptr torch; old environments unchanged'}


def evaluate(config,input_name):
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.tracking.evaluate import TrackingEval
    rows=[]
    for split in ['scene0061','mini_val']:
        folder=OUT/input_name/split
        summary=folder/'official_eval/metrics_summary.json'
        before=summary.stat() if summary.exists() else None
        TrackingEval(config_factory('tracking_nips_2019'),str(folder/'tracking.json'),
                     'mini_scene_0061' if split=='scene0061' else 'mini_val',str(folder/'official_eval'),
                     'v1.0-mini',str(ROOT/config['data']),verbose=True).main(render_curves=False)
        if not summary.exists() or summary.stat()==before:
            raise RuntimeError('Official evaluator did not publish fresh metrics: '+split)
        metrics=json.loads(summary.read_text())
        packets=json.loads((ROOT/config['splits'][split]).read_text())
        predictions=json.loads((folder/'tracking.json').read_text())['results']
        if set(predictions)!={p['sample_token'] for p in packets}:
            raise ValueError('Evaluation tracking token coverage mismatch '+split)
        rows.append(dict(input=input_name,split=split,frames=len(predictions),
            tracking_sha256=sha(folder/'tracking.json'),metrics_sha256=sha(summary),
            input_manifest_sha256=sha(OUT/input_name/'input_manifest.json'),
            **{key:metrics[key] for key in ['amota','amotp','ids','fp','fn','recall','mota']}))
    return rows


def postprocess(config,input_name,phase):
    """Evaluate frozen inference only; publish a distinct fresh runner receipt."""
    receipt=OUT/(phase+'_status.json')
    status_path=OUT/input_name/'status.json'
    if not status_path.exists():
        legacy=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
        if legacy.get('input_family')==input_name:write(status_path,legacy)
    state=json.loads(status_path.read_text()) if status_path.exists() else {}
    if state.get('inference')!='passed' or state.get('input_family')!=input_name or state.get('status') not in ('passed','partial'):
        write(receipt,dict(status='blocked_data',reason='Matching completed inference required',input_family=input_name))
        return 2
    write(receipt,dict(status='running',input_family=input_name))
    if phase=='evaluate':
        state['official_evaluation']='running';write(status_path,state)
    try:
        if phase=='evaluate':
            if input_name=='mambafusion_native':
                native=json.loads((OUT/input_name/'native_receipt.json').read_text())
                for split in ['scene0061','mini_val']:
                    if sha(OUT/input_name/split/'tracking.json')!=native['tracking_sha256'][split]:
                        raise ValueError('Tracking output differs from current inference receipt')
            rows=evaluate(config,input_name)
            write(OUT/input_name/'metrics.json',rows)
            # Retain each detection family independently, including frozen baseline.
            old=json.loads((OUT/'metrics.json').read_text()) if (OUT/'metrics.json').exists() else []
            rows=[r for r in old if r.get('input','frozen_reference')!=input_name]+rows
            write(OUT/'metrics.json',rows)
            import csv
            with (OUT/'metrics.csv').open('w') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(key for row in rows for key in row)),lineterminator="\n");writer.writeheader();writer.writerows(rows)
            state['official_evaluation']='passed';write(status_path,state);write(OUT/'status.json',state)
        else:
            teaching_audit(config,input_name)
        write(receipt,dict(status='passed',input_family=input_name,phase=phase))
        return 0
    except Exception as exc:
        if phase=='evaluate':
            state['official_evaluation']='failed';write(status_path,state);write(OUT/'status.json',state)
        write(receipt,dict(status='failed',input_family=input_name,error=repr(exc)))
        raise


def teaching_audit(config,input_name):
    """Independent GT evaluation only, after all tracker predictions are frozen."""
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.common.config import config_factory
    from nuscenes.eval.common.loaders import (load_gt_of_sample_tokens,load_prediction_of_sample_tokens,
                                              add_center_dist,filter_eval_boxes)
    from nuscenes.eval.tracking.data_classes import TrackingBox
    from evaluation_utils import match_boxes,identity_events
    nusc=NuScenes(version='v1.0-mini',dataroot=str(ROOT/config['data']),verbose=False)
    cfg=config_factory('tracking_nips_2019')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for split in ['scene0061','mini_val']:
        packets=json.loads((ROOT/config['splits'][split]).read_text())
        tokens=[p['sample_token'] for p in packets]
        folder=OUT/input_name/split
        gt=load_gt_of_sample_tokens(nusc,tokens,TrackingBox,verbose=False)
        pred,_=load_prediction_of_sample_tokens(str(folder/'tracking.json'),cfg.max_boxes_per_sample,
                                               TrackingBox,sample_tokens=tokens,verbose=False)
        gt=filter_eval_boxes(nusc,add_center_dist(nusc,gt),cfg.class_range,verbose=False)
        pred=filter_eval_boxes(nusc,add_center_dist(nusc,pred),cfg.class_range,verbose=False)
        history={};previous={};last_scene=None;rows=[];cases=[];cached={}
        for frame,packet in enumerate(packets):
            token,scene=packet['sample_token'],packet['scene_token']
            if scene!=last_scene:previous={};last_scene=scene
            truth=[{'class_name':b.tracking_name,'translation':list(b.translation),'instance_token':b.tracking_id} for b in gt[token]]
            predictions=[{'class_name':b.tracking_name,'translation':list(b.translation),'tracking_id':b.tracking_id,
                          'score':b.tracking_score} for b in pred[token] if b.tracking_score>=.25]
            matches,fn,fp=match_boxes(truth,predictions,2.,tracking=True,previous=previous)
            pairs=[(truth[g]['instance_token'],predictions[p]['tracking_id']) for g,p,d in matches]
            events=identity_events(history,scene,frame,pairs)
            for event in events:
                g=next(v for v in truth if v['instance_token']==event['instance_token'])
                cases.append(dict(event,sample_token=token,scene_token=scene,frame=frame,
                                  class_name=g['class_name'],translation=g['translation']))
            previous=dict(pairs)
            rows.append({'frame':frame,'sample_token':token,'scene_token':scene,'tp':len(matches),'fp':len(fp),'fn':len(fn),
                         'id_switch':sum(e['kind']=='id_switch' for e in events),
                         'gap_id_change':sum(e['kind']=='gap_id_change' for e in events),
                         'gap_recovery':sum(e['kind']=='gap_recovery' for e in events)})
            cached[token]=(packet,truth,predictions)
        totals={key:sum(r[key] for r in rows) for key in ['tp','fp','fn','id_switch','gap_id_change','gap_recovery']}
        selected=[]
        for kind in ['id_switch','gap_id_change','gap_recovery']:
            eligible=[c for c in cases if c['kind']==kind]
            if eligible:selected.append(max(eligible,key=lambda c:c['missed_frames']))
        images=[]
        for index,case in enumerate(selected):
            packet,truth,predictions=cached[case['sample_token']]
            ego=np.asarray(packet['sensors']['LIDAR_TOP']['T_ego_to_global'])[:3,3]
            center=np.asarray(case['translation'])[:2]
            fig,ax=plt.subplots(figsize=(7,6))
            for values,color,label,marker in [(truth,'#218c45','GT','o'),(predictions,'#2269bc','GRAE output','x')]:
                points=np.asarray([v['translation'][:2] for v in values])
                if len(points):ax.scatter(points[:,0]-ego[0],points[:,1]-ego[1],c=color,label=label,marker=marker)
                for item in values:
                    if np.linalg.norm(np.asarray(item['translation'])[:2]-center)<12:
                        name=item.get('tracking_id',item.get('instance_token',''))
                        ax.annotate(name[-12:],np.asarray(item['translation'])[:2]-ego[:2],fontsize=6)
            ax.scatter(*(center-ego[:2]),facecolors='none',edgecolors='red',s=180,label=case['kind'])
            ax.set_xlim(center[0]-ego[0]-14,center[0]-ego[0]+14);ax.set_ylim(center[1]-ego[1]-14,center[1]-ego[1]+14)
            ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
            ax.set_title(split+' | '+case['kind']+' | frame '+str(case['frame'])+'\n'+case['sample_token'])
            ax.set_xlabel('global X relative to ego (m)');ax.set_ylabel('global Y relative to ego (m)')
            image='teaching_case_%02d.png'%index;fig.tight_layout();fig.savefig(folder/image,dpi=140);plt.close(fig)
            images.append(dict(case,image=image))
        write(folder/'teaching_audit.json',{'protocol':'Independent post-inference GT; score>=.25; same-class XY<2m; '
              'previous valid correspondence retained then Hungarian; GT range/point/rack filtering from official devkit; '
              'raw outputs without official track interpolation; these IDSW are not official IDS',
              'totals':totals,'frames':rows,'identity_events':cases,'representative_cases':images})



def instrument_export_loops(predictor):
    """Time original output loops without changing their statements or tensor math."""
    import ast
    import types
    import trainer.predictor as original
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.BasePredictor._test_mini_seq)))
    class Timers(ast.NodeTransformer):
        def visit_For(self,node):
            self.generic_visit(node)
            if ast.unparse(node.target)=='idx' and ast.unparse(node.iter) in ['range(len(dets))','range(len(draw_track))']:
                before=ast.parse('_export_start = _profile_clock()').body[0]
                after=ast.parse('self._profile_export_ms += (_profile_clock() - _export_start) * 1000').body[0]
                return [before,node,after]
            return node
    tree=ast.fix_missing_locations(Timers().visit(tree))
    namespace=dict(original.__dict__);namespace['_profile_clock']=time.perf_counter
    exec(compile(tree,'<GRAE original export timing>','exec'),namespace)
    predictor._test_mini_seq=types.MethodType(namespace['_test_mini_seq'],predictor)
    predictor._profile_export_ms=0.
    return predictor


def profile(config,input_name):
    import resource
    cold_start=time.perf_counter()
    import torch
    model=load_model('cuda:0');torch.cuda.synchronize()
    cold_ms=(time.perf_counter()-cold_start)*1000
    torch.backends.cudnn.benchmark=False
    detection_path=sources(config,input_name)['scene0061']
    packet_path=ROOT/config['splits']['scene0061']
    packets=json.loads(packet_path.read_text())[:3]
    raw=json.loads(detection_path.read_text())['results']
    frames=[frame_arrays(raw[p['sample_token']],p) for p in packets]
    for _ in range(3):run_sequence(model,frames,'cuda:0')
    baseline=run_sequence(model,frames,'cuda:0')
    pairs=[];active=[]
    def before(module,args):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();active.append((start,end))
    def after(module,args,result):
        start,end=active.pop();end.record();pairs.append((start,end))
    handles=[model.register_forward_pre_hook(before),model.register_forward_hook(after)]
    predictor=instrument_export_loops(new_predictor(model,'cuda:0'))
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    wall=time.perf_counter()
    raw=json.loads(detection_path.read_text())['results'];packets=json.loads(packet_path.read_text())[:3]
    read_ms=(time.perf_counter()-wall)*1000
    begin=time.perf_counter();frames=[frame_arrays(raw[p['sample_token']],p) for p in packets]
    reader_ms=(time.perf_counter()-begin)*1000
    begin=time.perf_counter();instances=[make_instance(f,'cpu') for f in frames]
    cpu_format_ms=(time.perf_counter()-begin)*1000
    transfer_start=torch.cuda.Event(enable_timing=True);transfer_end=torch.cuda.Event(enable_timing=True)
    transfer_start.record()
    instances=[inst.to('cuda:0') for inst in instances]
    for inst in instances:
        if len(inst):
            predictor.setup_instance(inst)
            inst.time=torch.full((len(inst),1),inst._img_meta['timestamp'],dtype=torch.float64,device='cuda:0')
    transfer_end.record()
    begin=time.perf_counter()
    with torch.inference_mode():result=predictor._test_mini_seq(instances)
    torch.cuda.synchronize();association_ms=(time.perf_counter()-begin)*1000
    e2e_ms=(time.perf_counter()-wall)*1000
    for handle in handles:handle.remove()
    if result!=baseline:raise ValueError('Profiling instrumentation changed native outputs')
    write(OUT/input_name/'segmented_profile.json',{
        'adapter_sha256':sha(__file__),'input_sha256':sha(detection_path),'frames':[p['sample_token'] for p in packets],
        'batch':1,'warmup_sequences':3,'cold_model_load_and_first_cuda_context_ms':cold_ms,
        'cold_scope':'Torch import if not cached, author model construction, strict checkpoint load, model H2D; excludes interpreter start',
        'input_read_wall_ms':read_ms,'allowlist_and_official_nms_wall_ms':reader_ms,
        'cpu_instance_format_wall_ms':cpu_format_ms,'h2d_and_initial_state_cuda_ms':transfer_start.elapsed_time(transfer_end),
        'model_forward_cuda_ms':sum(a.elapsed_time(b) for a,b in pairs),'model_forward_calls':len(pairs),
        'native_association_and_export_wall_ms':association_ms,'final_output_d2h_and_record_wall_ms':predictor._profile_export_ms,
        'raw_detection_file_to_consumable_tracks_wall_ms':e2e_ms,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'outputs_exactly_equal_uninstrumented':True,
        'boundary_notes':'3 consecutive frames; whole source JSON parsed; filesystem cache warm; coordinate system already global so no coordinate conversion. GPU forward excludes graph preparation, matching and export. Forward CUDA and export wall are subregions of association wall, not additive. Cold model includes strict verification and manifest writes. RAM is process lifetime high water including warmup.',
        'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
        'precision':'float32','cudnn_benchmark':torch.backends.cudnn.benchmark,'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32})


def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','run','optimize','evaluate','audit','profile'],default='prepare')
    p.add_argument('--config',type=Path,default=ROOT/'configs/perception_2025.yaml')
    p.add_argument('--input',choices=['frozen_reference','mambafusion_native'],default='frozen_reference')
    p.add_argument('--output-dir',type=Path,help='Independent experiment output, relative to perception_lab or absolute; old default unchanged')
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if a.output_dir:
        selected=(ROOT/a.output_dir).resolve()
        if not selected.is_relative_to((ROOT/'reports').resolve()):
            raise ValueError('Output directory must be inside perception_lab/reports')
        OUT=selected
    method=cfg.get('models',{}).get('grae3dmot',{})
    for key,expected in [('source',SOURCE),('checkpoint',CHECKPOINT),('config',SOURCE/'config/val.json')]:
        base=SOURCE if key=='config' else ROOT
        if key in method and (base/method[key]).resolve()!=expected.resolve():
            raise ValueError('Unsupported GRAE config override: '+key)
    if not a.output_dir and 'report' in cfg and (ROOT/cfg['report']).resolve()!=OUT.parent.resolve():
        raise ValueError('Unsupported GRAE config override: report')
    OUT.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(SOURCE));sys.path.insert(0,str(DEPENDENCY))
    if a.phase in ('evaluate','audit'):return postprocess(cfg,a.input,a.phase)
    if a.phase=='profile':profile(cfg,a.input);return
    status={'model':'GRAE-3DMOT','status':'running','phase':a.phase,'input_family':a.input,'gpu_executed':False}
    status_path=OUT/('optimization.json' if a.phase=='optimize' else 'status.json')
    # Preserve the prior family's verified status before switching active input.
    if a.phase!='optimize' and status_path.exists():
        previous=json.loads(status_path.read_text())
        if previous.get('input_family') in ('frozen_reference','mambafusion_native'):
            write(OUT/previous['input_family']/'status.json',previous)
    write(status_path,status)
    if a.phase!='optimize':write(OUT/a.input/'status.json',status)
    try:
        resource={'source_repository':'https://github.com/altkddhfcjs/GRAE-3DMOT','commit':COMMIT,
            'source_acquisition':'official pinned-commit codeload archive (git transport timed out)',
            'source_files':{str(f.relative_to(SOURCE)):sha(f) for f in sorted(SOURCE.rglob('*')) if f.is_file()},
            'license':'No standalone LICENSE at pinned commit; author README License section credits CenterPoint/3DMOTFormer',
            'checkpoint_url':'https://drive.google.com/file/d/1r5IX9CdSXmJ7-MbCLsJCcTmDLAjfJ-08/view',
            'checkpoint_present':CHECKPOINT.exists(),'checkpoint_bytes':CHECKPOINT.stat().st_size if CHECKPOINT.exists() else None,
            'checkpoint_sha256':sha(CHECKPOINT) if CHECKPOINT.exists() else None,
            'dependency_simpletrack_commit':'05c96bb7ed98fc179856f327544612a66c839b5e','dependency_role':'NMS only, no old tracker execution',
            'environment':environment(),'adapter_sha256':sha(__file__)}
        write(OUT/'resource_manifest.json',resource)
        sequences=prepare(cfg,a.input)
        model=load_model('cpu' if a.phase=='prepare' else 'cuda:0')
        if a.phase=='prepare':
            status.update(status='partial',preparation='passed',strict_load='passed',gpu_executed=False,next='GPU smoke/run pending grant')
            write(status_path,status);return
        import torch
        status['gpu_executed']=True
        write(status_path,status)
        torch.backends.cudnn.benchmark=False;torch.manual_seed(0);np.random.seed(0)
        if a.phase=='optimize':
            frames=next(iter(sequences['scene0061'].values()))[:3];trials=[]
            for mode in [False,True]:
                for _ in range(3):run_sequence(model,frames,'cuda:0',mode)
            for repeat in range(30):
                outputs={};durations={}
                for mode in ([False,True] if repeat%2==0 else [True,False]):
                    torch.cuda.synchronize();start=time.perf_counter()
                    outputs[mode]=run_sequence(model,frames,'cuda:0',mode)
                    torch.cuda.synchronize();durations[mode]=(time.perf_counter()-start)*1000
                if outputs[False]!=outputs[True]:raise ValueError('Batched D2H changed tracks')
                trials.append({'round':repeat,'native_ms':durations[False],'batched_d2h_ms':durations[True],
                               'associations_and_outputs_equal':True,'order':'AB' if repeat%2==0 else 'BA'})
            write(OUT/a.input/'optimization.json',{'boundary':'prepared CPU detections through H2D, learned association, CPU output; excludes JSON read/NMS, same both sides',
                'frames_per_round':len(frames),'warmup':3,'paired_rounds':trials,'candidate':'batch final tensor D2H only, author association/state unchanged',
                'adoption':'native retained pending stability analysis'})
            status.update(status='partial',optimization='measured',adoption='native retained pending stability analysis')
            write(status_path,status)
            return
        calls={};handles=[]
        def observe(name):
            def hook(module,inputs,output):
                tensors=[]
                def visit(v):
                    if torch.is_tensor(v):tensors.append(v)
                    elif isinstance(v,(list,tuple)):
                        for x in v:visit(x)
                visit(output)
                if any(not torch.isfinite(t).all() for t in tensors):raise ValueError('Nonfinite '+name)
                record=calls.setdefault(name,{'calls':0,'output_shapes':[]});record['calls']+=1
                record['output_shapes']=[{'shape':list(t.shape),'dtype':str(t.dtype)} for t in tensors]
            return hook
        for name,module in model.named_modules():
            if name in ['spatial_dffl','temporal_dffl','spatial_proj','temporal_proj'] or name.startswith('affinity.') and name.count('.')==1:
                handles.append(module.register_forward_hook(observe(name)))
        first=next(iter(sequences['scene0061'].values()))
        with torch.inference_mode():
            smoke=run_sequence(model,first[:1],'cuda:0')
            samples=[first[i] for i in [0,len(first)//2,len(first)-1]]
            for f in samples:run_sequence(model,[f],'cuda:0')
            checks={'single_frame':len(smoke)==1,'first_middle_last':3}
            base=run_sequence(model,first[:3],'cuda:0',sentinel=-1)
            sentinel=run_sequence(model,first[:3],'cuda:0',sentinel=987654321)
            checks['sentinel_gt_id_invariance']=base==sentinel
            empty0,empty2=copy.deepcopy(first[0]),copy.deepcopy(first[2])
            for empty in [empty0,empty2]:
                for k,v in list(empty.items()):
                    if isinstance(v,np.ndarray):empty[k]=v[:0]
            empty_result=run_sequence(model,[empty0,first[1],empty2,first[3]],'cuda:0')
            checks['empty_detection_frames_execute']=empty_result[empty0['sample_token']]==[] and empty_result[empty2['sample_token']]==[]
            # Resume by replaying frozen history, avoiding invented author state serialization.
            prefix=run_sequence(model,first[:2],'cuda:0')
            resumed=run_sequence(model,first[:3],'cuda:0')
            checks['prefix_replay_resume_equal']=resumed==base and all(prefix[k]==base[k] for k in prefix)
            checks['resume_protocol']='replay fixed prefix from scene reset; no hidden state format change'
            if not all(v for k,v in checks.items() if isinstance(v,bool)):raise ValueError('State check failed '+str(checks))
            write(OUT/a.input/'state_checks.json',checks)
            for split,groups in sequences.items():
                predictions={};timings=[];torch.cuda.reset_peak_memory_stats()
                for scene,frames in groups.items():
                    torch.cuda.synchronize();start=time.perf_counter()
                    result=run_sequence(model,frames,'cuda:0')
                    torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    for token,boxes in result.items():
                        for b in boxes:b['tracking_id']=scene+':'+b['tracking_id']
                    predictions.update(result)
                    timings.append({'scene_token':scene,'frames':len(frames),'association_e2e_ms':elapsed*1000,'ms_per_frame':elapsed*1000/len(frames),'scene_reset':True})
                folder=OUT/a.input/split
                write(folder/'tracking.json',{'meta':{'use_camera':True,'use_lidar':True,'use_radar':False,'use_map':False,'use_external':False},'results':predictions})
                write(folder/'performance.json',{'scope':'CPU prepared detections through H2D + author association + CPU export; hook-instrumented validation, not final benchmark',
                    'scenes':timings,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'frames':len(predictions)})
        for h in handles:h.remove()
        if any(calls.get(k,{}).get('calls',0)==0 for k in ['spatial_dffl','temporal_dffl','spatial_proj','temporal_proj','affinity.0']):raise ValueError('Required learned module zero calls')
        write(OUT/a.input/'module_calls.json',calls)
        write(OUT/a.input/'native_receipt.json',{
            'input_family':a.input,'checkpoint_sha256':sha(CHECKPOINT),
            'input_manifest_sha256':sha(OUT/a.input/'input_manifest.json'),
            'tracking_sha256':{split:sha(OUT/a.input/split/'tracking.json') for split in ['scene0061','mini_val']}})
        status.update(status='partial',gpu_executed=True,inference='passed',official_evaluation='pending',frames=120)
    except Exception as exc:
        status.update(status='blocked_resource' if isinstance(exc,(FileNotFoundError,ModuleNotFoundError)) else 'failed',error=type(exc).__name__+': '+str(exc))
        write(status_path,status)
        if a.phase!='optimize':write(OUT/a.input/'status.json',status)
        print(traceback.format_exc())
        if status['status']=='blocked_resource':return 2
        raise
    write(status_path,status)
    if a.phase!='optimize':write(OUT/a.input/'status.json',status)

if __name__=='__main__':sys.exit(main())
