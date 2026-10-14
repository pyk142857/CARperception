#!/usr/bin/env python3
"""GT-free mini reader for pinned OptiPMB; author CenterPoint defaults unchanged.

Run with envs/optipmb_20261013/bin/python. No training or CUDA. Online assembly
follows Run_OptiPMB.py:225-302; no offline/smoothed estimates are exported.
"""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse, ast, collections, hashlib, importlib.metadata, json, platform, sys, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[2]
LAB = ROOT/'perception_lab'
SOURCE = LAB/'third_party/OptiPMB'
OUT = LAB/'reports/tracking_gpu_20261013/native_compare/optipmb'
CLASSES = ['bicycle','bus','car','motorcycle','pedestrian','trailer','truck']
EXPECTED = {'scene0061':'dd6df55652026f2b04c7b18f9adde44123b664e8cb3706e53d6d30faca344d2f', 'mini_val':'a6fb424ccbe629f3af27172202960c4533cd48ddd887cd201f0d464ee22c56a7'}
sys.path.insert(0,str(SOURCE))
from trackers.OptiPMB import OptiPMB_Tracker as native
from trackers.OptiPMB.utils import gen_OptiPMB_filter_model
from trackers.utils import mhtdaClink as murty


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)
def blobhash(x): return hashlib.sha256(json.dumps(x,default=plain,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,default=plain,indent=2,allow_nan=False)+'\n')

# Execute precisely the author's formatter AST, excluding module-level evaluation
# imports. This function has only numpy as a global dependency, and no GT reads.
_tree=ast.parse((SOURCE/'utils.py').read_text())
_node=next(n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name=='format_measurements')
_namespace={'np':np}
exec(compile(ast.Module(body=[_node],type_ignores=[]),str(SOURCE/'utils.py'),'exec'),_namespace)
format_measurements=_namespace['format_measurements']
CONFIG=SOURCE/'trackers/OptiPMB/nuscenes_centerpoint_config_optipmb.json'


def lidar(packet):
    sensor=packet['sensors']['LIDAR_TOP']
    path=LAB/'data/nuscenes'/sensor['relative_path']
    raw=np.fromfile(path,dtype=np.float32).reshape(-1,5)
    xyz=raw[:,:3].T
    transform=np.asarray(sensor['T_ego_to_global'])@np.asarray(sensor['T_sensor_to_ego'])
    global_xyz=(transform@np.vstack([xyz,np.ones((1,len(raw)))]))[:3]
    return {'position':xyz,'position_global':global_xyz,'remission':raw[:,3]}, path, transform


def measurement(boxes,cloud):
    # Author preprocessing includes Euler xyz (9 columns), with yaw in the last
    # column. Preserve that exact layout; the filter consumes [h,w,l,x,y,z,...,yaw].
    hwlxyzangles=[];quats=[];vels=[];names=[];scores=[]
    for b in boxes:
        if b['detection_name'] not in CLASSES: continue
        q=b['rotation']
        euler=Rotation.from_quat(q[1:]+q[:1]).as_euler('xyz',degrees=False)
        hwlxyzangles.append(list(np.asarray(b['size'])[[2,0,1]])+b['translation']+euler.tolist())
        quats.append(q);vels.append(b['velocity']);names.append(b['detection_name']);scores.append(b['detection_score'])
    return format_measurements(cloud,hwlxyzangles,quats,vels,names,scores,CLASSES)


def initial_state():
    return dict(unknownPPP=[],tracks=[],MBM_max_ID=0,global_hypo=[],global_hypo_weight=[],num_meas_last_frame=0)


def online_boxes(estimate,token,labels):
    rows=[]
    for i,label in enumerate(estimate['id']):
        if label not in labels: labels.append(label)
        target_id=labels.index(label)
        mean=estimate['mean'][i];aux=estimate['aux_info'][i];hwla=estimate['hwla'][i]
        speed=mean[2,-1];angle=mean[3,-1]
        rows.append({'sample_token':token,'translation':[mean[0,-1],mean[1,-1],aux[4]],
          'velocity':[speed*np.cos(angle),speed*np.sin(angle)],'tracking_id':aux[2]+'_'+str(target_id),
          'tracking_name':aux[2],'tracking_score':aux[-1], 'size':[hwla[1],hwla[2],hwla[0]],'rotation':hwla[3:7]})
    return rows


class Evidence:
    """Diagnostic-only source line/call trace; never active in performance runs."""
    def __init__(self): self.calls=collections.Counter();self.lines=collections.Counter()
    def trace(self,frame,event,arg):
        filename=frame.f_code.co_filename
        if filename == native.__file__:
            if event=='call': self.calls[frame.f_code.co_name]+=1
            elif event=='line' and frame.f_lineno in (28,101,112,447,450,596,599,825): self.lines[str(frame.f_lineno)]+=1
            return self.trace
        if filename == murty.__file__ and event=='call': self.calls[frame.f_code.co_name]+=1
        return None


def run(packets,detections,diagnostic=False,empty_at=None):
    config=json.loads(CONFIG.read_text());model=gen_OptiPMB_filter_model(config,CLASSES)
    results={};records=[];active=None;trace=Evidence();scene_resets=[]
    if diagnostic: sys.settrace(trace.trace)
    start=time.perf_counter()
    try:
      for p in packets:
        begun=time.perf_counter();token=p['sample_token'];stamp=p['sensors']['LIDAR_TOP']['timestamp_us']
        if p['scene_token'] != active:
            active=p['scene_token'];tracker=native.OptiPMB_Filter(model);state=initial_state();tracks={};labels=[];index=0;last=stamp
            scene_resets.append(token)
        dt=(stamp-last)/1e6;last=stamp
        cloud,path,transform=lidar(p)
        z=measurement([] if token==empty_at else detections[token],cloud)
        pre=tracker.predict(z,dt,index,filter_pruned=state)
        state=tracker.update(state,pre,index)
        tracker.prune(state)
        estimate,tracks=tracker.extract_states(index,state,tracks)
        results[token]=online_boxes(estimate,token,labels)
        elapsed=time.perf_counter()-begun
        row={'sample_token':token,'scene_token':active,'scene_name':p['scene_name'],'frame_index':index,'timestamp_us':stamp,'dt_seconds':dt,'total_ms':elapsed*1000,'raw_boxes':len(detections[token]),'allowlist_boxes':len(z['bbox_score']),'after_score':int(sum(s>model[c]['keep_score_thld'] for s,c in zip(z['bbox_score'],z['bbox_class']))),'after_nms':len(pre['used_bbox_score']),'tracks':len(results[token])}
        if diagnostic:
            inverse=(np.linalg.inv(transform)@np.vstack([cloud['position_global'],np.ones((1,cloud['position'].shape[1]))]))[:3]
            row.update(lidar_path=str(path.relative_to(ROOT)),lidar_sha256=sha(path),lidar_points=cloud['position'].shape[1],after_3m_points=pre['lidar_meas'].shape[1],transform_sha256=blobhash(transform),global_points_sha256=hashlib.sha256(cloud['position_global'].tobytes()).hexdigest(),remission_sha256=hashlib.sha256(cloud['remission'].tobytes()).hexdigest(),roundtrip_max_abs_m=float(np.max(np.abs(inverse-cloud['position']))),used_bbox_pt_num=pre['used_bbox_pt_num'].tolist(),measurement_sha256=blobhash(z),post_nms_sha256=blobhash({k:v for k,v in pre.items() if k!='lidar_meas'}))
        records.append(row);index+=1
        if not diagnostic and index%10==0: print(p['scene_name'],index,round(elapsed,3),'s',flush=True)
    finally:
        if diagnostic: sys.settrace(None)
    return results,{'frames':records,'scene_reset_tokens':scene_resets,'loop_seconds':time.perf_counter()-start,'calls':dict(trace.calls),'source_line_hits':dict(trace.lines)}


def known_murty():
    costs=np.array([[-9.,-1.],[-2.,-8.]])
    assoc=np.zeros((1,4,2),dtype=np.int32);out=np.zeros(1);work=murty.allocateWorkvarsforDA(2,2,1)
    try: error=murty.mhtda(murty.sparsifyByRow(costs,2),np.ones((1,2),dtype=np.bool_),np.zeros(1),np.ones((1,2),dtype=np.bool_),np.zeros(1),assoc,out,work)
    finally: murty.deallocateWorkvarsforDA(work)
    pairs=assoc[0][(assoc[0,:,0]>=0)&(assoc[0,:,1]>=0)]
    assert error==0 and out[0]==-17 and sorted(pairs.tolist())==[[0,0],[1,1]]
    return dict(input=costs,output=pairs,cost=out,error=error,library_path=murty.dll_path,library_sha256=sha(murty.dll_path))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['scene0061','mini_val','all'],default='all');parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/'fastmurty_verification.json',known_murty())
    for split in (EXPECTED if args.split=='all' else [args.split]):
        prediction=LAB/f'reports/deployment_4090_20261008/bevfusion_reference/{split}/evaluation/official_predictions.json'
        packetpath=LAB/f'reports/deployment_4090_20261008/inputs/{split}_packets.json'
        assert sha(prediction)==EXPECTED[split]
        begun=time.perf_counter();detections=json.loads(prediction.read_text())['results'];packets=json.loads(packetpath.read_text());readtime=time.perf_counter()-begun
        results,evidence=run(packets,detections,args.diagnostic)
        evidence.update(json_and_packet_read_seconds=readtime,end_to_end_seconds=readtime+evidence['loop_seconds'],prediction_sha256=sha(prediction),packet_sha256=sha(packetpath),online_output_sha256=blobhash(results),timing_includes='JSON, packets, LiDAR loading/global transform, format, author filter, online result construction; excludes import/config load, evidence hashing and disk export',instrumented=args.diagnostic)
        assert sha(prediction)==EXPECTED[split] and set(results)==set(detections)
        if args.diagnostic:
            write(OUT/split/'diagnostics.json',evidence)
            timed=json.loads((OUT/split/'tracking.json').read_text())['results']
            assert blobhash(results)==blobhash(timed),'instrumented replay changed predictions'
        else:
            write(OUT/split/'tracking.json',{'meta':{'use_camera':True,'use_lidar':True,'use_radar':False,'use_map':False,'use_external':False},'results':results})
            write(OUT/split/'timing.json',evidence)
        if args.validate:
            checks=[]
            for scene in dict.fromkeys(p['scene_token'] for p in packets):
                subset=[p for p in packets if p['scene_token']==scene]
                prefix,proof=run(subset[:3],detections,True)
                assert all(blobhash(prefix[t])==blobhash(results[t]) for t in prefix)
                empty,ep=run(subset[:2],detections,True,empty_at=subset[1]['sample_token'])
                cold,cp=run(subset[:1],detections,True,empty_at=subset[0]['sample_token'])
                assert cold[subset[0]['sample_token']]==[]
                checks.append({'scene_token':scene,'prefix_tokens':list(prefix),'prefix_equal_to_full':True,'isolated_scene_reset_equal':True,'prefix_call_proof':proof,'empty_after_live':ep,'empty_first':cp,'first_middle_last':[subset[i]['sample_token'] for i in [0,len(subset)//2,len(subset)-1]]})
            write(OUT/split/'online_validation.json',checks)
    files=['Run_OptiPMB.py','preprocess_data.py','utils.py','trackers/OptiPMB/OptiPMB_Tracker.py','trackers/OptiPMB/utils.py','trackers/OptiPMB/nuscenes_centerpoint_config_optipmb.json','trackers/utils/mhtdaClink.py','trackers/utils/murty_linux.so']
    write(OUT/'source_manifest.json',{'repository':'https://github.com/dinggh0817/OptiPMB','commit':'ed0e0467efb08fd8a96b03647b09169564a74acd','source_files':{f:sha(SOURCE/f) for f in files},'adapter_sha256':sha(__file__),'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':{k:importlib.metadata.version(k) for k in ['numpy','scipy','scikit-learn','powerboxes','cholupdates','opencv-python']},'config_label':'Author CenterPoint default applied unchanged to frozen BEVFusion detections','config':json.loads(CONFIG.read_text()),'weights':'none','GT_reader':'no GT tables, annotation boxes or instance IDs loaded; packets projected to sensor/sample/scene fields','source_format_measurements_ast_sha256':blobhash(ast.dump(_node))})

if __name__=='__main__': main()
