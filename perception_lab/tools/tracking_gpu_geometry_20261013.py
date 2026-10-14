"""CPU contracts and real input packing for the frozen GRAE geometry experiment."""
import hashlib,json,sys,time,cProfile,pstats,io
from pathlib import Path
import numpy as np
import grae3dmot_2025 as grae
from tracking_gpu_inputs_20261013 import ROOT,REPORT,SPECS,paths,sha
sys.path.insert(0,str(grae.DEPENDENCY))
from mot_3d.data_protos import BBox
from mot_3d.preprocessing import nms as cpu_nms
from mot_3d.preprocessing.bbox_coarse_hash import BBoxCoarseFilter
from pyquaternion import Quaternion

def make_boxes(raw):
 chosen=[b for b in raw if b['detection_name'] in grae.CLASSES]
 boxes=[BBox(x=b['translation'][0],y=b['translation'][1],z=b['translation'][2],w=b['size'][0],l=b['size'][1],h=b['size'][2],o=Quaternion(b['rotation']).yaw_pitch_roll[0]) for b in chosen]
 for box,b in zip(boxes,chosen):box.s=b['detection_score']
 return boxes,np.asarray([grae.CLASSES.index(b['detection_name']) for b in chosen],np.int64)

def pack(boxes,classes):
 """Keep CPU float64 corners, exact original argsort and coarse-set membership."""
 n=len(boxes);geo=np.empty((n,12),np.float64)
 for i,b in enumerate(boxes):
  geo[i,:8]=np.asarray(BBox.box2corners2d(b))[:,:2].reshape(-1)
  geo[i,8:]=[b.z,b.h,b.w,b.l]
 order=np.argsort(np.asarray([b.s for b in boxes]))[::-1].astype(np.int64).copy()
 coarse=BBoxCoarseFilter(grid_size=100,scaler=100);coarse.bboxes2dict(boxes)
 eligible=np.zeros((n,n),np.int32)
 for i,b in enumerate(boxes):
  ids=coarse.related_bboxes(b);eligible[i,ids]=1
 eligible*=np.asarray(classes)[:,None]==np.asarray(classes)[None,:]
 return geo,order,eligible

def array_hash(arrays):
 h=hashlib.sha256()
 for key in sorted(arrays):
  a=arrays[key]
  if isinstance(a,np.ndarray):h.update(key.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
 return h.hexdigest()

def profile():
 import importlib
 geometry=importlib.import_module('mot_3d.utils.geometry');utils=importlib.import_module('mot_3d.utils')
 rows=[];timing={};original_iou=utils.iou3d
 def timed_iou(*args):
  t=time.perf_counter();value=original_iou(*args);timing['iou_calls']=timing.get('iou_calls',0)+1;timing['iou_ms']=timing.get('iou_ms',0)+(time.perf_counter()-t)*1000;return value
 dependency={name:sha(grae.DEPENDENCY/name) for name in ['mot_3d/preprocessing/nms.py','mot_3d/preprocessing/bbox_coarse_hash.py','mot_3d/utils/geometry.py','mot_3d/data_protos/bbox.py']}
 profiler=cProfile.Profile();read_ms=0
 for split in SPECS:
  det,packets_path=paths(split);start=time.perf_counter();raw=json.loads(det.read_text())['results'];packets=json.loads(packets_path.read_text());read_ms+=(time.perf_counter()-start)*1000
  for index,p in enumerate(packets):
   boxes=raw[p['sample_token']];start=time.perf_counter();baseline=grae.frame_arrays(boxes,p);plain_ms=(time.perf_counter()-start)*1000
   timing.clear();utils.iou3d=timed_iou
   try:
    start=time.perf_counter();profiler.enable();candidate=grae.frame_arrays(boxes,p);profiler.disable();instrumented_ms=(time.perf_counter()-start)*1000
   finally:utils.iou3d=original_iou;profiler.disable()
   if array_hash(baseline)!=array_hash(candidate):raise ValueError('Instrumentation changed input')
   b,classes=make_boxes(boxes);start=time.perf_counter();geo,order,eligible=pack(b,classes);pack_ms=(time.perf_counter()-start)*1000
   rows.append(dict(split=split,index=index,sample_token=p['sample_token'],scene_token=p['scene_token'],raw_boxes=len(boxes),allowlist_boxes=len(b),output_boxes=len(baseline['score']),plain_reader_ms=plain_ms,instrumented_reader_ms=instrumented_ms,original_iou_calls=timing.get('iou_calls',0),original_iou_ms=timing.get('iou_ms',0),cpu_geometry_pack_ms=pack_ms,eligible_pairs=int(eligible.sum()),input_array_sha256=array_hash(baseline)))
 report=dict(status='passed',frames=len(rows),json_and_packets_read_ms=read_ms,rows=rows,dependency_sha256=dependency,scope='CPU native reader and cProfile diagnostic; instrumented vs plain adjacent order, not paired performance acceptance; IoU nested timing includes polygon+height',instrumentation_preserves_all_frame_arrays=True)
 (REPORT/'cpu_profile.json').write_text(json.dumps(report,indent=2)+'\n');stream=io.StringIO();pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(60);(REPORT/'cpu_cprofile.txt').write_text(stream.getvalue())
 print('CPU profile frames:',len(rows))
if __name__=='__main__':profile()
