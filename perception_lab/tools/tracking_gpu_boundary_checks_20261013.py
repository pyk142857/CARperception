"""Reader rejection and explicitly isolated upstream nontermination fixtures."""
import sys
if '--reference-tiny' in sys.argv:
 from tracking_gpu_geometry_20261013 import BBox,cpu_nms
 b=BBox(x=0,y=0,z=0,w=1e-4,l=1e-4,h=1e-4,o=0);b.s=.9
 print('REFERENCE_ENTERED',flush=True);cpu_nms([b],[0],.1)
 raise SystemExit('Unexpected reference completion')
import torch
torch.cuda.init()
import copy,json,subprocess,time
import numpy as np
from tracking_gpu_geometry_20261013 import BBox,cpu_nms,grae
from tracking_gpu_kernel_20261013 import GeometryNMS,REPORT
from tracking_gpu_inputs_20261013 import paths,sha
from mot_3d.utils import iou3d

def main():
 dp,pp=paths('scene0061');packet=json.loads(pp.read_text())[0];base=next(b for b in json.loads(dp.read_text())['results'][packet['sample_token']] if b['detection_name'] in grae.CLASSES)
 rows=[];candidate=GeometryNMS()
 for name,field,idx,value in [('zero_size','size',0,0),('negative_size','size',2,-1),('nan_translation','translation',0,float('nan')),('inf_velocity','velocity',0,float('inf')),('nan_score','detection_score',None,float('nan')),('nonunit_quaternion','rotation',0,2)]:
  raw=copy.deepcopy(base)
  if idx is None:raw[field]=value
  else:raw[field][idx]=value
  # original entry rejects before NMS and before any GPU replacement invocation.
  try:grae.frame_arrays([raw],packet)
  except ValueError as exc:rows.append(dict(name=name,status='rejected_at_unchanged_reader',exception=str(exc)))
  else:raise ValueError('Reader accepted invalid '+name)
 b=BBox(x=0,y=0,z=0,w=1e-4,l=1e-4,h=1e-4,o=0);b.s=.9
 proc=subprocess.Popen([sys.executable,__file__,'--reference-tiny'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  out,err=proc.communicate(timeout=8)
  raise RuntimeError('Expected reference nontermination; '+out+err)
 except subprocess.TimeoutExpired:
  proc.kill();out,err=proc.communicate()
  if 'REFERENCE_ENTERED' not in out:raise RuntimeError('Timeout occurred before actual reference entry: '+out+err)
  failure=dict(name='tiny_positive_self_iou',status='reference_failure',kind='timeout_after_actual_nms_entry',timeout_seconds=8,self_iou=iou3d(b,b)[1],stdout=out,stderr=err)
 try:candidate.nms([b],np.array([0],np.int64))
 except ValueError as exc:failure['candidate']='explicit_rejection: '+str(exc)
 else:raise ValueError('Tiny self IoU not rejected')
 votes=[]
 for yaw in [.3-1e-8,.3,.3+1e-8]:
  boxes=[BBox(x=0,y=0,z=0,w=2,l=4,h=h,o=yaw if i else 0) for i,h in enumerate([10,1,1])]
  for b in boxes:b.s=.9
  before=[vars(b).copy() for b in boxes];a=cpu_nms(boxes,[0,0,0],.1);b=candidate.nms(boxes,[0,0,0])
  if a!=b or before!=[vars(b).copy() for b in boxes]:raise ValueError('High/yaw fixture changed visible outputs/boxes')
  votes.append(dict(yaw=yaw,indices=list(map(int,a[0])),types=list(map(int,a[1])),unchanged_input_boxes=True,max_pair_iou=max(iou3d(x,y)[1] for x in boxes for y in boxes)))
 result=dict(status='passed',invalid_reader_cases=rows,reference_failures=[failure],default_high_yaw_cases=votes,high_branch_analysis='For positive finite rectangles and exact arithmetic, overlap_area<=min(area_a,area_b) and overlap_height<=(ha+hb)/2. Hence 2*intersection_volume<=volume_a+volume_b and IoU<1 with +1e-5. Default high=1 voting is unreachable on ordinary valid inputs; near yaw=.3 is explicitly tested. Original voting constructs a new local averaged BBox, never mutates inputs/returned index selection. Floating pathologies remain outside guaranteed valid domain.',actual_gpu_calls=candidate.calls,source_sha256=sha(__file__))
 (REPORT/'kernel_boundary_checks.json').write_text(json.dumps(result,indent=2)+'\n');print('boundary checks passed')
if __name__=='__main__':main()
