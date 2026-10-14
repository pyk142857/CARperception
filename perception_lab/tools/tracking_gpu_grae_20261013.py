"""Run unchanged GRAE with validated same-process TileLang NMS preprocessing."""
import torch
torch.cuda.init()
import sys,json,time,hashlib
from pathlib import Path
from tracking_gpu_kernel_20261013 import GeometryNMS,ROOT,REPORT
import grae3dmot_2025 as grae
sys.path.insert(0,str(grae.DEPENDENCY))
import mot_3d.preprocessing as preprocessing

def main():
 check=json.loads((REPORT/'kernel_correctness.json').read_text())
 if check['status']!='passed' or check['frames']!=120:raise ValueError('Full120 exact NMS/input validation required')
 candidate=GeometryNMS();preprocessing.nms=candidate.nms
 sys.argv=['grae3dmot_2025.py','--phase','run','--input','frozen_reference','--config',str(ROOT/'configs/tracking_gpu_20261013.json'),'--output-dir','reports/tracking_gpu_20261013/grae_gpu']
 result=grae.main()
 if result not in (None,0):raise RuntimeError('GRAE GPU inference failed')
 comparisons=[]
 for split in ['scene0061','mini_val']:
  new=REPORT/'grae_gpu/frozen_reference'/split/'tracking.json';old=REPORT/'native_compare/grae/frozen_reference'/split/'tracking.json'
  comparisons.append({'split':split,'bytes_exact':new.read_bytes()==old.read_bytes(),'cpu_sha256':grae.sha(old),'gpu_sha256':grae.sha(new)})
 if not all(x['bytes_exact'] for x in comparisons):raise ValueError('GPU GRAE track fields/order/ID differs from CPU reference')
 grae.write(REPORT/'grae_gpu/equivalence.json',{'status':'passed','tracks':comparisons,'kernel_calls':candidate.calls,'cpu_boundary_pairs':candidate.boundary_pairs,'nms_frames':candidate.frames,'adapter_sha256':grae.sha(__file__),'scope':'same original GRAE environment/model/association; GPU replaces NMS geometry and greedy index selection only'})
 print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
