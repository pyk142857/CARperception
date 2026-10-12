"""Compare captured real S1/S2 pre-downsample BEV pooling outputs on CPU."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--frame',type=int,default=0);a=p.parse_args();run=ROOT/'runs/deployment_4090_20261008';refpath=run/'pooling_captures'/f'pool_capture_{a.frame}.pt';prefix=run/'trt_pooling_captures'/f'frame_{a.frame}';meta=json.loads(prefix.with_suffix('.json').read_text());ref=torch.load(refpath,map_location='cpu');raw=ref['output'].detach().numpy();candidate=np.fromfile(prefix.with_suffix('.output.bin'),dtype=np.float16).astype(np.float32).reshape(1,meta['shape'][1],meta['out_h'],meta['out_w']);reference=np.concatenate([raw[:,:,i] for i in range(raw.shape[2])],axis=1);assert reference.shape==candidate.shape
 reference_inputs=json.loads((ROOT/'reports/deployment_4090_20261008/bevfusion_reference/scene0061/actual_inputs.json').read_text())[a.frame];trt_inputs=json.loads((ROOT/'outputs/bevfusion_tensorrt/scene0061/actual_inputs.json').read_text())[a.frame]
 delta=np.abs(reference-candidate);relative=delta/(np.abs(reference)+1e-6);cfg=json.loads((ROOT/'configs/deployment_4090.yaml').read_text());budget=cfg['tensorrt']['acceptance_fixed_before_run'];tol=budget['tensor_atol']+budget['tensor_rtol']*np.abs(reference);significant=np.abs(reference)>.01
 result={'scope':'real first-frame camera pre-downsample pooled BEV tensor; includes upstream camera/depth FP16 and geometry effects, not isolated pooling precision','frame':a.frame,'reference_sha256':sha(refpath),'candidate_capture_sha256':sha(prefix.with_suffix('.output.bin')),'reference_shape':list(raw.shape),'candidate_shape':list(candidate.shape),'layout_alignment':'PyTorch B,C,D,X,Y with D=1 collapsed by concatenating depth -> CUDA N,C,X,Y; no spatial transpose','elements':int(delta.size),'nan_or_inf':int((~np.isfinite(reference)).sum()+(~np.isfinite(candidate)).sum()),'absolute_error':{name:float(value) for name,value in [('mean',delta.mean()),('p50',np.percentile(delta,50)),('p95',np.percentile(delta,95)),('p99',np.percentile(delta,99)),('max',delta.max())]},'reference_abs_p99':float(np.percentile(np.abs(reference),99)),'candidate_abs_p99':float(np.percentile(np.abs(candidate),99)),'relative_error_on_reference_abs_gt001_p95':float(np.percentile(relative[significant],95)),'normalized_rmse':float(np.sqrt(np.mean((reference-candidate)**2))/np.sqrt(np.mean(reference**2))),'cosine_similarity':float(np.vdot(reference.ravel(),candidate.ravel())/(np.linalg.norm(reference)*np.linalg.norm(candidate))),'fixed_tolerance':{'atol':budget['tensor_atol'],'rtol':budget['tensor_rtol'],'exceeded_elements':int((delta>tol).sum()),'passed_all_elements':bool((delta<=tol).all())},'reference_inputs':reference_inputs,'candidate_inputs':trt_inputs}
 # A transposition control guards against silently choosing axes based on lower error.
 intervals=np.fromfile(prefix.with_suffix('.intervals.bin'),dtype=np.int32).reshape(-1,3)
 result['reference_valid_feature_rows']=int(ref['features'].shape[0]);result['candidate_valid_feature_rows']=int((intervals[:,1]-intervals[:,0]).sum());support_ref=np.any(reference!=0,axis=1);support_cand=np.any(candidate!=0,axis=1);result['support_cells']={'reference':int(support_ref.sum()),'candidate':int(support_cand.sum()),'xor':int((support_ref!=support_cand).sum())}
 result['upstream_last_interval_audit']={'last_interval':intervals[-1].tolist(),'numel_geometry':prefix.with_suffix('.indices.bin').stat().st_size//4,'exclusive_end_equals_numel_geometry':int(intervals[-1,1])==prefix.with_suffix('.indices.bin').stat().st_size//4}
 result['transpose_control_mean_abs']=float(np.abs(reference-candidate.transpose(0,1,3,2)).mean())
 out=ROOT/'reports/deployment_4090_20261008/bevfusion_tensorrt';(out/f'pool_intermediate_frame_{a.frame}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
