"""Project KernelBench task extension: typed correctness and cold-L2 diagnostics.

No upstream modification; not a standard KernelBench 250-task score.
CPU-only adapter regression: --cpu-only. Real kernels: --gpu.
"""
import argparse, hashlib, importlib.util, json
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/tracking_gpu_20261013/kernelbench'
KB = ROOT / 'third_party/KernelBench_20261013/src/kernelbench'


def process_input_tensor(value, device):
    """Preserve every declared dtype; move only the device."""
    return value.to(device=device) if isinstance(value, torch.Tensor) else value


def compare_output(actual, reference):
    if actual.shape != reference.shape or actual.dtype != reference.dtype:
        raise AssertionError('Output shape/dtype mismatch')
    actual, reference = actual.detach().cpu(), reference.detach().cpu()
    if not actual.is_floating_point():
        if not torch.equal(actual, reference):
            raise AssertionError('Integer/bool output differs (including order)')
        return {'comparison': 'exact', 'dtype': str(actual.dtype)}
    if actual.dtype not in (torch.float64, torch.float32):
        raise AssertionError('Only declared FP64/FP32 geometry budgets supported')
    atol, rtol = (1e-8, 1e-7) if actual.dtype == torch.float64 else (1e-5, 1e-5)
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError('Nonfinite geometry output')
    error = (actual-reference).abs()
    if not torch.all(error <= atol + rtol*reference.abs()):
        raise AssertionError('Geometry tolerance exceeded')
    return {'comparison': 'elementwise_atol_plus_rtol', 'dtype': str(actual.dtype),
            'atol': atol, 'rtol': rtol, 'max_abs': error.max().item() if error.numel() else 0.,
            'max_relative': (error/reference.abs().clamp_min(1e-30)).max().item() if error.numel() else 0.}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensor_manifest(value):
    return {'shape': list(value.shape), 'dtype': str(value.dtype),
            'sha256': hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()}


def cpu_checks():
    values = [torch.tensor([1.+2**-30], dtype=torch.float64),
              torch.tensor([2**30+1], dtype=torch.int32),
              torch.tensor([2**53+1], dtype=torch.int64)]
    for value in values:
        result = process_input_tensor(value, 'cpu')
        assert result.dtype == value.dtype and torch.equal(result, value)
    assert process_input_tensor('metadata', 'cpu') == 'metadata'
    rejected = []
    for name, actual, reference in [
        ('integer_off_by_one', torch.tensor([2**53+2]), values[2]),
        ('integer_order', torch.tensor([2,1]), torch.tensor([1,2])),
        ('dtype_cast', values[0].float(), values[0]),
        ('fp64_outside_budget', torch.tensor([1.0001], dtype=torch.float64), torch.ones(1, dtype=torch.float64)),
        ('nonfinite', torch.tensor([float('nan')]), torch.tensor([0.]))]:
        try: compare_output(actual, reference)
        except AssertionError: rejected.append(name)
        else: raise AssertionError('Invalid output accepted: '+name)
    assert compare_output(torch.tensor([1.+5e-9], dtype=torch.float64), torch.ones(1,dtype=torch.float64))
    return {'status':'passed','preserved_dtypes':[str(v.dtype) for v in values],
            'invalid_output_rejections':rejected,'preserves_non_tensor':True,
            'blanket_fp32_cast_loses_values':[not torch.equal(v.float().to(v.dtype),v) for v in values]}


def gpu_checks(trials):
    # Initialize CUDA before the legacy Numba/SimpleTrack dependency graph.
    torch.cuda.init()
    import numpy as np
    from tracking_gpu_kernel_20261013 import GeometryNMS
    from tracking_gpu_geometry_20261013 import pack, make_boxes, cpu_nms
    from tracking_gpu_inputs_20261013 import paths, SPECS
    from mot_3d.utils import iou3d
    spec=importlib.util.spec_from_file_location('pinned_kernelbench_timing',KB/'timing.py')
    timing=importlib.util.module_from_spec(spec);spec.loader.exec_module(timing)
    candidate=GeometryNMS();rows=[]
    for split in SPECS:
        detections, packet_path=paths(split)
        raw=json.loads(detections.read_text())['results'];packets=json.loads(packet_path.read_text())
        scenes={}
        for p in packets: scenes.setdefault(p['scene_token'],[]).append(p)
        for scene, ps in scenes.items():
            for position in [0,len(ps)//2,len(ps)-1]:
                packet=ps[position];boxes,classes=make_boxes(raw[packet['sample_token']]);n=len(boxes)
                geometry,order,eligible=pack(boxes,classes)
                g=np.zeros((256,12),np.float64);g[:n]=geometry
                e=np.zeros((256,256),np.int32);e[:n,:n]=eligible
                o=np.zeros(256,np.int64);o[:n]=order
                host=[torch.from_numpy(g),torch.from_numpy(e),torch.tensor([n],dtype=torch.int32),torch.from_numpy(o)]
                G,E,N,Order=[process_input_tensor(x,'cuda') for x in host]
                matrix=torch.empty((256,256),dtype=torch.float64,device='cuda')
                keep=torch.empty(257,dtype=torch.int64,device='cuda')
                def resident():
                    candidate.launch('overlap',G,E,N,matrix)
                    candidate.launch('suppress',matrix,Order,N,keep)
                resident();torch.cuda.synchronize()
                ref=np.zeros((n,n),np.float64)
                for i,j in np.argwhere(eligible):ref[i,j]=iou3d(boxes[i],boxes[j])[1]
                geometry_check=compare_output(matrix[:n,:n],torch.from_numpy(ref))
                ref_indices,ref_types=cpu_nms(boxes,classes,.1)
                actual=keep.cpu();indices=actual[:int(actual[256])]
                indices_check=compare_output(indices,torch.tensor(ref_indices,dtype=torch.int64))
                types_check=compare_output(torch.tensor([int(classes[i]) for i in indices],dtype=torch.int64),torch.tensor(ref_types,dtype=torch.int64))
                # Separate integrated candidate includes its documented boundary CPU checks.
                integrated=candidate.nms(boxes,classes)
                assert integrated==(ref_indices,ref_types)
                before=dict(candidate.calls)
                times=timing.time_execution_with_cuda_event(resident,[],num_warmup=2,num_trials=trials,discard_first=0,verbose=False)
                rows.append({'split':split,'scene_token':scene,'sample_token':packet['sample_token'],'n':n,
                             'inputs':[tensor_manifest(x) for x in host], 'geometry_check':geometry_check,
                             'indices_check':indices_check,'types_check':types_check,'integrated_nms_exact':True,
                             'output_indices':tensor_manifest(indices),'output_geometry':tensor_manifest(matrix[:n,:n]),
                             'cold_l2_resident_ms':times,'timed_launches':{k:candidate.calls[k]-before[k] for k in before}})
    return {'status':'passed','representative_frames':len(rows),'rows':rows,'actual_kernel_calls':candidate.calls,
            'timing_scope':'KernelBench cold-L2 device-resident overlap+suppress diagnostic; excludes transfers, CPU packing/boundary checks; not warm AB/BA acceptance',
            'kernelbench_timing_sha256':sha(KB/'timing.py')}


def main():
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--cpu-only',action='store_true');mode.add_argument('--gpu',action='store_true')
    parser.add_argument('--trials',type=int,default=3);args=parser.parse_args()
    if args.trials<1:parser.error('--trials must be positive')
    result={'scope':'project task extension, not standard KernelBench score','adapter_source_sha256':sha(__file__),
            'kernelbench_commit':'423217d9fda91e0c2d67e4a43bf62f96f6d104f1',
            'kernelbench_eval_sha256':sha(KB/'eval.py'),'cpu_adapter_tests':cpu_checks()}
    if args.gpu:result['gpu']=gpu_checks(args.trials)
    OUT.mkdir(parents=True,exist_ok=True)
    filename='typed_gpu_validation.json' if args.gpu else 'typed_cpu_validation.json'
    (OUT/filename).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
