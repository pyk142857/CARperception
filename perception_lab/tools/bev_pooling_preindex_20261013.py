#!/usr/bin/env python3
"""One pooling candidate: precompute feature offsets, with cost in every forward.

Default analyze stage is CPU-only. benchmark must be scheduled after B–E
publication, with exclusive GPU access. No existing upstream files are changed.
"""
import argparse, collections, ctypes, csv, hashlib, json, os, platform, subprocess, time
from pathlib import Path
import numpy as np
from bev_pooling_4090 import load_capture, validate_case, edge_cases, Pool, compare, sha256
LAB=Path(__file__).resolve().parents[1]
OUT=LAB/'reports/tracking_gpu_20261013/bev_pooling'
CAP=LAB/'runs/deployment_4090_20261008/trt_pooling_captures'
LIB=LAB/'runs/tracking_gpu_20261013/bev_pooling/libpreindex.so'
REF=LAB/'third_party/Lidar_AI_Solution/CUDA-BEVFusion/build/libbevfusion_core.so'
SOURCE=Path(__file__).with_suffix('.cu')

def write(name,data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')

def analyze():
    baseline=json.loads((LAB/'reports/deployment_4090_20261008/bev_pooling/input_statistics.json').read_text())
    byname={x['name']:x for x in baseline};cases=[]
    for i in [0,19,38]:
        case=load_capture(CAP/f'frame_{i}');validate_case(case)
        a=case['arrays'];m=case['metadata'];N,C,D,H,W=m['shape'];intervals=a['intervals']
        lengths=intervals[:,1]-intervals[:,0]
        assert np.all(intervals[1:,0]>=intervals[:-1,1])
        valid=np.concatenate([a['indices'][b:e] for b,e,_ in intervals])
        # Independent CPU index algebra and all active feature bounds.
        camera=valid//(D*H*W);pixel=valid%(H*W)
        offsets=(camera*(H*W)+pixel)*C
        assert np.all(offsets>=0) and np.all(offsets+C<=len(a['features']))
        assert np.array_equal(offsets,((valid//(D*H*W))*(H*W)+(valid%(H*W)))*C)
        assert case['sha256']==byname[case['name']]['hashes'],'capture has changed from corrected upstream record'
        lengths_sorted=np.sort(lengths)
        cases.append({'name':case['name'],'prefix':str((CAP/f'frame_{i}').relative_to(LAB.parent)),'metadata':m,'sha256':case['sha256'],
            'bytes':{k:int(v.nbytes) for k,v in a.items()},'valid_projection_count':int(len(valid)),
            'index_capacity':int(len(a['indices'])),'unused_index_slots':int(len(a['indices'])-len(valid)),
            'unique_feature_pixels':int(len(np.unique(offsets))),'active_output_cells':int(len(intervals)),
            'output_cells':int(m['out_h']*m['out_w']),
            'interval_lengths':{'min':int(lengths.min()),'max':int(lengths.max()),'mean':float(lengths.mean()),'p50':float(np.percentile(lengths,50)),'p95':float(np.percentile(lengths,95)),'p99':float(np.percentile(lengths,99)),'top_1percent_projection_fraction':float(lengths_sorted[-max(1,len(lengths)//100):].sum()/len(valid))},
            'offset_workspace_bytes':int(len(a['indices'])*4),
            'logical_original_divmod_pairs':int(len(valid)*(C//10)),
            'candidate_precompute_divmod_pairs':int(len(a['indices'])),
            'candidate_extra_logical_offset_reads_bytes':int(len(valid)*(C//10)*4),
            'candidate_extra_precompute_read_write_bytes':int(len(a['indices'])*8),
            'cpu_active_offset_bounds_passed':True,'capture_matches_corrected_historical_hashes':True})
    config=LAB/'configs/deployment_4090.yaml'
    budget=json.loads(config.read_text())['pooling']
    write('input_analysis.json',{'status':'cpu_analysis_passed_gpu_not_run','candidate':'feature_offset_precompute_each_forward',
        'cases':cases,'budget':budget,'config_sha256':sha256(config),
        'scope':'Existing actual CUDA-BEVFusion FP16 captures only; no FP32 captures or FP16 conversion invented.',
        'candidate_source_sha256':sha256(SOURCE),'candidate_library_sha256':sha256(LIB),
        'upstream_library_sha256':sha256(REF),'tool_sha256':sha256(__file__),
        'compile_command':'/usr/local/cuda-11.8/bin/nvcc -O3 -std=c++14 -arch=sm_89 -Xcompiler -fPIC --shared perception_lab/tools/bev_pooling_preindex_20261013.cu -o perception_lab/runs/tracking_gpu_20261013/bev_pooling/libpreindex.so -Xptxas=-v',
        'compiler':subprocess.check_output(['/usr/local/cuda-11.8/bin/nvcc','--version'],text=True),
        'GPU_execution':'not performed in analyze; CUDA compilation does not launch device kernels',
        'actual_hardware_counter_claim':'none; operation/byte estimates are logical source-level counts, not measured DRAM traffic'})
    print(json.dumps({'stage':'cpu_analysis','cases':len(cases),'GPU_executed':False}))


class Preindex(Pool):
    def __init__(self):
        super().__init__(REF)
        self.candidate_lib=ctypes.CDLL(str(LIB))
        self.candidate=self.candidate_lib.car_bev_pool_preindex_forward
        self.candidate.restype=ctypes.c_int
        self.candidate.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint]*8+[ctypes.c_void_p]*3
        self.calls=collections.Counter()
    def prepare(self,case):
        import torch
        super().prepare(case)
        self.offsets=torch.empty_like(self.arrays['indices'])
    def __call__(self,backend):
        import torch
        self.calls[str(backend)]+=1
        if backend==0: return super().__call__(0)
        x=self.arrays;m=self.metadata;_,C,D,H,W=m['shape']
        code=self.candidate(x['features'].data_ptr(),x['depth'].data_ptr(),x['indices'].data_ptr(),x['intervals'].data_ptr(),m['num_intervals'],C,D,H,W,m['out_h'],m['out_w'],x['indices'].numel(),self.offsets.data_ptr(),self.outputs[1].data_ptr(),torch.cuda.current_stream().cuda_stream)
        if code: raise RuntimeError('Candidate CUDA error '+str(code))
        return self.outputs[1]


def benchmark(rounds,warmup):
    import torch
    if rounds<30: raise ValueError('At least 30 alternating AB/BA pairs required')
    budget=json.loads((LAB/'configs/deployment_4090.yaml').read_text())['pooling']
    pool=Preindex();cases=list(edge_cases())+[load_capture(CAP/f'frame_{i}') for i in [0,19,38]]
    results=[];rows=[]
    processes_before=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True)
    for case in cases:
        pool.prepare(case);a=pool(0);b=pool(1);torch.cuda.synchronize()
        equality=compare(b,a,budget['fp16_atol'],budget['fp16_rtol'])
        captured=compare(a,pool.arrays['output'],budget['fp16_atol'],budget['fp16_rtol'])
        record={'case':case['name'],'real_capture':case['real_capture'],'candidate_vs_upstream':equality,'upstream_vs_capture_or_cpu':captured,'hashes':case.get('sha256'), 'candidate_output_sha256':hashlib.sha256(b.cpu().numpy().tobytes()).hexdigest()}
        # This variant retains exact per-element FMA ordering, so demand bit equality.
        if not equality['bitwise_equal'] or not captured['passed']:
            record['status']='failed';results.append(record);write('results.json',{'status':'correctness_failed','cases':results,'adopted':False});return
        if not case['real_capture']:
            results.append(record);continue
        for backend in [0,1]:
            for _ in range(warmup):pool(backend)
        torch.cuda.synchronize()
        for k in range(rounds):
            order=[0,1] if k%2==0 else [1,0];row={'case':case['name'],'round':k,'order':'AB' if order[0]==0 else 'BA'}
            for backend in order:
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();wall=time.perf_counter();begin.record();pool(backend);end.record();end.synchronize()
                row['upstream_ms' if backend==0 else 'candidate_ms']=begin.elapsed_time(end)
                row['upstream_host_wall_ms' if backend==0 else 'candidate_host_wall_ms']=(time.perf_counter()-wall)*1000
            row['faster']=row['candidate_ms']<row['upstream_ms'];rows.append(row)
        local=[r for r in rows if r['case']==case['name']]
        record['timing']={key:{'mean':float(np.mean([r[key] for r in local])),'p50':float(np.percentile([r[key] for r in local],50)),'p95':float(np.percentile([r[key] for r in local],95))} for key in ['upstream_ms','candidate_ms','upstream_host_wall_ms','candidate_host_wall_ms']}
        record['faster_pairs']=sum(r['faster'] for r in local)
        record['mean_saved_ms_by_order']={o:float(np.mean([r['upstream_ms']-r['candidate_ms'] for r in local if r['order']==o])) for o in ['AB','BA']}
        record['stable_local_benefit']=(record['faster_pairs']>=.8*rounds and min(record['mean_saved_ms_by_order'].values())>0 and record['timing']['candidate_ms']['p95']<=record['timing']['upstream_ms']['p95'])
        results.append(record)
    with (OUT/'paired_timings.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    real=[r for r in results if r['real_capture']]
    benefit=all(r['stable_local_benefit'] for r in real)
    write('results.json',{'status':'measured','cases':results,'rounds':rounds,'warmup':warmup,'call_counts':dict(pool.calls),
        'candidate_kernels_per_nonempty_forward':2,'candidate_launches_derive_from':'Every nonempty candidate call dispatches preindex_kernel then preindexed_pool_kernel; source and binary hashes attached',
        'timing_boundary':'Device-resident existing input and persistent output/workspace; original includes output memset + pool; candidate includes output memset + fresh preindex + pool. H2D/D2H, allocation and capture I/O excluded equally; no index-precompute amortization.',
        'stable_local_benefit':benefit,'optimization_adopted':False,'full_model_required':benefit,
        'full_model_status':'pending' if benefit else 'not_required_no_stable_local_benefit',
        'candidate_source_sha256':sha256(SOURCE),'candidate_library_sha256':sha256(LIB),'upstream_library_sha256':sha256(REF),'tool_sha256':sha256(__file__),
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'processes_before':processes_before,'processes_after':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True),
        'peak_torch_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_torch_reserved_bytes':torch.cuda.max_memory_reserved()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['analyze','benchmark']);p.add_argument('--rounds',type=int,default=30);p.add_argument('--warmup',type=int,default=10);a=p.parse_args()
    if a.stage=='analyze': analyze()
    else: benchmark(a.rounds,a.warmup)
