"""Isolated TensorRT >=10.6 FP16 build, COCO validation and paired native timing."""
import argparse,gzip,json,os,sys,time,subprocess,resource,traceback
from pathlib import Path
os.environ['NVIDIA_TF32_OVERRIDE']='0'
from deimv2_2025 import ROOT,sha,write,coco_rows
OUT=ROOT/'reports/perception_2025_20261013/deimv2/tensorrt'
MODEL=ROOT/'checkpoints/perception_2025/deimv2/deimv2_s.onnx'
ENGINE=MODEL.with_suffix('.fp16.engine')

def build(trt):
    logger=trt.Logger(trt.Logger.INFO);builder=trt.Builder(logger)
    network=builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(MODEL.read_bytes()):raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.set_flag(trt.BuilderFlag.FP16);config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,4*1024**3)
    profile=builder.create_optimization_profile()
    for name,shape in [('images',(1,3,640,640)),('orig_target_sizes',(1,2))]:profile.set_shape(name,shape,shape,shape)
    config.add_optimization_profile(profile);start=time.perf_counter()
    plan=builder.build_serialized_network(network,config)
    if plan is None:raise RuntimeError('TensorRT engine build returned None')
    ENGINE.write_bytes(bytes(plan));write(OUT/'build.json',dict(tensorrt=trt.__version__,onnx_sha256=sha(MODEL),engine_sha256=sha(ENGINE),seconds=time.perf_counter()-start,flags=['FP16'],tf32=False,workspace_bytes=4*1024**3,profile={'images':[1,3,640,640],'orig_target_sizes':[1,2]},precision_note='FP16 enabled with TensorRT layer precision selection, FP32 input/output'))

class Runner:
    def __init__(self,trt,torch):
        load_start=time.perf_counter();self.torch=torch;self.device_used_peak=0;self.logger=trt.Logger(trt.Logger.WARNING);self.runtime=trt.Runtime(self.logger);self.engine=self.runtime.deserialize_cuda_engine(ENGINE.read_bytes());self.context=self.engine.create_execution_context();self.buffers={};self.inputs=[];self.outputs=[];self.io=[]
        mapping={trt.float32:torch.float32,trt.float16:torch.float16,trt.int64:torch.int64,trt.int32:torch.int32,trt.bool:torch.bool}
        for i in range(self.engine.num_io_tensors):
            n=self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
                self.context.set_input_shape(n,(1,3,640,640) if n=='images' else (1,2));self.inputs.append(n)
            else:self.outputs.append(n)
        for n in self.inputs+self.outputs:
            shape=tuple(self.context.get_tensor_shape(n));dtype=self.engine.get_tensor_dtype(n);assert all(x>0 for x in shape),(n,shape)
            self.buffers[n]=torch.empty(shape,dtype=mapping[dtype],device='cuda');self.context.set_tensor_address(n,self.buffers[n].data_ptr());self.io.append(dict(name=n,shape=shape,dtype=str(dtype),mode='input' if n in self.inputs else 'output'))
        self.load_ms=(time.perf_counter()-load_start)*1000
    def __call__(self,x,size):
        torch=self.torch;start=time.perf_counter();self.buffers['images'].copy_(x);self.buffers['orig_target_sizes'].copy_(size);torch.cuda.synchronize();h2d=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):raise RuntimeError('execute_async_v3 failed')
        b.record();b.synchronize();free,total=torch.cuda.mem_get_info();self.device_used_peak=max(self.device_used_peak,total-free);result=[self.buffers[n][0].cpu().numpy() for n in ['labels','boxes','scores']];end=time.perf_counter()
        return result,dict(h2d_wall_ms=(h2d-start)*1000,forward_post_gpu_ms=a.elapsed_time(b),decode_d2h_ms=(end-h2d)*1000-a.elapsed_time(b))

def geometry(reference,candidate):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    details=[]
    for cat in sorted(set(r['category_id'] for r in reference+candidate)):
        a=[r for r in reference if r['category_id']==cat];b=[r for r in candidate if r['category_id']==cat]
        if not a or not b:details.append(dict(category_id=cat,unmatched=len(a)+len(b),matched=0));continue
        A=np.array([r['bbox'] for r in a]);B=np.array([r['bbox'] for r in b]);ii,jj=linear_sum_assignment(np.linalg.norm(A[:,None]-B[None],axis=2));delta=np.abs(A[ii]-B[jj]);details.append(dict(category_id=cat,matched=len(ii),unmatched=abs(len(a)-len(b)),max_box_component_px=float(delta.max()),mean_box_component_px=float(delta.mean()),max_score_difference=max(abs(a[i]['score']-b[j]['score']) for i,j in zip(ii,jj))))
    return details

def audit_outliers():
    """CPU-only inspection; IoU gating avoids interpreting forced assignments as identity."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    with gzip.open(OUT/'all_image_geometry.json.gz','rt') as f:diagnostics=json.load(f)
    worst=sorted(diagnostics,key=lambda r:max((c.get('max_box_component_px',0) for c in r['score_ge_025']),default=0),reverse=True)[:3]
    ids={r['image_id'] for r in worst};native={i:[] for i in ids};candidate={i:[] for i in ids}
    for path,target in [(ROOT/'runs/perception_2025/deimv2/coco_predictions.json',native),(ROOT/'runs/perception_2025/deimv2/trt_predictions.json',candidate)]:
        for r in json.loads(path.read_text()):
            if r['image_id'] in ids:target[r['image_id']].append(r)
    def iou(a,b):
        A=np.asarray([r['bbox'] for r in a],dtype=float);B=np.asarray([r['bbox'] for r in b],dtype=float)
        lo=np.maximum(A[:,None,:2],B[None,:,:2]);hi=np.minimum(A[:,None,:2]+A[:,None,2:],B[None,:,:2]+B[None,:,2:]);inter=np.prod(np.maximum(hi-lo,0),axis=2);return inter/np.maximum(np.prod(A[:,2:],axis=1)[:,None]+np.prod(B[:,2:],axis=1)[None,:]-inter,1e-12)
    results=[]
    for image_id in [r['image_id'] for r in worst]:
        a=[r for r in native[image_id] if r['score']>=.25];b=[r for r in candidate[image_id] if r['score']>=.25];matched=[];unmatched=[];forced=[]
        for cat in sorted({r['category_id'] for r in a+b}):
            aa=[r for r in a if r['category_id']==cat];bb=[r for r in b if r['category_id']==cat];taken_a=set();taken_b=set()
            if aa and bb:
                overlap=iou(aa,bb);ii,jj=linear_sum_assignment(1-overlap)
                for i,j in zip(ii,jj):
                    if overlap[i,j]>=.5:
                        taken_a.add(int(i));taken_b.add(int(j));matched.append(dict(category_id=cat,iou=float(overlap[i,j]),native=aa[i],tensorrt=bb[j],max_xywh_component_difference_px=float(np.abs(np.array(aa[i]['bbox'])-bb[j]['bbox']).max())))
                A=np.asarray([r['bbox'] for r in aa]);B=np.asarray([r['bbox'] for r in bb]);fi,fj=linear_sum_assignment(np.linalg.norm(A[:,None]-B[None],axis=2))
                for i,j in zip(fi,fj):forced.append(dict(category_id=cat,iou=float(overlap[i,j]),native=aa[i],tensorrt=bb[j],max_xywh_component_difference_px=float(np.abs(A[i]-B[j]).max())))
            for name,rows,taken,other in [('native',aa,taken_a,candidate[image_id]),('tensorrt',bb,taken_b,native[image_id])]:
                other=[r for r in other if r['category_id']==cat]
                for i,r in enumerate(rows):
                    if i in taken:continue
                    best=None
                    if other:
                        overlaps=iou([r],other)[0];j=int(np.argmax(overlaps));best=dict(iou=float(overlaps[j]),prediction=other[j],below_score_threshold=other[j]['score']<.25)
                    unmatched.append(dict(backend=name,prediction=r,best_same_class_in_other_complete_300=best))
        results.append(dict(image_id=image_id,score_threshold=.25,iou_gate=.5,native_candidates=len(a),tensorrt_candidates=len(b),matched=matched,unmatched=unmatched,worst_forced_xywh_assignment=max(forced,key=lambda r:r['max_xywh_component_difference_px']) if forced else None,interpretation='Matching is diagnostic correspondence only, not decoder query identity or physical motion; failed IoU-gate pairs remain unmatched.'))
    write(OUT/'worst_three_iou_audit.json',results);write(OUT/'audit_status.json',dict(status='passed',phase='audit',images=len(results),meaning='diagnostic generated; not candidate equivalence acceptance'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['build','validate','benchmark','audit'],required=True);ap.add_argument('--limit',type=int);args=ap.parse_args();OUT.mkdir(parents=True,exist_ok=True);status_path=OUT/({'build':'build_status.json','validate':'status.json','benchmark':'benchmark_status.json','audit':'audit_status.json'}[args.phase]);write(status_path,dict(status='running',phase=args.phase,tool_sha256=sha(Path(__file__))))
    if args.phase=='audit':audit_outliers();return
    import tensorrt as trt
    assert tuple(map(int,trt.__version__.split('.')[:2]))>=(10,6)
    if args.phase=='build':build(trt);write(status_path,dict(status='passed',phase='build'));return
    manifest=json.loads((OUT/'build.json').read_text());assert manifest['onnx_sha256']==sha(MODEL),'ONNX changed since build';assert manifest['engine_sha256']==sha(ENGINE),'engine changed since build';assert manifest['tensorrt']==trt.__version__,'TensorRT version changed since build'
    import numpy as np
    import torch,torchvision.transforms as T
    from PIL import Image
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runner=Runner(trt,torch);write(OUT/'io.json',dict(tensorrt=trt.__version__,torch=torch.__version__,cuda=torch.version.cuda,io=runner.io,engine_sha256=sha(ENGINE),tool_sha256=sha(Path(__file__)),runtime_load_ms=runner.load_ms,cold_start_scope='engine deserialize, execution context and buffer allocation after Python imports; not complete process startup'))
    trans=T.Compose([T.Resize((640,640)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])]);inventory=json.loads((OUT.parents[1]/'inputs/coco_files.json').read_text());ann=ROOT/'data/coco/annotations/instances_val2017.json';cats=sorted(x['id'] for x in json.loads(ann.read_text())['categories'])
    def infer(item,fn=runner):
        start=time.perf_counter()
        with Image.open(item['file']) as source:im=source.convert('RGB')
        x=trans(im).unsqueeze(0);size=torch.tensor([im.size],dtype=torch.int64);prep=time.perf_counter();result,t=fn(x,size);end=time.perf_counter();assert np.all((result[2]>=0)&(result[2]<=1)),'score out of range';return coco_rows(item['id'],*result,cats),dict(read_preprocess_ms=(prep-start)*1000,e2e_ms=(end-start)*1000,**t)
    for _ in range(5):infer(inventory[0])
    if args.phase=='benchmark':
        native_load_start=time.perf_counter();cfgdata=json.loads((ROOT/'configs/perception_2025.yaml').read_text())['models']['deimv2'];source=ROOT/cfgdata['source'];sys.path.insert(0,str(source));from engine.core import YAMLConfig
        cfg=YAMLConfig(str(source/cfgdata['config']),resume=str(ROOT/cfgdata['checkpoint']));cfg.yaml_cfg['DINOv3STAs']['weights_path']=None
        ck=torch.load(ROOT/cfgdata['checkpoint'],map_location='cpu',weights_only=False);model=cfg.model;model.load_state_dict(ck['ema']['module'] if 'ema' in ck else ck['model'],strict=True);model=model.eval().cuda().deploy();post=cfg.postprocessor.deploy().cuda();native_load_ms=(time.perf_counter()-native_load_start)*1000
        def native(x,size):
            start=time.perf_counter();x=x.cuda();size=size.cuda();torch.cuda.synchronize();h2d=time.perf_counter();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
            with torch.inference_mode():result=post(model(x),size)
            b.record();b.synchronize();result=[v[0].cpu().numpy() for v in result];end=time.perf_counter();return result,dict(h2d_wall_ms=(h2d-start)*1000,forward_post_gpu_ms=a.elapsed_time(b),decode_d2h_ms=(end-h2d)*1000-a.elapsed_time(b))
        for _ in range(10):infer(inventory[0],native);infer(inventory[0],runner)
        rounds=[];torch.cuda.reset_peak_memory_stats()
        for i in range(30):
            item=inventory[[0,len(inventory)//2,len(inventory)-1][i%3]];order=['native','tensorrt'] if i%2==0 else ['tensorrt','native'];row=dict(round=i,image_id=item['id'],order=order)
            for name in order:_,row[name]=infer(item,native if name=='native' else runner)
            rounds.append(row)
        summary={name:{key:dict(mean=float(np.mean(v)),p50=float(np.percentile(v,50)),p95=float(np.percentile(v,95)),std=float(np.std(v))) for key in rounds[0][name] for v in [[r[name][key] for r in rounds]]} for name in ['native','tensorrt']}
        write(OUT/'paired_benchmark.json',dict(load_ms={'native':native_load_ms,'tensorrt':runner.load_ms},load_scope='sequential setup after initial imports; native loading after TensorRT warmup, not comparable complete process cold starts',rounds=rounds,summary=summary,paired_e2e_saving_ms=[r['native']['e2e_ms']-r['tensorrt']['e2e_ms'] for r in rounds],gpu_peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_device_used_peak_bytes=runner.device_used_peak,gpu_memory_scope='device total used including TensorRT and other resident processes; sampled at inference completion',cpu_torch_threads=torch.get_num_threads(),ram_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='30 alternating paired rounds on first/middle/last COCO images; both reread, decode, normalize, H2D, inference with postprocess, D2H; JSON conversion excluded; OS cache warm; TensorRT internal allocations excluded from torch peak',tf32=False,batch=1));write(status_path,dict(status='passed',phase='benchmark',rounds=30));return
    refby={}
    for row in json.loads((ROOT/'runs/perception_2025/deimv2/coco_predictions.json').read_text()):refby.setdefault(row['image_id'],[]).append(row)
    gates=[]
    for idx in [0,len(inventory)//2,len(inventory)-1]:
        item=inventory[idx];rows,t=infer(item);gates.append(dict(image_id=item['id'],predictions=len(rows),timings=t,all_predictions=geometry(refby[item['id']],rows),score_ge_025=geometry([r for r in refby[item['id']] if r['score']>=.25],[r for r in rows if r['score']>=.25])))
    write(OUT/'three_image_geometry.json',gates)
    selected=inventory[:args.limit] if args.limit else inventory;pred=[];timings=[]
    for i,item in enumerate(selected):
        rows,t=infer(item);pred.extend(rows);timings.append(dict(image_id=item['id'],**t))
        if i%100==0:print('TensorRT',i,len(selected),flush=True)
    path=ROOT/'runs/perception_2025/deimv2/trt_predictions.json';write(path,pred)
    with gzip.open(OUT/'predictions.json.gz','wt') as f:json.dump(pred,f)
    write(OUT/'image_ids.json',[r['id'] for r in selected]);write(OUT/'timings.json',timings)
    with (OUT/'official_eval.log').open('w') as log:subprocess.run([str(ROOT/'envs/nuscenes_eval/bin/python'),str(ROOT/'tools/coco_eval_2025.py'),'--annotations',str(ann),'--predictions',str(path),'--image-ids',str(OUT/'image_ids.json'),'--out',str(OUT/'official_eval')],check=True,stdout=log,stderr=subprocess.STDOUT)
    diagnostics=[]
    for i,item in enumerate(selected):
        rows=pred[i*300:(i+1)*300];assert len(rows)==300 and all(r['image_id']==item['id'] for r in rows),'unexpected candidate count/order'
        diagnostics.append(dict(image_id=item['id'],all_predictions=geometry(refby[item['id']],rows),score_ge_025=geometry([r for r in refby[item['id']] if r['score']>=.25],[r for r in rows if r['score']>=.25])))
    with gzip.open(OUT/'all_image_geometry.json.gz','wt') as f:json.dump(diagnostics,f)
    stats={}
    for scope in ['all_predictions','score_ge_025']:
        values=[c.get('max_box_component_px',0.) for row in diagnostics for c in row[scope]];stats[scope]=dict(max_box_component_px=max(values),p95_class_max_box_component_px=float(np.percentile(values,95)),unmatched=sum(c['unmatched'] for row in diagnostics for c in row[scope]),matched=sum(c['matched'] for row in diagnostics for c in row[scope]))
    write(OUT/'geometry_summary.json',dict(images=len(selected),checks=dict(finite=True,nonnegative_box_extent=True,valid_categories=True,scores_in_0_1=True,predictions=len(pred)),matching='within-class Hungarian Euclidean xywh; no row identity assumption; unmatched counts include confidence cutoff changes',scopes=stats))
    metrics=json.loads((OUT/'official_eval/metrics.json').read_text());reference=json.loads((OUT.parent/'official_eval/metrics.json').read_text());same=[r['id'] for r in selected]==json.loads((OUT.parent/'image_ids.json').read_text());drop=reference['AP']-metrics['AP'] if same else None
    write(OUT/'status.json',dict(status='partial',images=len(selected),metrics=metrics,ap_drop=drop,ap_budget_pass=drop<=.005 if same else None,ap_comparison_same_image_ids=same,geometry='all_image_geometry.json.gz and geometry_summary.json; all images checked; low-score candidate ordering may differ',engine_sha256=sha(ENGINE),prediction_sha256=sha(path),gpu_device_used_peak_bytes=runner.device_used_peak,gpu_memory_scope='device total used including TensorRT and other resident processes; sampled at inference completion',ram_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,timing_limit='unpaired sequence; paired benchmarking separately',source='new_measured'))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        phase=sys.argv[sys.argv.index('--phase')+1] if '--phase' in sys.argv else 'unknown'
        name={'build':'build_status.json','validate':'status.json','benchmark':'benchmark_status.json','audit':'audit_status.json'}.get(phase,'error.json')
        write(OUT/name,dict(status='failed',phase=phase,error=str(exc),traceback=traceback.format_exc()))
        raise
