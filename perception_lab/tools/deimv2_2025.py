"""Pinned native DEIMv2-S inference, sparse COCO categories and audited deployment."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def publish_native_manifest(out,run):
    """Enable resume only after every required native output exists and is hashed."""
    status=json.loads((out/'status.json').read_text())
    if status.get('status')!='passed' or status.get('coco_images')!=5000:
        raise ValueError('Complete native COCO run required for artifact manifest')
    names=['status.json','coco_predictions.json.gz','mini_camera_predictions.json.gz',
           'image_ids.json','official_eval/metrics.json','official_eval/protocol.json',
           'author_faster_coco_eval/metrics.json','profile.json','timings.json',
           'strict_load.json','hooks.json','deploy_equivalence.json','three_image_gate.json',
           'resolved_config.json','resource_manifest.json']
    paths=[out/name for name in names]+[run/'coco_predictions.json',run/'mini_camera_predictions.json']
    # Construct all hashes before publishing complete=true; missing outputs fail closed.
    records=[dict(path=str(path.resolve()),sha256=sha(path)) for path in paths]
    write(out/'artifact_manifest.json',dict(complete=True,scope='complete native DEIM inference and evaluation',artifacts=records))

def coco_rows(image_id,labels,boxes,scores,categories):
    import numpy as np
    boxes=np.asarray(boxes);scores=np.asarray(scores);labels=np.asarray(labels)
    if boxes.shape!=(len(scores),4) or labels.shape!=scores.shape:raise ValueError('output shape')
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():raise ValueError('nonfinite prediction')
    if not ((labels>=0)&(labels<len(categories))).all():raise ValueError('unknown category')
    if not (boxes[:,2:]>=boxes[:,:2]).all():raise ValueError('negative box extent')
    return [dict(image_id=int(image_id),category_id=int(categories[int(l)]),bbox=[float(b[0]),float(b[1]),float(b[2]-b[0]),float(b[3]-b[1])],score=float(s)) for l,b,s in zip(labels,boxes,scores)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/perception_2025.yaml');ap.add_argument('--phase',choices=['run','optimize','preflight'],default='run');ap.add_argument('--limit',type=int);a=ap.parse_args()
    cp=Path(a.config);cp=cp if cp.is_absolute() else ROOT/cp;c=json.loads(cp.read_text());m=c['models']['deimv2'];out=ROOT/c['report']/'deimv2';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/m['source'];checkpoint=ROOT/m['checkpoint'];modelcfg=source/m['config']
    old_status=json.loads((out/'status.json').read_text()) if (out/'status.json').exists() else {}
    if a.phase=='run':write(out/'status.json',dict(status='running',phase=a.phase,inference_images=0))
    identity_manifest=json.loads((out/'resource_manifest.json').read_text())
    assert identity_manifest['source_commit']==m['commit']
    for f in identity_manifest['files']:assert sha(source/f['path'])==f['sha256'],f['path']
    if not checkpoint.exists() or not modelcfg.exists():
        write(out/'status.json',dict(status='blocked_resource',reason='missing native checkpoint or pinned source',inference_images=0));return 2
    if a.phase=='preflight':print(json.dumps({'checkpoint_sha256':sha(checkpoint),'model_config_sha256':sha(modelcfg)}));return 0
    if a.phase=='optimize':
        if old_status.get('coco_images')!=5000:
            write(out/'optimization.json',dict(status='blocked_data',reason='complete native COCO evaluation is prerequisite'));return 2
        write(out/'optimization.json',dict(status='running'))
        run=ROOT/'runs/perception_2025/deimv2';run.mkdir(parents=True,exist_ok=True)
        original=(source/'tools/deployment/export_onnx.py').read_text()
        patched=original.replace('data = torch.rand(32, 3, *img_size)', 'data = torch.rand(1, 3, *img_size)')
        patched=patched.replace("sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))", 'sys.path.insert(0, '+repr(str(source))+')')
        assert patched!=original
        export_script=run/'export_onnx_batch1.py';export_script.write_text(patched)
        import difflib
        (out/'export_batch1.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='author/export_onnx.py',tofile='diagnostic/export_onnx_batch1.py')))
        jobs=[('onnx_export',[sys.executable,str(export_script),'-c',str(modelcfg),'-r',str(checkpoint),'--check'])]
        for key in ['onnx']:
            exe=ROOT/c['environments'][key]
            if not exe.exists():
                write(out/'optimization.json',dict(status='blocked_resource',reason='missing isolated '+key+' environment'));return 2
        jobs.append(('onnx_full',[str(ROOT/c['environments']['onnx']),str(ROOT/'tools/deimv2_onnx_2025.py')]))
        jobs.append(('onnx_paired',[sys.executable,str(ROOT/'tools/deimv2_paired_onnx_2025.py')]))
        jobs.append(('onnx_geometry',[str(ROOT/c['environments']['preflight']),str(ROOT/'tools/deimv2_onnx_geometry_2025.py')]))
        trt_available=(ROOT/c['environments']['tensorrt']).exists()
        if trt_available:jobs.extend(('tensorrt_'+phase,[str(ROOT/c['environments']['tensorrt']),str(ROOT/'tools/deimv2_trt_2025.py'),'--phase',phase]) for phase in ['build','validate','benchmark'])
        env=dict(os.environ,OMP_NUM_THREADS='8')
        for name,cmd in jobs:
            with (out/(name+'.log')).open('w') as log:
                rc=subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
            if name=='onnx_export' and rc==0:
                write(out/'export_identity.json',dict(author_sha256=sha(source/'tools/deployment/export_onnx.py'),executed_sha256=sha(export_script),checkpoint_sha256=sha(checkpoint),model_config_sha256=sha(modelcfg),onnx_sha256=sha(checkpoint.with_suffix('.onnx')),source_manifest_sha256=sha(out/'resource_manifest.json'),changes=['demo batch32 to1','absolute pinned-source import path'],args=['--check'],scope='Successful controlled export; native/source checks preceded invocation'))
            if rc:
                write(out/'optimization.json',dict(status='failed',step=name,returncode=rc));return 1
        if not trt_available:
            write(out/'optimization.json',dict(status='blocked_resource',reason='missing isolated TensorRT environment',onnx='completed'));return 2
        write(out/'optimization.json',dict(status='partial',backend_runs='completed',adoption='native retained pending full geometry and paired-performance acceptance',onnx_status='onnx/status.json',tensorrt_status='tensorrt/status.json'))
        return 0
    import numpy as np
    import torch
    import torchvision
    import torchvision.transforms as T
    from PIL import Image
    sys.path.insert(0,str(source));from engine.core import YAMLConfig
    os.chdir(source)
    cfg=YAMLConfig(str(modelcfg),resume=str(checkpoint));resolved=cfg.yaml_cfg
    assert resolved['DINOv3STAs']['embed_dim']==192 and resolved['DINOv3STAs']['interaction_indexes']==[3,7,11]
    assert resolved['DEIMTransformer']['num_layers']==4
    assert resolved['DINOv3STAs']['name']=='vit_tiny'
    assert resolved['DEIMTransformer']['num_queries']==300
    # Complete detector strict loading covers every backbone tensor. No init-weight download.
    resolved['DINOv3STAs']['weights_path']=None
    write(out/'resolved_config.json',resolved)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);state=ck['ema']['module'] if 'ema' in ck else ck['model'];model=cfg.model
    result=model.load_state_dict(state,strict=True)
    identity=dict(checkpoint_sha256=sha(checkpoint),model_config_sha256=sha(modelcfg),checkpoint_branch='ema.module' if 'ema' in ck else 'model',missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,tensors=len(state),parameters=sum(p.numel() for p in model.parameters()),torch=torch.__version__,torchvision=torchvision.__version__,cuda=torch.version.cuda,weights_path_override='None; full checkpoint loaded strictly before execution')
    write(out/'strict_load.json',identity)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required, CPU is not a 4090 result')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model.eval().cuda();post=cfg.postprocessor.deploy().cuda()
    trans=T.Compose([T.Resize((640,640)),T.ToTensor(),T.Normalize([.485,.456,.406],[.229,.224,.225])])
    hooks={};handles=[]
    def hook(name):
        def fn(mod,ins,outs):
            def shape(x):
                if isinstance(x,torch.Tensor):
                    if not torch.isfinite(x).all():raise ValueError('nonfinite '+name)
                    return {'shape':list(x.shape),'dtype':str(x.dtype)}
                if isinstance(x,(list,tuple)):return [shape(z) for z in x]
                if isinstance(x,dict):return {k:shape(v) for k,v in x.items()}
                return type(x).__name__
            old=hooks.get(name,{'calls':0});hooks[name]={'calls':old['calls']+1,'output':shape(outs)}
        return fn
    required_names=['backbone.dinov3','backbone.sta','backbone','encoder','decoder']
    for name in required_names:
        handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    inventory=json.loads((ROOT/c['report']/'inputs/coco_files.json').read_text());sample=Image.open(inventory[0]['file']).convert('RGB');x=trans(sample).unsqueeze(0).cuda();size=torch.tensor([sample.size],device='cuda')
    with torch.inference_mode():
        before=model(x);before={k:v.clone() for k,v in before.items() if isinstance(v,torch.Tensor)}
        model=model.deploy();after=model(x)
        diffs={k:dict(max_abs=float((before[k]-after[k]).abs().max()),passed=bool(torch.allclose(before[k],after[k],atol=1e-4,rtol=1e-3))) for k in before}
    write(out/'deploy_equivalence.json',diffs);write(out/'hooks.json',hooks)
    assert all(v['passed'] for v in diffs.values()),'native deploy equivalence failed'
    assert all(hooks.get(n,{}).get('calls',0)>0 for n in required_names)
    assert len(hooks['backbone']['output'])==3
    for h in handles:h.remove()
    timings=[];pred=[];ann=ROOT/c['coco']/'annotations/instances_val2017.json';gt=json.loads(ann.read_text());cats=sorted(x['id'] for x in gt['categories']);assert len(cats)==80
    with torch.inference_mode():
        for _ in range(5):post(model(x),size)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    def infer(path):
        begin=time.perf_counter();im=Image.open(path).convert('RGB');xx=trans(im).unsqueeze(0);orig=torch.tensor([im.size]);read=time.perf_counter();xx=xx.cuda();orig=orig.cuda();torch.cuda.synchronize();h2d=time.perf_counter()
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
        with torch.inference_mode():ys=post(model(xx),orig)
        end.record();end.synchronize();labels,boxes,scores=[v[0].detach().cpu().numpy() for v in ys];finish=time.perf_counter()
        return labels,boxes,scores,{'read_preprocess_ms':(read-begin)*1000,'h2d_wall_ms':(h2d-read)*1000,'forward_post_gpu_ms':start.elapsed_time(end),'e2e_ms':(finish-begin)*1000},im.size
    # Three real COCO images before the full sequence.
    gates=[]
    for idx in [0,len(inventory)//2,len(inventory)-1]:
        item=inventory[idx];lab,box,sc,t,wh=infer(item['file']);rr=coco_rows(item['id'],lab,box,sc,cats);gates.append({'image_id':item['id'],'predictions':len(rr),'size':wh,'timings':t})
    write(out/'three_image_gate.json',gates)
    selected=inventory[:a.limit] if a.limit is not None else inventory
    for i,item in enumerate(selected):
        lab,box,sc,t,wh=infer(item['file']);pred.extend(coco_rows(item['id'],lab,box,sc,cats));timings.append({'image_id':item['id'],**t})
        if i%100==0:print('COCO',i,len(selected),flush=True)
    run=ROOT/'runs/perception_2025/deimv2';run.mkdir(parents=True,exist_ok=True);predpath=run/'coco_predictions.json';write(predpath,pred)
    with gzip.open(out/'coco_predictions.json.gz','wt') as f:json.dump(pred,f)
    write(out/'timings.json',timings)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco=COCO(str(ann));ev=COCOeval(coco,coco.loadRes(str(predpath)),'bbox');ev.params.imgIds=[r['id'] for r in selected];ev.evaluate();ev.accumulate();ev.summarize()
    names=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl'];metrics=dict(zip(names,map(float,ev.stats)));write(out/'author_faster_coco_eval/metrics.json',metrics)
    write(out/'image_ids.json',[r['id'] for r in selected])
    with (out/'official_eval.log').open('w') as log:
        subprocess.run([str(ROOT/'envs/nuscenes_eval/bin/python'),str(ROOT/'tools/coco_eval_2025.py'),'--annotations',str(ann),'--predictions',str(predpath),'--image-ids',str(out/'image_ids.json'),'--out',str(out/'official_eval')],check=True,stdout=log,stderr=subprocess.STDOUT)
    metrics=json.loads((out/'official_eval/metrics.json').read_text())
    six=[]
    for packet in json.loads((ROOT/c['splits']['scene0061']).read_text()):
        for cam in packet['camera_order']:
            lab,box,sc,t,wh=infer(packet['sensors'][cam]['path']);six.append({'sample_token':packet['sample_token'],'timestamp_us':packet['timestamp_us'],'camera':cam,'image':packet['sensors'][cam]['path'],'labels':lab.tolist(),'boxes_xyxy':box.tolist(),'scores':sc.tolist(),'timing':t})
    write(run/'mini_camera_predictions.json',six)
    with gzip.open(out/'mini_camera_predictions.json.gz','wt') as f:json.dump(six,f)
    perf={k:{'mean':float(np.mean([r[k] for r in timings])),'p50':float(np.percentile([r[k] for r in timings],50)),'p95':float(np.percentile([r[k] for r in timings],95))} for k in timings[0] if k!='image_id'}
    write(out/'profile.json',dict(timings=perf,gpu_peak_bytes=torch.cuda.max_memory_allocated(),ram_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,tf32=False,dtype='float32',batch=1,excluded='evaluation, writing, visualization',optimization=False))
    write(out/'status.json',dict(status='passed' if len(selected)==5000 else 'partial',native_inference='passed',coco_images=len(selected),mini_camera_images=len(six),official_eval=metrics,backend='not_yet_run',source='new_measured',prediction_sha256=sha(predpath)))
    if len(selected)==5000:publish_native_manifest(out,run)
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config',default='configs/perception_2025.yaml');parser.add_argument('--phase',default='run');args,_=parser.parse_known_args()
        cp=Path(args.config);cp=cp if cp.is_absolute() else ROOT/cp
        try:report=ROOT/json.loads(cp.read_text())['report']/'deimv2'
        except Exception:report=ROOT/'reports/perception_2025_20261013/deimv2'
        write(report/('optimization.json' if args.phase=='optimize' else 'status.json'),dict(status='failed',error=repr(exc)))
        raise
