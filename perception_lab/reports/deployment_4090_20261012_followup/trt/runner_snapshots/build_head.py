"""Isolated FP16 diagnostic capture/replay. Production plans/libraries are read-only.

Source tools/bevfusion_trt_env.sh before use. prepare is CPU-only. run requires
GPU, and must be invoked separately with --library production or diagnostic.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'third_party/Lidar_AI_Solution/CUDA-BEVFusion'
RUN = ROOT / 'runs/deployment_4090_20261012_followup/trt'
REPORT = ROOT / 'reports/deployment_4090_20261012_followup/trt'
MODEL = ROOT / 'checkpoints/deployment_4090/nvidia/model/resnet50'
TOKENS = ['e0845f5322254dafadbbed75aaa07969', 'b26e791522294bec90f86fd72226e35c', '747aa46b9a4641fe90db05d97db2acea']
NODES = {'camera.feature':'camera_feature', 'camera.depth':'camera_depth_weights', 'camera.sparse_depth':'depth', 'camera.bev_pool':'feat_in', 'camera.view_transform':'feat_out', 'lidar.bev':'lidar', 'head.input':'middle'}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')

def identity():
    paths=[SRC/'build/libbevfusion_core.so',SRC/'build/libpybev.so',SRC/'build/libcustom_layernorm.so']+list((MODEL/'build').glob('*.plan'))+list(MODEL.glob('*.onnx'))
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}

def selected_packets():
    base=ROOT/'reports/deployment_4090_20261008/inputs'
    first=json.loads((base/'scene0061_packets.json').read_text())[0]
    pool={p['sample_token']:p for s in ['scene0061','mini_val'] for p in json.loads((base/(s+'_packets.json')).read_text())}
    return [first]+[pool[t] for t in TOKENS if t!=first['sample_token']]

def prepare():
    RUN.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    original=identity();write(REPORT/'production_identity.json',original)
    write(REPORT/'packets.json',selected_packets())
    isolated=RUN/'build';isolated.mkdir(exist_ok=True)
    source=RUN/'source/src/common';source.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SRC/'src/common/tensorrt.cpp',source/'tensorrt.cpp')
    subprocess.run(['patch','-p1','-i',str(ROOT/'tools/bevfusion_followup_capture.patch')],cwd=RUN/'source',check=True)
    flags={}
    for line in (SRC/'build/CMakeFiles/bevfusion_core.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            k,v=line.split(' = ',1);flags[k]=shlex.split(v)
    command=['/bin/c++']+sum([flags[k] for k in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']],[])+['-c',str(source/'tensorrt.cpp'),'-o',str(isolated/'tensorrt.cpp.o')]
    subprocess.run(command,check=True)
    link=shlex.split((SRC/'build/CMakeFiles/bevfusion_core.dir/link.txt').read_text())
    link[link.index('-o')+1]=str(isolated/'libbevfusion_core.so')
    link=[str(isolated/'tensorrt.cpp.o') if x=='CMakeFiles/bevfusion_core.dir/src/common/tensorrt.cpp.o' else x for x in link]
    subprocess.run(link,cwd=SRC/'build',check=True)
    for name in ['libpybev.so','libcustom_layernorm.so']:shutil.copy2(SRC/'build'/name,isolated/name)
    assert original==identity(),'Production identity changed'
    write(REPORT/'build.json',dict(status='compiled_cpu_only',compile=command,link=link,diagnostic_library=sha(isolated/'libbevfusion_core.so'),production_unchanged=True))

def stats(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:return dict(passed=False,shape_match=False,reference_shape=list(a.shape),candidate_shape=list(b.shape))
    if a.dtype.kind in 'biu' or b.dtype.kind in 'biu':
        return dict(shape=list(a.shape),reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype),passed=bool(np.array_equal(a,b)),exceeded_elements=int((a!=b).sum()),tolerance='exact integer identity; no floating tolerance')
    x=a.astype(np.float64);y=b.astype(np.float64);d=np.abs(x-y);mask=np.abs(x)>.01
    q=lambda z:{k:float(v) for k,v in zip(['p50','p95','p99','max'],np.percentile(z[np.isfinite(z)],[50,95,99,100]))} if np.isfinite(z).any() else None
    return dict(shape=list(a.shape),reference_dtype=str(a.dtype),candidate_dtype=str(b.dtype),bitwise_equal=bool(a.dtype==b.dtype and a.tobytes()==b.tobytes()),nonfinite_reference=int((~np.isfinite(x)).sum()),nonfinite_candidate=int((~np.isfinite(y)).sum()),absolute=q(d),relative_abs_reference_gt_001=q(d[mask]/np.abs(x[mask])),near_zero_reference_elements=int((~mask).sum()),near_zero_absolute=q(d[~mask]),elements=int(a.size),passed=bool(np.isfinite(x).all() and np.isfinite(y).all() and (d<=.05+.02*np.abs(x)).all()),exceeded_elements=int((d>.05+.02*np.abs(x)).sum()),tolerance='abs(candidate-reference) <= 0.05 + 0.02*abs(reference)')

def read_binding(directory,name):
    meta=json.loads((directory/(name+'.json')).read_text())
    return np.fromfile(directory/(name+'.bin'),dtype=meta['dtype']).reshape(meta['shape'])

def projection_collision_mask(points,matrices):
    pts=points[:,:3].astype(np.float32);lidar2image=matrices[2][0];aug=matrices[3][0]
    mask=np.zeros((6,256,704),dtype=bool)
    for cam in range(6):
        xyz=pts@lidar2image[cam,:3,:3].T+lidar2image[cam,:3,3]
        dist=np.clip(xyz[:,2],1e-5,1e5);xyz[:,:2]/=dist[:,None];xyz[:,2]=dist
        uv=xyz@aug[cam,:3,:3].T+aug[cam,:3,3]
        valid=(uv[:,0]>=0)&(uv[:,0]<704)&(uv[:,1]>=0)&(uv[:,1]<256)
        xy=uv[valid,:2].astype(np.int64);counts=np.bincount(xy[:,1]*704+xy[:,0],minlength=256*704)
        mask[cam]=counts.reshape(256,704)>1
    return mask

def run(args):
    from bevfusion_4090 import packet_input,make_pipeline,standardize
    from bevfusion_compare import compare
    cfg=json.loads((ROOT/'configs/deployment_4090.yaml').read_text());sys.path.insert(0,str(ROOT/cfg['bevfusion']['source']))
    build=SRC/'build' if args.library=='production' else RUN/'build'
    # Explicit preload overrides production RUNPATH without rewriting it.
    ctypes.CDLL(str(build/'libbevfusion_core.so'),mode=ctypes.RTLD_GLOBAL)
    sys.path.insert(0,str(build));import libpybev
    import torch
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    from mmdet3d.core.bbox import LiDARInstance3DBoxes,Box3DMode
    configs.load(str(ROOT/cfg['bevfusion']['config']),recursive=True);mc=Config(recursive_eval(configs));pipeline,_=make_pipeline(mc)
    head=cfg['tensorrt']['head_engine'];core=libpybev.load_bevfusion(str(MODEL/'build/camera.backbone.plan'),str(MODEL/'build/camera.vtransform.plan'),str(MODEL/'lidar.backbone.xyz.onnx'),str(MODEL/'build/fuser.plan'),str(MODEL/'build'/(head+'.plan')),'fp16');assert core is not None
    loaded=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if any(x in line for x in ['libbevfusion_core','libpybev','libnvinfer','libcustom_layernorm']) and '/' in line.split()[-1]})
    loaded_hashes={p:sha(p) for p in loaded};write(REPORT/(args.library+'_loaded_libraries.json'),loaded_hashes)
    core_hashes={h for p,h in loaded_hashes.items() if 'libbevfusion_core' in p}
    assert core_hashes=={sha(build/'libbevfusion_core.so')},'Unexpected loaded core library'
    saved=identity();allrows=[]
    for packet in selected_packets():
        token=packet['sample_token'];out=RUN/args.library/token;out.mkdir(parents=True,exist_ok=True)
        item=packet_input(packet);item.update(box_type_3d=LiDARInstance3DBoxes,box_mode_3d=Box3DMode.LIDAR);item=pipeline(item)
        def array(k):
            v=item[k];v=v.data if hasattr(v,'data') else v;v=v.numpy() if hasattr(v,'numpy') else v;return np.ascontiguousarray(v,dtype=np.float32)
        images=array('img')[None];points=array('points');matrices=[array(k)[None] for k in ['camera2lidar','camera_intrinsics','lidar2image','img_aug_matrix']]
        np.savez(out/'inputs.npz',images=images,points=points,**dict(zip(['camera2lidar','camera_intrinsics','lidar2image','img_aug_matrix'],matrices)))
        write(out/'inputs.json',{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(images=images,points=points,**dict(zip(['camera2lidar','camera_intrinsics','lidar2image','img_aug_matrix'],matrices))).items()})
        images=images.astype(np.float16);points=points.astype(np.float16);core.update(*matrices)
        for _ in range(3):core.forward(images,points,False,False)
        raw0=None;boxes0=None
        for repeat in range(args.repeats):
            d=out/('repeat_%d'%repeat);d.mkdir(exist_ok=True)
            if args.library=='diagnostic':
                os.environ['CAR_TRT_CAPTURE']=str(d);os.environ['CAR_BEV_POOL_CAPTURE']=str(d/'pool')
            try:
                raw=np.asarray(core.forward(images,points,False,False)).copy();torch.cuda.synchronize()
            finally:
                os.environ.pop('CAR_TRT_CAPTURE',None);os.environ.pop('CAR_BEV_POOL_CAPTURE',None)
            np.save(d/'native.npy',raw)
            detection=dict(boxes_3d=LiDARInstance3DBoxes(torch.tensor(raw[:,:9]),box_dim=9),scores_3d=torch.tensor(raw[:,10]),labels_3d=torch.tensor(raw[:,9].astype(np.int64)))
            boxes,audit=standardize(detection,packet,mc.object_classes);write(d/'standardized.json',boxes)
            if raw0 is None:raw0=raw;boxes0=boxes
            # Decoder atomic append order is not candidate identity. Raw row equality is reported only as such.
            row=dict(token=token,repeat=repeat,native_row_order=stats(raw0,raw))
            output_compare=compare([dict(sample_token=token,boxes3d=boxes0)],[dict(sample_token=token,boxes3d=boxes)],2.0)
            write(d/'output_hungarian_repeat.json',output_compare)
            row['output_sameclass_2m_hungarian']={k:output_compare[k] for k in ['matching','radius_m','frames','summary']}
            if args.library=='diagnostic' and (RUN/'production'/token/'repeat_0/standardized.json').exists():
                production_boxes=json.loads((RUN/'production'/token/'repeat_0/standardized.json').read_text())
                instrument_compare=compare([dict(sample_token=token,boxes3d=production_boxes)],[dict(sample_token=token,boxes3d=boxes)],2.0)
                write(d/'production_vs_diagnostic_hungarian.json',instrument_compare)
                row['production_vs_diagnostic_hungarian']={k:instrument_compare[k] for k in ['matching','radius_m','frames','summary']}
            if args.library=='diagnostic':
                row['nodes']={node:stats(read_binding(out/'repeat_0',binding),read_binding(d,binding)) for node,binding in NODES.items()}
                intervals=np.fromfile(d/'pool.intervals.bin',dtype=np.int32).reshape(-1,3)
                lengths=intervals[:,1]-intervals[:,0]
                row['geometry']=dict(intervals=len(intervals),contributions=int(lengths.sum()),indices_sha256=sha(d/'pool.indices.bin'),intervals_sha256=sha(d/'pool.intervals.bin'),last_interval=intervals[-1].tolist(),indices_count=(d/'pool.indices.bin').stat().st_size//4,contribution_count_percentiles=np.percentile(lengths,[50,95,99,100]).tolist())
                row['head']={name:stats(read_binding(out/'repeat_0',name),read_binding(d,name)) for name in ['reg','height','dim','rot','vel','score']}
            if args.library=='diagnostic':
                firstdepth=read_binding(out/'repeat_0','depth');depthnow=read_binding(d,'depth')
                changed=firstdepth!=depthnow
                collisions=projection_collision_mask(points,matrices)
                row['sparse_depth_collision_diagnostic']=dict(changed_pixels=int(changed.sum()),collision_pixels=int(collisions.sum()),changed_at_cpu_projected_collision=int((changed.reshape(collisions.shape)&collisions).sum()),projection_scope='CPU float32 reconstruction of CUDA FP16 point projection; FMA/boundary rounding may differ')
            allrows.append(row);write(REPORT/(args.library+'_repeat.json'),allrows)
    assert saved==identity();write(REPORT/(args.library+'_identity.json'),dict(production=saved,loaded_core=sha(build/'libbevfusion_core.so'),repeats=args.repeats,capture=args.library=='diagnostic',timing='diagnostic only; no performance claim'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','replay','mark','build-head','analyze']);p.add_argument('--library',choices=['production','diagnostic'],default='diagnostic');p.add_argument('--repeats',type=int,default=5);p.add_argument('--reference',type=Path);a=p.parse_args()
    if a.repeats<5:raise ValueError('At least 5 repeats required')
    if a.action=='prepare':prepare()
    elif a.action=='run':run(a)
    elif a.action=='mark':mark()
    elif a.action=='build-head':build_head()
    elif a.action=='analyze':analyze()
    else:replay(a)

# Replay and diagnostic graph export are intentionally separate from capture.
def mark():
    import onnx
    m=onnx.load(MODEL/'head.bbox.layernormplugin.onnx')
    top=[n for n in m.graph.node if n.op_type=='TopK'];assert len(top)==1
    n=top[0]
    for name,dtype,shape in [(n.input[0],onnx.TensorProto.FLOAT,[1,324000]),(n.output[0],onnx.TensorProto.FLOAT,[1,200]),(n.output[1],onnx.TensorProto.INT64,[1,200])]:
        m.graph.output.append(onnx.helper.make_tensor_value_info(name,dtype,shape))
    path=RUN/'head.diagnostic.onnx';onnx.save(m,path)
    write(REPORT/'head_diagnostic_export.json',dict(source_sha256=sha(MODEL/'head.bbox.layernormplugin.onnx'),diagnostic_sha256=sha(path),onnx_checker='Standard checker cannot validate existing NVIDIA CustomLayerNormalization op; TensorRT parser validation required',topk_node=n.name,input=n.input[0],values=n.output[0],indices=n.output[1],warning='Marked outputs may alter tactics. Validate original six outputs with identical middle before attributing production query sources.'))


def analyze():
    result={}
    for backend in ['production','diagnostic']:
        records=[]
        for p in sorted((RUN/backend).glob('*/repeat_*/output_hungarian_repeat.json')):
            comparison=json.loads(p.read_text());row={'token':p.parent.parent.name,'repeat':p.parent.name,'groups':{}}
            for threshold in [0,.25,.5]:
                matches=[m for m in comparison['matched'] if min(m['reference_score'],m['candidate_score'])>=threshold]
                row['groups'][str(threshold)]={'count':len(matches),'metrics':{k:{'p95':float(np.percentile([m[k] for m in matches],95)),'max':max(m[k] for m in matches)} if matches else None for k in ['center_m','size_m','yaw_rad','score_abs','velocity_mps']}}
            records.append(row)
        result[backend]=records
    write(REPORT/'output_repeat_score_groups.json',dict(scope='same-class 2m Hungarian; full candidates retained; score groups explanatory only',results=result))

def build_head():
    marked=RUN/'head.diagnostic.onnx';assert marked.exists()
    command=[str(ROOT/'envs/tensorrt/bin/trtexec'),'--onnx='+str(marked),'--fp16','--inputIOFormats=fp16:chw','--outputIOFormats='+','.join(['fp16:chw']*6+['fp32:chw','fp32:chw','int32:chw']),'--plugins='+str(SRC/'build/libcustom_layernorm.so'),'--saveEngine='+str(RUN/'head.diagnostic.plan'),'--memPoolSize=workspace:2048','--skipInference','--profilingVerbosity=detailed','--exportLayerInfo='+str(RUN/'head.diagnostic.layers.json')]
    original=identity();log=RUN/'head.diagnostic.build.log'
    with log.open('w') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT)
    write(REPORT/'head_diagnostic_build.json',dict(command=command,returncode=result.returncode,log_sha256=sha(log),engine_sha256=sha(RUN/'head.diagnostic.plan') if result.returncode==0 else None,production_identity_unchanged=identity()==original))
    (REPORT/'head_diagnostic_build_tail.log').write_text(''.join(log.read_text().splitlines(True)[-60:]))
    result.check_returncode();assert identity()==original

class Plan:
    """TensorRT binding runner with cudaRuntime ctypes; no PyTorch dependency."""
    def __init__(self,path):
        import tensorrt as trt
        self.trt=trt;self.logger=trt.Logger(trt.Logger.WARNING)
        ctypes.CDLL(str(SRC/'build/libcustom_layernorm.so'),mode=ctypes.RTLD_GLOBAL)
        trt.init_libnvinfer_plugins(self.logger,'')
        self.runtime=trt.Runtime(self.logger);self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes());assert self.engine
        self.context=self.engine.create_execution_context();self.cuda=ctypes.CDLL('libcudart.so')
        self.cuda.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]
        self.cuda.cudaMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
        self.cuda.cudaFree.argtypes=[ctypes.c_void_p]
        self.names=[self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
    def __call__(self,inputs):
        arrays={};buffers=[];pointers=[]
        try:
            for i,name in enumerate(self.names):
                dtype=np.dtype(self.trt.nptype(self.engine.get_binding_dtype(i)))
                if self.engine.binding_is_input(i):
                    a=np.ascontiguousarray(inputs[name],dtype=dtype)
                    declared=tuple(self.engine.get_binding_shape(i))
                    if -1 in declared:assert self.context.set_binding_shape(i,a.shape)
                    else:assert a.shape==declared,('Input shape mismatch',name,a.shape,declared)
                else:a=np.empty(tuple(self.context.get_binding_shape(i)),dtype=dtype)
                arrays[name]=a;p=ctypes.c_void_p();assert self.cuda.cudaMalloc(ctypes.byref(p),a.nbytes)==0
                buffers.append(p);pointers.append(p.value)
                if self.engine.binding_is_input(i):assert self.cuda.cudaMemcpy(p,ctypes.c_void_p(a.ctypes.data),a.nbytes,1)==0
            assert self.context.execute_v2(pointers)
            for i,name in enumerate(self.names):
                if not self.engine.binding_is_input(i):
                    a=arrays[name];assert self.cuda.cudaMemcpy(ctypes.c_void_p(a.ctypes.data),buffers[i],a.nbytes,2)==0
            return {n:a for i,(n,a) in enumerate(arrays.items()) if not self.engine.binding_is_input(i)}
        finally:
            for p in buffers:self.cuda.cudaFree(p)

def pool_replay(values):
    cuda=ctypes.CDLL('libcudart.so');cuda.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t];cuda.cudaMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int];cuda.cudaFree.argtypes=[ctypes.c_void_p]
    lib=ctypes.CDLL(str(SRC/'build/libbevfusion_core.so'));fun=lib.car_bev_pool_forward
    fun.argtypes=[ctypes.c_void_p]*4+[ctypes.c_uint]*7+[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
    features=np.ascontiguousarray(values['camera_feature'],dtype=np.float16);depth=np.ascontiguousarray(values['camera_depth_weights'],dtype=np.float16);indices=np.ascontiguousarray(values['indices'],dtype=np.uint32);intervals=np.ascontiguousarray(values['intervals'],dtype=np.int32)
    n,h,w,c=features.shape;depth_count=depth.shape[1];output=np.empty((1,c,360,360),dtype=np.float16);ptr=[]
    try:
        for a in [features,depth,indices,intervals,output]:
            p=ctypes.c_void_p();assert cuda.cudaMalloc(ctypes.byref(p),a.nbytes)==0;ptr.append(p)
            assert cuda.cudaMemcpy(p,ctypes.c_void_p(a.ctypes.data),a.nbytes,1)==0
        assert fun(*ptr[:4],len(intervals),c,depth_count,h,w,360,360,ptr[4],0,None)==0
        assert cuda.cudaMemcpy(ctypes.c_void_p(output.ctypes.data),ptr[4],output.nbytes,2)==0
        return output
    finally:
        for p in ptr:cuda.cudaFree(p)

def replay(args):
    reference=args.reference or ROOT/'runs/deployment_4090_20261012_followup/reference'
    engines={name:Plan(MODEL/'build'/(name+'.plan')) for name in ['camera.backbone','camera.vtransform','fuser','head.bbox.layernormplugin']}
    diagnostic=Plan(RUN/'head.diagnostic.plan') if (RUN/'head.diagnostic.plan').exists() else None
    summary=[]
    for packet in selected_packets():
        token=packet['sample_token'];d=RUN/'diagnostic'/token/'repeat_0';out=RUN/'replay'/token;out.mkdir(parents=True,exist_ok=True)
        ref=np.load(reference/token/'repeat_00.npz');inp=np.load(RUN/'diagnostic'/token/'inputs.npz')
        # Pair each upstream source with the same production downstream engine.
        # Reference replay must consume these saved half-representable inputs too.
        poolref=ref['camera.bev_pool'];poolref=np.concatenate([poolref[:,:,i] for i in range(poolref.shape[2])],axis=1)
        sparse=ref['camera.sparse_depth'];sparse=sparse.reshape(1,6,1,256,704)
        cases={
          'camera.reference_upstream':('camera.backbone',{'img':inp['images'],'depth':sparse}),
          'camera.trt_upstream':('camera.backbone',{'img':read_binding(d,'img'),'depth':read_binding(d,'depth')}),
          'view.reference_upstream':('camera.vtransform',{'feat_in':poolref}),
          'view.trt_upstream':('camera.vtransform',{'feat_in':read_binding(d,'feat_in')}),
          'fuser.reference_upstream':('fuser',{'camera':ref['camera.view_transform'],'lidar':ref['lidar.bev']}),
          'fuser.trt_upstream':('fuser',{'camera':read_binding(d,'camera'),'lidar':read_binding(d,'lidar')}),
          'head.reference_upstream':('head.bbox.layernormplugin',{'middle':ref['head.input']}),
          'head.trt_upstream':('head.bbox.layernormplugin',{'middle':read_binding(d,'middle')})}
        row={'token':token,'replays':{}}
        for upstream in ['reference','trt']:
            values=dict(camera_feature=np.ascontiguousarray(ref['camera.feature'] if upstream=='reference' else read_binding(d,'camera_feature'),dtype=np.float16),camera_depth_weights=np.ascontiguousarray(ref['camera.depth'] if upstream=='reference' else read_binding(d,'camera_depth_weights'),dtype=np.float16),indices=np.fromfile(d/'pool.indices.bin',dtype=np.uint32),intervals=np.fromfile(d/'pool.intervals.bin',dtype=np.int32).reshape(-1,3))
            case='pool.'+upstream+'_upstream';np.savez(out/(case+'.inputs.npz'),**values);result=pool_replay(values);np.savez(out/(case+'.outputs.npz'),feat_in=result)
            row['replays'][case]={'same_input_repeat':[stats(result,pool_replay(values)) for _ in range(4)],'geometry':'identical captured TRT integer mapping for both implementations'}
            if upstream=='trt':row['replays'][case]['captured_production_vs_replay']=stats(read_binding(d,'feat_in'),result)

        for case,(engine,values) in cases.items():
            values={k:np.ascontiguousarray(v,dtype=np.float16) for k,v in values.items()};np.savez(out/(case+'.inputs.npz'),**values)
            outputs=engines[engine](values);np.savez(out/(case+'.outputs.npz'),**outputs)
            repeated=[engines[engine](values) for _ in range(4)]
            row['replays'][case]={'same_input_repeat':[ {k:stats(outputs[k],o[k]) for k in outputs} for o in repeated]}
            if case.endswith('trt_upstream'):
                row['replays'][case]['captured_production_vs_replay']={k:stats(read_binding(d,k),v) for k,v in outputs.items()}
            if case.startswith('head.') and diagnostic:
                marked=diagnostic(values);np.savez(out/(case+'.marked.npz'),**marked)
                row['replays'][case]['marked_same_input_repeat']=[{k:stats(marked[k],v) for k,v in diagnostic(values).items()} for _ in range(4)]
                row['replays'][case]['marking_perturbation']={k:stats(outputs[k],marked[k]) for k in outputs}
                graph=json.loads((REPORT/'head_diagnostic_export.json').read_text())
                indices=marked[graph['indices']].astype(np.int64);scores=marked[graph['input']]
                vals=marked[graph['values']]
                unselected=scores.copy();np.put_along_axis(unselected,indices,-np.inf,axis=1);cutoff_gap=vals[:,-1]-unselected.max(axis=1)
                np.savez(out/(case+'.query.npz'),pre_topk_scores=scores,source_indices=indices,selected_scores=vals,adjacent_gaps=vals[:,:-1]-vals[:,1:],selection_cutoff_gap=cutoff_gap)
                row['replays'][case]['query_cutoff_gap']=cutoff_gap.tolist()
                sources=ref['query.source_indices'];row['replays'][case]['source_overlap_with_reference']=int(len(set(indices.ravel())&set(sources.ravel())))
        coords=ref['camera.pool.coords'];refcounts=np.bincount((coords[:,0]*360+coords[:,1]).astype(np.int64),minlength=360*360)
        intervals=np.fromfile(d/'pool.intervals.bin',dtype=np.int32).reshape(-1,3);trtcounts=np.zeros(360*360,dtype=np.int64);trtcounts[intervals[:,2]]=intervals[:,1]-intervals[:,0]
        row['geometry_contribution_counts']=dict(reference_total=int(refcounts.sum()),candidate_total=int(trtcounts.sum()),mismatched_cells=int((refcounts!=trtcounts).sum()),max_count_difference=int(np.abs(refcounts-trtcounts).max()))
        row['cross_backend_original']={node:stats((poolref if node=='camera.bev_pool' else ref[node].reshape(read_binding(d,binding).shape) if node=='camera.sparse_depth' else ref[node]),read_binding(d,binding)) for node,binding in NODES.items()}
        summary.append(row);write(REPORT/'module_replay.json',summary)
    write(REPORT/'capture_gaps.json',dict(raw_fuser='Production fuser plan combines fusion+decoder; only combined middle exposed.',topk='Marked-output diagnostic engine changes graph optimization; perturbation metrics govern attribution.' if diagnostic else 'Diagnostic TopK plan not built; production plan does not expose indices.',lidar_internal='LiDAR voxelization and sparse convolution internals not captured; final LiDAR BEV captured.',pool_local='Standalone production car_bev_pool_forward replay inputs saved with explicit captured TRT geometry; cross-backend module comparison requires reference replay.',timing='All diagnostic capture/replay timings excluded from production performance.'))

if __name__=='__main__':main()

