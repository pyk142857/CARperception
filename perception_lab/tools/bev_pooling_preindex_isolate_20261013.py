#!/usr/bin/env python3
"""Prepare reviewable isolated full-model patches; no build or GPU execution.

Default writes only diffs/plan. --materialize additionally copies the small
current src tree and tooling to a new run directory; refuses existing targets.
Use only if the independently measured preindex candidate has stable benefit.
"""
import argparse, ast, difflib, hashlib, json, shutil
from pathlib import Path
LAB=Path(__file__).resolve().parents[1]
UPSTREAM=LAB/'third_party/Lidar_AI_Solution/CUDA-BEVFusion'
BASE=LAB/'runs/tracking_gpu_20261013/bev_pooling/isolated'
DEST=BASE/'CUDA-BEVFusion'
REPORT=LAB/'reports/tracking_gpu_20261013/bev_pooling'

def replace_once(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def patch_source():
    original=(UPSTREAM/'src/bevfusion/camera-bevpool.cu').read_text();s=original
    declaration='''extern "C" int car_bev_pool_preindex_forward(
    const void*, const void*, const void*, const void*, unsigned int,
    unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, void*, void*, void*);

'''
    s=replace_once(s,'namespace bevfusion {',declaration+'namespace bevfusion {')
    s=replace_once(s,'    if (output_feature_) checkRuntime(cudaFree(output_feature_));','    if (output_feature_) checkRuntime(cudaFree(output_feature_));\n    if (preindex_offsets_) checkRuntime(cudaFree(preindex_offsets_));')
    s=replace_once(s,'    checkRuntime(cudaMalloc(&output_feature_, volumn_output_ * sizeof(nvtype::half)));','''    checkRuntime(cudaMalloc(&output_feature_, volumn_output_ * sizeof(nvtype::half)));
    preindex_count_ = camera_shape[0] * camera_shape[2] * camera_shape[3] * camera_shape[4];
    checkRuntime(cudaMalloc(&preindex_offsets_, preindex_count_ * sizeof(unsigned int)));''')
    old='''    checkRuntime(static_cast<cudaError_t>(car_bev_pool_forward(
        camera_feature, depth_weights, indices, intervals, num_intervals, C, D, H, W,
        bev_height_, bev_width_, output_feature_, candidate, stream)));'''
    new='''    if (mode && std::string(mode) == "preindex") {
      checkRuntime(static_cast<cudaError_t>(car_bev_pool_preindex_forward(
          camera_feature, depth_weights, indices, intervals, num_intervals, C, D, H, W,
          bev_height_, bev_width_, preindex_count_, preindex_offsets_, output_feature_, stream)));
    } else {
      checkRuntime(static_cast<cudaError_t>(car_bev_pool_forward(
          camera_feature, depth_weights, indices, intervals, num_intervals, C, D, H, W,
          bev_height_, bev_width_, output_feature_, candidate, stream)));
    }'''
    s=replace_once(s,old,new)
    s=replace_once(s,'  half* output_feature_ = nullptr;','  half* output_feature_ = nullptr;\n  unsigned int* preindex_offsets_ = nullptr;\n  unsigned int preindex_count_ = 0;')
    return original,s

def patch_runner():
    path=LAB/'tools/bevfusion_trt_4090.py';original=path.read_text();s=original
    s=replace_once(s,'from bevfusion_4090 import',f'sys.path.insert(0, {str(LAB/"tools")!r})\nfrom bevfusion_4090 import')
    s=replace_once(s,'ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(LAB)!r})')
    s=replace_once(s,"    build_dir = ROOT / tcfg['source'] / 'CUDA-BEVFusion/build'",f'    build_dir = Path({str(DEST/"build")!r})')
    s=replace_once(s,'    a.out.mkdir(parents=True, exist_ok=True)',f'''    permitted = ROOT/'reports/tracking_gpu_20261013/bev_pooling/full_model'
    if not a.out.resolve().is_relative_to(permitted.resolve()):
        raise ValueError('Isolated pooling output must stay below '+str(permitted))
    if os.environ.get('CAR_BEV_POOLING','upstream') not in ('upstream','preindex'):
        raise ValueError('This experiment only permits upstream or preindex')
    a.out.mkdir(parents=True, exist_ok=True)''')
    s=replace_once(s,'    import libpybev','''    import libpybev
    if Path(libpybev.__file__).resolve().parent != build_dir.resolve():
        raise RuntimeError('Wrong Python extension loaded: '+libpybev.__file__)
    mapped = Path('/proc/self/maps').read_text()
    core_paths = {line.split()[-1] for line in mapped.splitlines() if 'libbevfusion_core.so' in line}
    if core_paths != {str((build_dir/'libbevfusion_core.so').resolve())}:
        raise RuntimeError('Wrong core library mapping: '+repr(core_paths))''')
    s=replace_once(s,'        from bev_pooling_full_model import paired_forwards','        from bev_pooling_preindex_full_model import paired_forwards')
    s=replace_once(s,"        paired_forwards(core, *first, a.paired_pooling_rounds, cfg_deploy['pooling']['warmup'], a.out)",'''        # Stateless detector: each scene's first/middle/last actual frame.
        for scene in dict.fromkeys(p['scene_token'] for p in packets):
            seq = [p for p in packets if p['scene_token']==scene]
            for idx in [0,len(seq)//2,len(seq)-1]:
                pair_out = a.out/'paired'/seq[idx]['sample_token']
                pair_out.mkdir(parents=True,exist_ok=True)
                prepared = prepare(seq[idx])
                paired_forwards(core,*prepared,a.paired_pooling_rounds,cfg_deploy['pooling']['warmup'],pair_out)''')
    ast.parse(s)
    return original,s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--materialize',action='store_true');a=p.parse_args()
    REPORT.mkdir(parents=True,exist_ok=True)
    pairs={'src/bevfusion/camera-bevpool.cu':patch_source(),'tooling/bevfusion_preindex_run.py':patch_runner()}
    helper=LAB/'tools/bev_pooling_full_model.py'
    # Keep upstream/candidate output-column names while routing candidate to new switch.
    old=helper.read_text();new=old.replace("os.environ['CAR_BEV_POOLING'] = mode","os.environ['CAR_BEV_POOLING'] = 'preindex' if mode == 'candidate' else 'upstream'")
    assert old.count("os.environ['CAR_BEV_POOLING'] = mode")==2
    pairs['tooling/bev_pooling_preindex_full_model.py']=(old,new);ast.parse(new)
    for name,(before,after) in pairs.items():
        diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+name,tofile='b/'+name))
        (REPORT/(Path(name).name+'.patch')).write_text(diff)
    if a.materialize:
        if BASE.exists():raise FileExistsError('Refuse to overwrite isolated tree '+str(BASE))
        DEST.mkdir(parents=True)
        shutil.copytree(UPSTREAM/'src',DEST/'src')
        shutil.copy2(UPSTREAM/'CMakeLists.txt',DEST/'CMakeLists.txt')
        for dependency in ['libraries','dependencies']:
            (BASE/dependency).symlink_to((UPSTREAM.parent/dependency).resolve(),target_is_directory=True)
        (DEST/'tooling').mkdir()
        for name,(_,after) in pairs.items():(DEST/name).write_text(after)
        shutil.copy2(LAB/'tools/bev_pooling_preindex_20261013.cu',DEST/'src/bevfusion/car-preindex-20261013.cu')
    (REPORT/'isolated_integration_plan.json').write_text(json.dumps({'status':'materialized_not_built' if a.materialize else 'patches_prepared_not_materialized',
        'isolated_source':str(DEST),'isolated_build':str(DEST/'build'),'old_source_untouched':True,
        'old_library_sha256':sha(UPSTREAM/'build/libbevfusion_core.so'),'old_python_library_sha256':sha(UPSTREAM/'build/libpybev.so'),
        'old_camera_source_sha256':sha(UPSTREAM/'src/bevfusion/camera-bevpool.cu'),
        'patches':{Path(k).name:hashlib.sha256(v[1].encode()).hexdigest() for k,v in pairs.items()},
        'configuration':'Exact existing deployment_4090.yaml; no config/checkpoint/engine rebuild or alteration',
        'full_model_measurement':'30 alternating upstream/preindex pairs for first/middle/last of all3scenes, resident identical core; CPU prepare outside, H2D/forward/D2H included',
        'actual_execution':'No build, no CUDA calls, no model/eval execution performed by this script'},indent=2)+'\n')

if __name__=='__main__':main()
