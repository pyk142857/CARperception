"""Audited stage runner. Reuse requires identical inputs/config/code and outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STAGES=['preflight','bevfusion_reference','bevfusion_tensorrt','bev_pooling','sparse4d','gaussianworld','visualize','summarize']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def commands(config,stage,config_path):
    env=config['environments'];report=config['report'];b=config['bevfusion']
    def cmd(script,*args,python=None):return [python or env.get(stage,sys.executable),'tools/'+script,*map(str,args)]
    result=[]
    if stage=='preflight':result=[cmd('deployment_preflight.py','--out',report+'/inputs')]
    elif stage in ['bevfusion_reference','bevfusion_tensorrt']:
        if stage=='bevfusion_tensorrt':
            names=['camera.backbone','camera.vtransform','fuser',config['tensorrt'].get('head_engine','head.bbox')]
            if any(not (ROOT/config['tensorrt']['model_dir']/'build'/(n+'.plan')).exists() for n in names):result.append(['bash','tools/bevfusion_trt_setup.sh','--build-engines'])
        for split,packets in config['splits'].items():
            out=report+'/bevfusion_reference/'+split if stage=='bevfusion_reference' else 'outputs/bevfusion_tensorrt/'+split
            if stage=='bevfusion_reference':
                result.append(cmd('bevfusion_4090.py','--source',b['source'],'--model-config',b['config'],'--checkpoint',b['checkpoint'],'--dataroot',config['data'],'--packets',packets,'--out',out))
            else:
                run=cmd('bevfusion_trt_4090.py','--config',config_path,'--packets',packets,'--out',out)
                if split=='scene0061':run+=['--capture-pooling','runs/deployment_4090_20261008/trt_pooling_captures']
                result.append(['bash','-c','source tools/bevfusion_trt_env.sh\nexec '+shlex.join(run)])
            eval_out=out+'/evaluation' if stage=='bevfusion_reference' else report+'/bevfusion_tensorrt/'+split+'/official'
            evaluation=cmd('bevfusion_evaluate.py','--predictions',out+'/predictions.json','--dataroot',config['data'],'--out',eval_out,'--eval-set','mini_scene_0061' if split=='scene0061' else 'mini_val',python=env['evaluate'])
            if split=='scene0061':evaluation+=['--scene','scene-0061']
            result.append(evaluation)
            if stage=='bevfusion_tensorrt':result.append(cmd('bevfusion_compare.py','--reference',report+'/bevfusion_reference/'+split+'/predictions.json','--candidate',out+'/predictions.json','--reference-metrics',report+'/bevfusion_reference/'+split+'/evaluation/metrics/metrics_summary.json','--candidate-metrics',eval_out+'/metrics/metrics_summary.json','--config',config_path,'--out',report+'/bevfusion_tensorrt/'+split+'/comparison.json',python=env['evaluate']))
        if stage=='bevfusion_tensorrt':result.append(cmd('deployment_trt_report.py',python=env['evaluate']))
    elif stage=='sparse4d':
        for split,packets in config['splits'].items():
            out='outputs/sparse4d/'+split
            result+=[cmd('sparse4d_4090.py','--packets',packets,'--out',out,'--split',split,'--state-tests'),cmd('sparse4d_audit.py','--out',out,python=env['evaluate']),cmd('sparse4d_evaluate.py','--out',out,'--split','mini_scene_0061' if split=='scene0061' else 'mini_val',python=env['sparse4d'])]
        result.append(cmd('sparse4d_cases.py','--out','outputs/sparse4d/scene0061',python=env['sparse4d']))
        result.append(cmd('sparse4d_summarize.py',python=env['evaluate']))
    elif stage=='gaussianworld':
        out='runs/deployment_4090_20261008/gaussianworld'
        result=[cmd('gaussianworld_4090.py','--packets',config['splits']['scene0061'],'--out',out,'--validate-label-independence'),cmd('gaussianworld_summarize.py','--run',out,'--report',report+'/gaussianworld',python=env['evaluate'])]
    elif stage=='bev_pooling':
        result=[['bash','-c','source tools/bevfusion_trt_env.sh\nexec '+shlex.join(cmd('bev_pooling_4090.py','--capture','runs/deployment_4090_20261008/trt_pooling_captures/frame_0','--capture','runs/deployment_4090_20261008/trt_pooling_captures/frame_19','--capture','runs/deployment_4090_20261008/trt_pooling_captures/frame_38','--out',report+'/bev_pooling'))]]
        for split,packets in config['splits'].items():
            out='outputs/bevfusion_tensorrt_candidate/'+split
            run=cmd('bevfusion_trt_4090.py','--config',config_path,'--packets',packets,'--out',out,python=env['bevfusion_tensorrt'])
            if split=='scene0061':run+=['--paired-pooling-rounds','30']
            result.append(['bash','-c','source tools/bevfusion_trt_env.sh\nexport CAR_BEV_POOLING=candidate\nexec '+shlex.join(run)])
            evaluation=cmd('bevfusion_evaluate.py','--predictions',out+'/predictions.json','--dataroot',config['data'],'--out',report+'/bev_pooling/full_model/'+split+'/official','--eval-set','mini_scene_0061' if split=='scene0061' else 'mini_val',python=env['evaluate'])
            if split=='scene0061':evaluation+=['--scene','scene-0061']
            result.append(evaluation)
        result.append(cmd('bev_pooling_full_model.py','--config',config_path,python=env['evaluate']))
        result.append(cmd('bev_pooling_report.py',python=env['evaluate']))
    elif stage=='visualize':
        run=cmd('deployment_visualize.py')
        for model,path in [('bevfusion_reference',report+'/bevfusion_reference/scene0061/predictions.json'),('bevfusion_tensorrt','outputs/bevfusion_tensorrt/scene0061/predictions.json'),('sparse4d','outputs/sparse4d/scene0061/predictions.json')]:
            if (ROOT/path).exists():run+=['--prediction',model+'='+path]
        result=[run,cmd('deployment_visualize.py','--serve'),['/usr/bin/python3','tools/deployment_visualize_qa.py'],['/usr/bin/python3','tools/deployment_rerun_qa.py'],cmd('deployment_visualize_report.py',python=env['evaluate'])]
    elif stage=='summarize':result=[cmd('deployment_summarize.py','--config',config_path,python=sys.executable)]
    return result

def fingerprint(config,stage,config_path):
    paths={config_path,Path(__file__)}
    paths.update((ROOT/'tools').glob('*4090*'))
    for name in ['bevfusion_compare.py','deployment_visualize.py','deployment_preflight.py']:
        if (ROOT/'tools'/name).exists():paths.add(ROOT/'tools'/name)
    if stage!='preflight':
        for packet_path in config['splits'].values():
            paths.add(ROOT/packet_path)
            if (ROOT/packet_path).exists():
                for packet in json.loads((ROOT/packet_path).read_text()):
                    paths.update(Path(s['path']) for s in list(packet['sensors'].values())+packet['lidar_sweeps'])
    models=['bevfusion'] if stage.startswith('bev') else [stage] if stage in ['sparse4d','gaussianworld'] else []
    extra={'runtime_environment':{k:os.environ.get(k) for k in ['CAR_BEV_POOLING','CUDA_VISIBLE_DEVICES']}}
    for name in models:
        model=config[name];paths.add(ROOT/model['checkpoint'])
        if 'config' in model:paths.add(ROOT/model['config'])
        source=ROOT/model['source']
        extra[name]=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),diff=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=source)).hexdigest())
    if stage in ['bevfusion_tensorrt','bev_pooling']:
        paths.update((ROOT/config['tensorrt']['model_dir']/'build').glob('*.plan'))
        paths.update((ROOT/config['tensorrt']['source']/'CUDA-BEVFusion/build').glob('*.so'))
    python=ROOT/config['environments'].get(stage,'envs/nuscenes_eval/bin/python')
    if python.exists():extra['environment_freeze_sha256']=hashlib.sha256(subprocess.check_output([str(python),'-m','pip','freeze'],cwd=ROOT)).hexdigest()
    files={str(p.resolve()):sha(p) for p in sorted(paths) if p.is_file()}
    return hashlib.sha256(json.dumps(dict(files=files,extra=extra),sort_keys=True).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=STAGES,required=True);p.add_argument('--force',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    a.config=a.config.resolve();os.chdir(ROOT);config=json.loads(a.config.read_text());report=ROOT/config['report'];status_path=report/'runner_status.json'
    runs=commands(config,a.stage,a.config)
    if a.dry_run:
        print('\n'.join(shlex.join(run) for run in runs));return
    fp=fingerprint(config,a.stage,a.config)
    state=json.loads(status_path.read_text()) if status_path.exists() else {}
    previous=state.get(a.stage,{})
    if not a.force and previous.get('fingerprint')==fp and previous.get('status') in ['passed','partial'] and previous.get('artifacts') and all(Path(k).exists() and sha(k)==v for k,v in previous['artifacts'].items()):
        print('Reused verified measured run:',previous['run_id']);return
    run_id=time.strftime('%Y%m%dT%H%M%S')+'_'+a.stage
    record=dict(status='running',run_id=run_id,fingerprint=fp,commands=[],started=time.time());state[a.stage]=record;write(status_path,state)
    code=0
    for i,run in enumerate(runs):
        log=report/'commands'/(run_id+'_%02d.log'%i);log.parent.mkdir(parents=True,exist_ok=True)
        print(shlex.join(run),flush=True)
        with log.open('w') as f:
            result=subprocess.run(run,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        entry=dict(argv=run,cwd=str(ROOT),returncode=result.returncode,log=str(log),log_sha256=sha(log));record['commands'].append(entry);write(status_path,state)
        if result.returncode:code=result.returncode;break
    record.update(status='failed' if code else 'passed',finished=time.time())
    # A successful process is execution success, not an accuracy/label acceptance claim.
    record['scope']='CLI execution; acceptance lives in stage reports/status.json'
    folders=[report/a.stage]
    if a.stage=='preflight':folders=[report/'inputs']
    if a.stage=='sparse4d':folders+=[ROOT/'outputs/sparse4d']
    if a.stage=='gaussianworld':folders+=[ROOT/'runs/deployment_4090_20261008/gaussianworld']
    if a.stage=='bevfusion_tensorrt':folders+=[ROOT/'outputs/bevfusion_tensorrt']
    if a.stage=='visualize':folders+=[ROOT/'outputs/rerun/deployment_4090']
    record['artifacts']={str(f):sha(f) for folder in folders if folder.exists() for f in folder.rglob('*') if f.is_file() and f.suffix in ['.json','.csv','.npz','.rrd','.mp4','.jpg','.png']}
    write(status_path,state)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
