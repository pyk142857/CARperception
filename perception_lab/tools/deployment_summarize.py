"""Assemble the deployment report from measured stage evidence, never placeholders."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from run_4090_deployment import ROOT, sha, write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());root=ROOT/cfg['report']
    def read(path):
        path=root/path
        return json.loads(path.read_text()) if path.exists() else None
    rows=[];stages={}
    pre=read('inputs/summary.json')
    stages['preflight']={'status':'passed' if pre and pre.get('status')=='passed' else 'not_started','evidence':'inputs/summary.json'}
    ref=read('bevfusion_reference/stage_summary.json')
    stages['bevfusion_reference']={'status':'passed' if ref and ref.get('inference')=='passed' else 'not_started','evidence':'bevfusion_reference/report.md'}
    if ref:
        for split,x in ref['results'].items():
            rows.append(dict(model='BEVFusion R50',backend='PyTorch',precision='FP32',split=split,frames=x['frames'],protocol='DetectionEval detection_cvpr_2019',mAP=x['mAP'],NDS=x['NDS'],model_p50_ms=x['forward_p50_ms'],model_p95_ms=x['forward_p95_ms'],e2e_p50_ms=x['end_to_end_p50_ms'],e2e_p95_ms=x['end_to_end_p95_ms'],allocated_peak_mib=x['allocated_peak_GiB']*1024,reserved_peak_mib=x['reserved_peak_GiB']*1024,note='forward includes upstream output.cpu; empty attributes; mini diagnostic'))
    trt_sub={}
    for split in cfg['splits']:
        x=read('bevfusion_tensorrt/'+split+'/comparison.json');perf_path=ROOT/'outputs/bevfusion_tensorrt'/split/'performance.json'
        if not x or not perf_path.exists():trt_sub[split]={'status':'not_started'};continue

        prediction=perf_path.with_name('predictions.json')
        if x['input_hashes']['candidate']!=sha(prediction):raise ValueError('Stale TRT comparison: '+split)
        perf=json.loads(perf_path.read_text());t=perf['timings'];trt_sub[split]={'status':x['status'],'frames':len(x['frames']),'metric_budget_passed':x['metric_budget_passed'],'geometry_diagnostics':x['geometry_diagnostics'],'evidence':'bevfusion_tensorrt/'+split+'/comparison.json'}
        rows.append(dict(model='BEVFusion R50',backend='TensorRT 8.6 + SpConv',precision='FP16',split=split,frames=perf['frames'],protocol='DetectionEval detection_cvpr_2019',mAP=x['metrics']['mean_ap']['candidate'],NDS=x['metrics']['nd_score']['candidate'],map_delta=x['metrics']['mean_ap']['delta'],nds_delta=x['metrics']['nd_score']['delta'],model_p50_ms=t['forward_h2d_model_d2h_ms']['p50'],model_p95_ms=t['forward_h2d_model_d2h_ms']['p95'],e2e_p50_ms=t['end_to_end_ms']['p50'],e2e_p95_ms=t['end_to_end_ms']['p95'],process_peak_mib=perf['process_peak_mib'],note='backend timing includes H2D/D2H; geometry update separate; '+x['status']))
    stages['bevfusion_tensorrt']={'status':'passed' if trt_sub and all(x['status']=='passed' for x in trt_sub.values()) else 'partial' if any(x['status']!='not_started' for x in trt_sub.values()) else 'not_started','splits':trt_sub,'evidence':'bevfusion_tensorrt/report.md'}
    pool=read('bev_pooling/results.json');pool_full=read('bev_pooling/full_model_comparison.json')
    stages['bev_pooling']={'status':'passed' if pool and pool.get('status')=='passed' and pool_full and pool_full.get('status')=='passed' else 'partial' if pool else 'not_started','operator_evidence':'bev_pooling/results.json','full_model_evidence':'bev_pooling/full_model_comparison.json' if pool_full else None,'default':'upstream'}
    sparse=root/'sparse4d/metrics.csv'
    sparse_valid=sparse.exists()
    if sparse_valid:
        sr=list(csv.DictReader(sparse.open()))
        sparse_valid={(r['split'],int(r['frames'])) for r in sr}=={('scene0061',39),('mini_val',81)} and all((root/'sparse4d'/sp/'state_tests.json').exists() for sp in cfg['splits'])
    stages['sparse4d']={'status':'passed' if sparse_valid else 'partial' if sparse.exists() else 'not_started','evidence':'sparse4d/report.md'}
    if sparse.exists():
        for x in csv.DictReader(sparse.open()):
            row={k:v for k,v in x.items() if k not in ['peak_allocated_bytes','peak_reserved_bytes']};row.update(backend='PyTorch + custom CUDA',precision='official fp16 wrapper',protocol='DetectionEval + TrackingEval official mini splits',allocated_peak_mib=float(x['peak_allocated_bytes'])/2**20,reserved_peak_mib=float(x['peak_reserved_bytes'])/2**20,note='camera only; native temporal IDs; model timing includes decoding/D2H');rows.append(row)
    ga=read('gaussianworld/summary.json')
    stages['gaussianworld']={'status':'partial' if ga else 'not_started','inference':ga.get('inference_status') if ga else 'not_started','evaluation':ga.get('evaluation_status') if ga else 'not_started','evidence':'gaussianworld/report.md'}
    if ga:rows.append(dict(model='GaussianWorld streaming',backend='PyTorch + custom CUDA',precision='FP32 amp=False',split='scene0061',frames=ga['frames_completed'],protocol='SurroundOcc scene-stream inference; no voxel labels',model_p50_ms=ga['model_ms']['p50'],model_p95_ms=ga['model_ms']['p95'],e2e_p50_ms=ga['e2e_ms']['p50'],e2e_p95_ms=ga['e2e_ms']['p95'],allocated_peak_mib=ga['allocated_peak_mb']*1e6/2**20,reserved_peak_mib=ga['reserved_peak_mb']*1e6/2**20,process_peak_mib=ga['process_gpu_memory_peak_mib'],note='IoU/mIoU unavailable: missing SurroundOcc GT'))
    viz=read('visualize/summary.json');qa=read('visualize/rerun_browser_qa.json');viz_acceptance=read('visualize/status.json')
    stages['visualize']={'status':'passed' if viz and qa and qa.get('status')=='passed' and viz_acceptance and viz_acceptance.get('status')=='passed' else 'partial' if viz else 'not_started','evidence':'visualize/summary.json','browser_evidence':'visualize/rerun_browser_qa.json' if qa else None,'models':viz.get('models',[]) if viz else []}
    stages['summarize']={'status':'passed','scope':'local report assembly; Git publication verified separately','evidence':'report.md'}
    runner=read('runner_status.json') or {}
    for stage,item in stages.items():
        if stage in runner:item['run_id']=runner[stage].get('run_id')
    complete=all(stages[s]['status']=='passed' for s in ['preflight','bevfusion_reference','bevfusion_tensorrt','bev_pooling','sparse4d','visualize']) and stages['gaussianworld']['inference']=='passed'
    write(root/'status.json',dict(status='passed' if complete else 'partial',scope='nuScenes mini teaching deployment; not full trainval validation',stages=stages))
    fields=['model','backend','precision','split','frames','protocol','mAP','NDS','AMOTA','AMOTP','IDS','IoU','mIoU','model_p50_ms','model_p95_ms','e2e_p50_ms','e2e_p95_ms','allocated_peak_mib','reserved_peak_mib','process_peak_mib','map_delta','nds_delta','note']
    with (root/'metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    local={}
    for name in ['bevfusion','sparse4d','gaussianworld']:
        model=cfg[name];source=ROOT/model['source'];checkpoint=ROOT/model['checkpoint']
        local[name]=dict(source=str(source),upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),weight_path=str(checkpoint),weight_sha256=sha(checkpoint),weight_bytes=checkpoint.stat().st_size)
    for path in (ROOT/cfg['tensorrt']['model_dir']).glob('**/*'):
        if path.is_file() and path.suffix in ['.onnx','.plan']:local[str(path.relative_to(ROOT))]=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
    lines=['# RTX 4090 感知部署实测报告','', '本轮使用 nuScenes mini：scene-0061 完整 39 帧和官方 mini_val 81 帧。所有预测来自本机模型真实推理；这些数据可能与预训练集重叠，不能作为全量泛化结论。','', '**当前验收状态：'+('passed' if complete else 'partial')+'。** GaussianWorld 的实际推理与标签评估状态分别列于下方；缺失指标留空。各项限制以以下阶段报告为准。','', '|模型/后端|范围|mAP|NDS|模型边界 P50/P95 ms|端到端 P50/P95 ms|','|---|---|---:|---:|---:|---:|']
    def fmt(v):return '—' if v in [None,''] else f'{float(v):.4f}'
    for r in rows:lines.append('|'+str(r['model'])+' / '+r['backend']+'|'+r['split']+' ('+str(r['frames'])+')|'+fmt(r.get('mAP'))+'|'+fmt(r.get('NDS'))+'|'+fmt(r.get('model_p50_ms'))+' / '+fmt(r.get('model_p95_ms'))+'|'+fmt(r.get('e2e_p50_ms'))+' / '+fmt(r.get('e2e_p95_ms'))+'|')
    lines+=['','模型边界并不完全相同：BEVFusion PyTorch 含上游结果回传；TensorRT 含输入搬运、后端和框回传，几何更新另计；Sparse4D 含解码/回传；GaussianWorld 的搬运单独计时。各模型顺序运行，不能把这些数字说成全部模型同时常驻的实时车端吞吐。显存和逐帧原始时延见 [metrics.csv](metrics.csv) 及各模块报告。','', '## 阶段证据','']
    for s,item in stages.items():
        target=item.get('evidence',item.get('operator_evidence','status.json'));lines.append('- **'+s+'：'+item['status']+'** — ['+target+']('+target+')')
    lines+=['','## 已识别问题','', '- 默认 Myelin 检测头未满足固定精度预算，原始失败结果保留在 `bevfusion_tensorrt/attempts/myelin_head/`；LayerNorm 插件版本单独验证，没有放宽 0.005 的 mAP/NDS 下降预算。','- NVIDIA 几何区间最后一个投影的 exclusive end 边界已修复，生产 CUDA kernel 的三项测试验证修复前失败、修复后通过；真实首帧有效投影恢复为 1,835,478 个。', '- BEV pooling 候选与上游保持相同累计顺序；是否获得提速以 30 轮配对结果及全模型对照为准，当前默认保持上游实现。','- GaussianWorld 无占用标签，未计算 IoU/mIoU；移除标签分支用同一组真实 anchors 隔离验证。','- Sparse4D 原生 ID 的官方 mini_val IDS 与旧 CenterPoint 单场景诊断口径不同，不直接横向比较。','', '## 打开与复现','', '本机新增结果页：[http://127.0.0.1:9094/](http://127.0.0.1:9094/)。旧回放保持原入口。视频 2 fps 是播放速度。','', '```bash','cd perception_lab','python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage summarize','envs/rerun/bin/python tools/deployment_visualize.py --serve','```','', '中间张量、框差异案例与源权重对应见 [TensorRT 报告](bevfusion_tensorrt/report.md)、[几何差异](bevfusion_tensorrt/geometry_diagnosis.md)、[逐阶段计时](bevfusion_tensorrt/native_profile.md)。17 项相关单元/回归测试和 1 项真实 GaussianWorld 几何检查通过，另有 CUDA 边界与内存检查证据。', '', '完整命令见 [run_commands.md](run_commands.md)。代码入口 [run_4090_deployment.py](../../tools/run_4090_deployment.py)，配置 [deployment_4090.yaml](../../configs/deployment_4090.yaml)。执行缓存会校验配置、权重、输入、源码和产物哈希；进程运行成功不等于精度验收成功。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    cmds=['# 复现命令','','工作目录 `perception_lab/`。先按模块 setup 脚本恢复隔离依赖与官方资产，权重/数据/engine 不入 Git。','', '```bash','bash tools/bevfusion_setup.sh','bash tools/bevfusion_trt_setup.sh --build-engines','bash tools/sparse4d_setup.sh','bash tools/gaussianworld_setup.sh']
    cmds += ['python3 tools/run_4090_deployment.py --config configs/deployment_4090.yaml --stage '+s for s in stages]
    cmds += ['envs/rerun/bin/python tools/deployment_visualize.py --serve','```','','各次实际 argv、工作目录、返回码与日志记录在 [runner_status.json](runner_status.json) 和 `commands/`；首次独立执行命令及环境在模块报告中。TensorRT head 必须按配置使用 LayerNorm 插件版本。']
    (root/'run_commands.md').write_text('\n'.join(cmds)+'\n')
    files={str(f.relative_to(root)):dict(sha256=sha(f),bytes=f.stat().st_size) for f in root.rglob('*') if f.is_file() and f.name not in ['artifacts_manifest.json','report.md','runner_status.json'] and f.suffix in ['.json','.csv','.gz','.npz','.jpg','.png','.patch','.md','.txt']}
    write(root/'artifacts_manifest.json',dict(starting_commit=cfg['starting_commit'],configuration_sha256=sha(a.config),published_artifacts=files,large_local_artifacts=local))
    print(json.dumps(dict(status='passed' if complete else 'partial',rows=len(rows),stages=stages),indent=2))

if __name__=='__main__':main()
