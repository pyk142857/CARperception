#!/usr/bin/env python3
"""Publish measured TRT artifacts; never overwrite preserved failed attempts."""
import argparse,csv,gzip,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=ROOT/'configs/deployment_4090.yaml');a=p.parse_args();cfg=read(a.config);out=ROOT/cfg['report']/'bevfusion_tensorrt';rows=[];parts=[];artifacts=[]
 for split in cfg['splits']:
  run=ROOT/'outputs/bevfusion_tensorrt'/split;dest=out/split;dest.mkdir(parents=True,exist_ok=True)
  manifest=read(run/'manifest.json');perf=read(run/'performance.json');comparison=read(dest/'comparison.json');metric=read(dest/'official/metrics/metrics_summary.json')
  # Do not summarize stale comparison against a newly changed run.
  if comparison['input_hashes']['candidate']!=sha(run/'predictions.json'):raise RuntimeError(f'Stale comparison: {split}')
  for name in ['predictions.json','native_predictions.json','actual_inputs.json','coordinate_audit.json']:
   src=run/name;dst=dest/(name+'.gz')
   with dst.open('wb') as f:
    with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(src.read_bytes())
   artifacts.append({'path':str(dst.relative_to(ROOT)),'sha256':sha(dst),'original_sha256':sha(src)})
  for name in ['manifest.json','performance.json','timings.csv']:shutil.copy2(run/name,dest/name)
  # Keep full pair list compressed, publish easily reviewable aggregates separately.
  with (dest/'comparison.json.gz').open('wb') as f:
   with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write((dest/'comparison.json').read_bytes())
  compact={k:v for k,v in comparison.items() if k not in ['matched','frames']};compact['frame_count']=len(comparison['frames']);compact['matched_box_count']=len(comparison['matched']);(dest/'comparison_summary.json').write_text(json.dumps(compact,indent=2))
  for name,delta in comparison['class_ap_deltas'].items():
   parts.append({'split':split,'class':name,'reference_ap':sum(read(ROOT/cfg['report']/'bevfusion_reference'/split/'evaluation/metrics/metrics_summary.json')['label_aps'][name].values())/4,'candidate_ap':sum(metric['label_aps'][name].values())/4,'delta':delta})
  gpu=perf['timings']['forward_h2d_model_d2h_ms'];e2e=perf['timings']['end_to_end_ms'];row={'model':'bevfusion_trt','backend':'CUDA/TensorRT8.6.1+SpConv','precision':'FP16','split':split,'frames':manifest['frames'],'mAP':metric['mean_ap'],'NDS':metric['nd_score'],'mAP_delta':comparison['metrics']['mean_ap']['delta'],'NDS_delta':comparison['metrics']['nd_score']['delta'],'model_with_transfers_p50_ms':gpu['p50'],'model_with_transfers_p95_ms':gpu['p95'],'e2e_p50_ms':e2e['p50'],'e2e_p95_ms':e2e['p95'],'process_gpu_peak_mib':perf['process_peak_mib'],'fixed_metric_budget_passed':comparison['metric_budget_passed'],'comparison_status':comparison['status']};rows.append(row)
 for name,data in [('metrics.csv',rows),('per_class_ap.csv',parts)]:
  with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
 (out/'published_artifacts.json').write_text(json.dumps(artifacts,indent=2))
 table='\n'.join(f"| {r['split']} | {r['frames']} | {r['mAP']:.6f} | {r['NDS']:.6f} | {r['mAP_delta']:+.6f} | {r['NDS_delta']:+.6f} | {r['model_with_transfers_p50_ms']:.3f} / {r['model_with_transfers_p95_ms']:.3f} | {r['comparison_status']} |" for r in rows)
 status='passed' if all(r['comparison_status']=='passed' for r in rows) else 'partial' if all(r['fixed_metric_budget_passed'] for r in rows) else 'failed'
 report=f'''# CUDA-BEVFusion TensorRT FP16 实测

本次实际执行覆盖 scene-0061 39帧与官方 mini_val 81帧。阶段状态 **{status}**；推理运行、mAP/NDS固定预算和完整几何诊断分别判断。partial表示固定指标预算已通过、几何/数值复核仍有失败项，未改写原comparison.status或任何诊断阈值。

| Split | 帧数 | mAP | NDS | ΔmAP | ΔNDS | forward含传输 P50/P95 ms | 配对状态 |
|---|---:|---:|---:|---:|---:|---:|---|
{table}

精度预算在首轮结果前固定：同split mAP、NDS 各下降不超过 **0.005**，未因结果调整。逐类AP见 [per_class_ap.csv](per_class_ap.csv)，几何/分数/速度分布与预先固定阈值见各split/comparison_summary.json。框按同类、中心距离≤2m的Hungarian匹配诊断；不等同官方检测评估匹配。

[完整指标CSV](metrics.csv)；各split的 predictions.json.gz/native_predictions.json.gz 保存实际标准化/原生输出，不筛除低分。timings.csv 为逐帧计时，performance.json 给出边界与显存；manifest.json 固定配置、engine、ONNX及native库哈希。FP32 CPU预处理与参考共用 pinned pipeline，向CUDA传FP16输入；传输差异不能称纯TensorRT kernel速度。

默认Myelin head 的失败尝试保留于 `attempts/myelin_head/`，其预测、指标与固定预算不覆盖。上游注明该融合可能有LayerNorm问题，另构建 `head.bbox.layernormplugin.plan` 进行定位；最终实际head以当前manifest配置/engine哈希为准，不能仅因采用插件便声称精度问题已解决。

补充的分数分层和五个实际部署差异案例见 [geometry_diagnosis.md](geometry_diagnosis.md)，逐类表格不改变完整配对验收范围。

范围是mini教学诊断，可能与预训练数据重叠，不是官方完整val；属性沿参考真实来源，不用GT补预测。桌面RTX4090时延不代表车端硬件。构建与来源见 [build_report.md](build_report.md)，权重审计见 [weight_audit.json](weight_audit.json)。
'''
 (out/'report.md').write_text(report)
 print(json.dumps({'status':status,'splits':rows},indent=2))
if __name__=='__main__':main()
