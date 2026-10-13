"""Aggregate current measured evidence; stale metrics never imply successful execution."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('deimv2', 'harpnext', 'mambafusion', 'grae3dmot')
READY = ('passed', 'partial')


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def collect(out):
    states = {m: read(out/m/'status.json', {'status': 'planned'}) for m in MODELS}
    rows, evidence, blockers = [], [], []

    def metric(model, scope, n, key, value, source='new_measured'):
        rows.append(dict(model=model, scope=scope, samples=n, metric=key, value=value, source=source))

    for backend, folder in [('native', out/'deimv2'), ('onnx', out/'deimv2/onnx'),
                            ('tensorrt_fp16', out/'deimv2/tensorrt')]:
        status = read(folder/'status.json', {})
        metrics = read(folder/'official_eval/metrics.json', {})
        ids = read(folder/'image_ids.json', [])
        protocol = read(folder/'official_eval/protocol.json', {})
        n = status.get('coco_images') if backend == 'native' else status.get('images')
        valid = (status.get('status') in READY and metrics and ids and len(ids) == len(set(ids))
                 and n == len(ids) and protocol.get('images') == len(ids))
        if valid:
            for key, value in metrics.items():
                metric('DEIMv2-S '+backend, 'COCO val2017', len(ids), key, value)
            evidence.append(f'DEIMv2-S {backend}: {len(ids)} images; AP={metrics.get("AP")}')
        elif metrics:
            blockers.append('DEIMv2-S '+backend+': saved metrics excluded because current status/scope evidence is incomplete')

    grae = states['grae3dmot']
    if (out/'grae3dmot/metrics.json').exists():
        for row in read(out/'grae3dmot/metrics.json', []):
            family = row.get('input', 'frozen_reference')
            family_state=read(out/'grae3dmot'/family/'status.json',grae if grae.get('input_family',family)==family else {})
            if not (family_state.get('status') in READY and family_state.get('inference')=='passed' and family_state.get('official_evaluation')=='passed'):
                blockers.append('GRAE '+family+': current family evaluation not complete')
                continue
            folder = out/'grae3dmot'/family/row['split']
            official = read(folder/'official_eval/metrics_summary.json', {})
            tracking = read(folder/'tracking.json', {}).get('results', {})
            bindings=[('tracking_sha256',folder/'tracking.json'),('metrics_sha256',folder/'official_eval/metrics_summary.json'),
                      ('input_manifest_sha256',out/'grae3dmot'/family/'input_manifest.json')]
            if any(not path.is_file() or row.get(key)!=hashlib.sha256(path.read_bytes()).hexdigest() for key,path in bindings):
                blockers.append('GRAE '+family+'/'+row['split']+': evaluation artifact hash binding missing or changed')
                continue
            keys = ['amota', 'amotp', 'ids', 'fp', 'fn']
            if len(tracking) != row['frames'] or not all(row.get(k) == official.get(k) and k in official for k in keys):
                blockers.append('GRAE '+family+'/'+row['split']+': inconsistent official metrics or frame coverage')
                continue
            for key in keys:
                metric('GRAE-3DMOT', row['split']+' '+family, row['frames'], key, row[key])
            evidence.append(f'GRAE {family}/{row["split"]}: {row["frames"]} frames; AMOTA={row["amota"]}; IDS={row["ids"]}')

    # HARP emits these exact original-point metrics only after genuine full-network inference.
    harp = states['harpnext']
    if harp.get('status') in READY and harp.get('inference_frames', 0) > 0:
        for split, result in read(out/'harpnext/metrics.json', {}).items():
            for key in ['miou_union_gt_zero', 'accuracy', 'valid_points']:
                if key in result:
                    metric('HARP-NeXt', split+' original-points-only', result['frame_count'], key, result[key])
    # Mamba native explicitly leaves official_metrics null until DetectionEval is performed.
    mamba = states['mambafusion']
    if mamba.get('status') in READY and mamba.get('frames_inferred', 0) > 0:
        for split in ['scene0061', 'mini_val']:
            folder = out/'mambafusion'/split/'official_eval'
            result = read(folder/'metrics_summary.json', {})
            receipt=mamba.get('official_metrics') or {}
            bound=receipt.get(split,{}) if isinstance(receipt,dict) else {}
            predpath=out/'mambafusion'/(split+'_predictions.json')
            valid=(mamba.get('official_evaluation')=='passed' and bound and predpath.exists()
                   and bound.get('prediction_sha256')==hashlib.sha256(predpath.read_bytes()).hexdigest()
                   and bound.get('metrics_sha256')==hashlib.sha256((folder/'metrics_summary.json').read_bytes()).hexdigest()) if result else False
            if valid:
                for key in ['mean_ap', 'nd_score']:
                    if key in result:
                        metric('MambaFusion', split, 39 if split == 'scene0061' else 81, key, result[key])
                for key,value in result.get('tp_errors',{}).items():
                    metric('MambaFusion',split,39 if split=='scene0061' else 81,key,value)
    for model, status in states.items():
        if status.get('status') not in READY:
            blockers.append(model+': '+status.get('status', 'planned')+'; '+str(status.get('reason', status.get('blockers', status.get('error', 'no current completed result')))))
    if not (out/'grae3dmot/mambafusion_native/mini_val/tracking.json').exists():
        blockers.append('GRAE MambaFusion input branch has no measured mini_val tracking output')
    return states, rows, evidence, blockers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/perception_2025.yaml')
    args = parser.parse_args()
    path = Path(args.config)
    config = read(path if path.is_absolute() else ROOT/path)
    out = ROOT/config['report']
    out.mkdir(parents=True, exist_ok=True)
    write(out/'summarize/status.json', {'status': 'running'})
    states, rows, evidence, blockers = collect(out)
    visualization = read(out/'visualization/status.json', {'status': 'planned'})
    aggregate = 'passed' if all(s.get('status') == 'passed' for s in states.values()) and visualization.get('status') == 'passed' and not blockers else 'partial'
    write(out/'metrics.json', rows)
    with (out/'metrics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'scope', 'samples', 'metric', 'value', 'source'])
        writer.writeheader()
        writer.writerows(rows)
    write(out/'status.json', dict(status=aggregate, scope='2025 candidates; only explicitly verified sample scopes below', models=states,
        visualization=visualization.get('status'), source='new_measured; frozen_reference detection inputs are reused_archive', blockers=blockers, training=False))
    resources = {}
    for model in MODELS:
        manifest = out/model/'resource_manifest.json'
        entry = config.get('models', {}).get(model, {})
        resources[model] = dict(manifest=str(manifest.relative_to(out)), sha256=hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
                               source_commit=entry.get('commit'), checkpoint=entry.get('checkpoint'))
    write(out/'resource_manifest.json', dict(models=resources, input_manifests=['inputs/coco_files.json', 'inputs/lidarseg_files.json', 'inputs/scene0061_packets.json', 'inputs/mini_val_packets.json'],
         policy='Raw data, weights, environments and third-party source remain local; linked manifests record provenance.'))
    (out/'environment.md').write_text('# 本轮环境证据\n\n环境版本与本机实际运行范围以各阶段记录为准；摘要生成不执行模型或修改环境。\n\n'
        '- DEIM native：[环境](deimv2/environment_freeze.txt)、[资源](deimv2/resource_manifest.json)。\n'
        '- ONNX：[实际 provider、TF32及模型哈希](deimv2/onnx/io.json)。\n'
        '- TensorRT：[构建版本及参数](deimv2/tensorrt/build.json)。\n'
        '- HARP：[报告](harpnext/report.md)。Mamba：[环境](mambafusion/environment.json)。GRAE：[环境](grae3dmot/environment.md)。\n')
    table = '\n'.join('| '+model+' | '+states[model].get('status', 'planned')+' |' for model in MODELS)
    measured = '\n'.join('- '+item for item in evidence) or '当前没有满足状态和范围核验条件的检测/跟踪指标。'
    blocked = '\n'.join('- '+item for item in blockers) or '无摘要层面阻塞；仍以各模块完整验收为准。'
    (out/'report.md').write_text(f'''# 2025 感知部署实测报告

本轮汇总状态 **{aggregate}**。以下指标只读取当前完成状态及匹配范围的评价证据；失败或未完成阶段的旧指标不会自动作为本次成功结果。

| 模块 | 当前状态 |
|---|---|
{table}

## 当前实测证据

{measured}

[完整指标 CSV](metrics.csv) · [JSON](metrics.json) · [状态](status.json) · [资源](resource_manifest.json) · [环境](environment.md)

## 阻塞与未完成项

{blocked}

## 方法、诊断与采用决定

[DEIM报告](deimv2/report.md) · [GRAE报告](grae3dmot/report.md) · [HARP报告](harpnext/report.md) · [Mamba报告](mambafusion/report.md)。阶段报告可能包含明确标注的历史执行快照；与当前运行状态分开理解。

DEIM失败案例采用IoU0.5/score0.25独立诊断，不是AP：[案例](deimv2/badcases/summary.json)。GRAE冻结检测输入属 reused_archive，跟踪器真实执行才计 new_measured；不得把检测输入变化归因于tracker。HARP预处理速度不代表整网FPS，点语义精度必须仅评价原始有效GT点。CUDA算子smoke不等于Mamba整网部署。AP预算通过不能代替几何与数值一致性验收。优化的配对范围、稳定性与采用决定以各阶段报告为准。

## 回放、复现与局限

回放当前状态：**{visualization.get('status', 'planned')}**。[本地入口](http://127.0.0.1:9094/perception_2025/) · [验收记录](visualization/status.json)。缺少的预测应显式显示缺失，不能沿用前帧。

nuScenes仅mini教学诊断，预训练可能覆盖样本，不能视为全量独立泛化评价。COCO实际样本数见上表及image_ids。未测结果不填论文数字。原始数据、权重、第三方源码、环境及大型中间产物留本地。

[复现命令](run_commands.md) · [独立审查](review.md) · [大文件清单](artifact_receipt.json) · [原始计划](plan.md)
''')
    # Publish completion last so failure while rendering resources/report cannot look successful.
    write(out/'summarize/status.json', dict(status='passed', aggregate_status=aggregate, rows=len(rows)))
    print(json.dumps(dict(status=aggregate, metrics=len(rows))))


if __name__ == '__main__':
    main()
