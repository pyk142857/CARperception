"""Offline report of recorded evidence, including incomplete work."""
import argparse
import csv
import html
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    status=json.loads((root/'results/status.json').read_text())
    plan=json.loads((root/'configs/execution_plan.yaml').read_text())
    report=root/'reports';report.mkdir(exist_ok=True)
    rows=[];metrics=[];gallery=[];videos=[];case_count=0
    for mid,module in plan['modules'].items():
        for stage,job in module['jobs'].items():
            record=status['modules'].get(mid,{}).get(stage,{'status':'pending','reason':'Not executed'})
            if mid == 'M14' and stage == 'report' and record['status'] == 'running':
                continue  # The report cannot include its own eventual completion status.
            rows.append([mid,module['model'],stage,record['status'],record.get('reason',''),record.get('log','')])
            if record['status']!='passed':continue
            directory=Path(record['log']).parent if record.get('log') else None
            if directory:
                if (directory/'cases.json').is_file():
                    case_count += len(json.loads((directory/'cases.json').read_text()))
                videos += [(mid,stage,os.path.relpath(v,report)) for v in directory.glob('*.mp4')]
                for picture in list(directory.glob('*.jpg'))[:6]+list(directory.glob('*.png'))[:6]:
                    gallery.append((mid,stage,os.path.relpath(picture,report)))
                if (directory/'metrics.json').exists():
                    found=json.loads((directory/'metrics.json').read_text())
                    for metric in found:
                        if metric.get('source')!='measured':raise ValueError('Unmeasured primary metric')
                        metrics.append(dict(module=mid,**metric))
    fields=['module','name','value','unit','direction','protocol','split','scope','sample_count','run_id','source','raw_evaluator_path']
    with (root/'results/metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(metrics)
    with (root/'results/performance.csv').open('w') as f:
        csv.writer(f).writerow(['model','run','precision','shape','batch','sample_count','timing_scope','p50_ms','p95_ms','throughput','peak_memory','hardware'])
    lines=['# Perception lab — partial','','A/B/C 均未达到。已通过项目仅代表该阶段及记录的数据范围。',
        '示例图片和 mini 工程诊断不构成正式 val 回放。没有生成完整联合回放视频，也没有完成训练或部署收益对比。','',
        '## 实测指标','']
    if metrics:
        lines+=['| 模块 | 指标 | 值 | 范围 | 样本数 |','|---|---|---:|---|---:|']
        for m in metrics:lines.append(f"| {m['module']} | {m['name']} | {m['value']:.6f} | {m['protocol']} | {m['sample_count']} |")
    else:lines+=['尚无完整任务评估。所有未知指标为 null；CSV 仅保留已测行。']
    lines+=['','## 阶段状态','','| 模块 | 阶段 | 状态 | 原因 |','|---|---|---|---|']
    lines += [f'| {r[0]} | {r[2]} | {r[3]} | {r[4]} |' for r in rows]
    lines+=['','## 证据与限制','',
        '- [完整状态](../results/status.json)；[实测指标](../results/metrics.csv)；[缺失资产](missing_assets.md)；[资源预算](resource_budget.md)。',
        '- [仓库 commit](../configs/locked/repositories.json)；[权重来源和哈希](../checkpoints/manifest.json)。',
        '- GPU 存在其他用户任务；尚无满足50次预热、200次测量的独占GPU性能结果。performance.csv只有表头，不填入伪FPS。',
        '- 未完成 Cityscapes、nuScenes trainval、SurroundOcc 官方评估；训练A/B均未执行，不宣称精度提升。',
        f'- 已保存 {case_count} 个本次COCO预测的阈值定义漏检诊断案例。它们不替代官方COCO AP，也不证明天气/遮挡等成因；未知属性明确为unknown。',
        '- 新建隔离环境复用已有PyTorch，依赖已有基础环境路径；版本冻结见envs/。',
        '- M04–M10 TensorRT可行性不等同导出；自定义稀疏卷积、BEV pooling、deformable attention仍需逐目标验证。',
        '- 重跑入口及已验证命令见[README](../README.md)。','']
    for mid,stage,rel in videos:lines += [f'- [{mid} / {stage} mini诊断视频]({rel})']
    for mid,stage,rel in gallery:lines += [f'### {mid} / {stage}',f'![{mid} {stage}]({rel})','']
    (report/'final_report.md').write_text('\n'.join(lines))
    (out/'report_source.md').write_text('\n'.join(lines))
    h=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Perception lab — partial</title>',
       '<style>body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#f7f8fa;color:#182638}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #ccd5df;padding:8px;text-align:left}img{max-width:100%}pre{white-space:pre-wrap}a{color:#1264a3}</style>',
       '<p>当前学习目标与入口见 <a href="mini_learning.html">mini 教学流程</a>；以下保留原完整计划状态。</p><h1>Perception lab — partial</h1><p>A/B/C 均未达到。数据范围与阶段分别记载；无实测值处不填零。</p>',
       '<p><a href="final_report.md">完整报告</a> · <a href="missing_assets.md">缺失资产</a> · <a href="../results/metrics.csv">实测指标 CSV</a> · <a href="../results/status.json">状态 JSON</a></p>',
       '<h2>实测指标</h2><pre>'+html.escape(json.dumps(metrics,ensure_ascii=False,indent=2))+'</pre>',
       '<h2>状态</h2><table><tr><th>模块</th><th>模型</th><th>阶段</th><th>状态</th><th>原因</th><th>日志</th></tr>']
    for row in rows:
        cells=[html.escape(str(x)) for x in row[:5]]
        link='<a href="'+html.escape(os.path.relpath(row[5],report))+'">日志</a>' if row[5] else ''
        h.append('<tr>'+''.join('<td>'+x+'</td>' for x in cells+[link])+'</tr>')
    h.append('</table><h2>本次真实输出</h2>')
    for mid,stage,rel in videos:h.append(f'<h3>{mid} / {stage} — mini diagnostic</h3><video controls preload="metadata" width="800" src="{html.escape(rel)}"></video>')
    for mid,stage,rel in gallery:h.append(f'<h3>{mid} / {stage}</h3><img loading="lazy" src="{html.escape(rel)}" alt="{mid} {stage}">')
    h.append('</html>');(report/'index.html').write_text('\n'.join(h))
    (out/'report_source.html').write_text('\n'.join(h))
    (out/'report_manifest.json').write_text(json.dumps({'status':'partial','milestones':{'A':False,'B':False,'C':False},
        'report':str(report/'final_report.md'),'html':str(report/'index.html'),'metric_count':len(metrics)},indent=2))


if __name__=='__main__':main()
