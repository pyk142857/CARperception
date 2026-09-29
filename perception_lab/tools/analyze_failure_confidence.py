"""Confidence-stratified failures and score-threshold replay of fixed predictions."""
import json,math
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EDGES=[.25,.4,.6,.8,1.]
THRESHOLDS=[.1,.2,.25,.3,.4,.5,.6,.7,.8,.9]

def score_bin(score):
    if score is None:return 'N/A'
    if not math.isfinite(score) or not EDGES[0]<=score<=1:raise ValueError('Score outside baseline range')
    for low,high in zip(EDGES,EDGES[1:]):
        if score<high or high==1:return f'[{low:.2f},{high:.2f}'+(']' if high==1 else ')')

def case_confidence(case,boxes):
    if case['kind']=='false_negative':return None
    if case['kind']=='false_positive':
        box=boxes[case['prediction_index']]
        if box['class_name']!=case['class_name'] or box['score']!=case['score']:
            raise ValueError('FP source mismatch')
    else:
        found=[b for b in boxes if b.get('tracking_id')==case['tracking_id']]
        if len(found)!=1 or found[0]['class_name']!=case['class_name']:
            raise ValueError('Identity event prediction missing or ambiguous')
        box=found[0]
    score=float(box['score'])
    if not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid confidence')
    return score

def main():
    from evaluate_mini import load_frames,evaluate,measured_results,write_csv
    from failure_overlay import load_cases
    from evidence import sha256
    out=ROOT/'reports/confidence_analysis';out.mkdir(parents=True,exist_ok=True)
    packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
    status=json.loads((ROOT/'results/status.json').read_text())
    det,dp=measured_results(status,'M05','mini_scene',packets)
    tracks,tp=measured_results(status,'M11','mini_tracking_M05',packets)
    _,baseline=load_cases(ROOT/'reports/mini_evaluation',packets,.25)
    frames,_,_=load_frames(packets,det,tracks)
    original=json.loads((ROOT/'reports/mini_evaluation/cases.json').read_text())
    detailed=[];counts=Counter();class_counts=Counter()
    for case in original:
        if case['kind'] not in {'false_negative','false_positive','id_switch','gap_id_change'}:continue
        boxes=det[case['frame']]['boxes3d'] if case['module']=='detection' else tracks[case['frame']]['tracks3d']
        score=case_confidence(case,boxes);bucket=score_bin(score)
        row=dict(case,confidence=score,confidence_bin=bucket,
            confidence_basis='no matched prediction' if score is None else
                ('FP prediction score' if case['kind']=='false_positive' else 'current/new tracking ID prediction score'))
        detailed.append(row);counts[(case['module'],case['kind'],bucket)]+=1
        class_counts[(case['module'],case['kind'],case['class_name'],bucket)]+=1
    histogram=[dict(module=m,kind=k,confidence_bin=b,count=n) for (m,k,b),n in sorted(counts.items())]
    write_csv(out/'confidence_bins.csv',histogram)
    write_csv(out/'confidence_bins_by_class.csv',[dict(module=m,kind=k,class_name=c,confidence_bin=b,count=n)
              for (m,k,c,b),n in sorted(class_counts.items())])
    write_csv(out/'cases_with_confidence.csv',detailed)
    (out/'cases_with_confidence.json').write_text(json.dumps(detailed,ensure_ascii=False,indent=2)+'\n')
    sweeps=[];per_class=[];baseline_checked=False
    for threshold in THRESHOLDS:
        totals,_,classes,cases,_=evaluate(frames,packets,threshold,baseline['distance_gate_m'])
        if threshold==.25:
            for i,c in enumerate(cases):c['case_id']=f'case_{i:05d}'
            assert cases==original and totals==baseline['totals']
            baseline_checked=True
        for module,v in totals.items():
            sweeps.append(dict(threshold=threshold,module=module,tp=v['tp'],fn=v['fn'],fp=v['fp'],
                id_switch=v['events'].get('id_switch',0) if module=='tracking' else None,
                gap_id_change=v['events'].get('gap_id_change',0) if module=='tracking' else None,
                gt=v['gt'],precision=v['precision'],recall=v['recall']))
        identity_counts=Counter((c['kind'],c['class_name']) for c in cases if c['kind'] in {'id_switch','gap_id_change'})
        for row in classes:
            per_class.append(dict(threshold=threshold,module=row['module'],class_name=row['class_name'],
                tp=row['tp'],fp=row['fp'],fn=row['fn'],
                id_switch=identity_counts[('id_switch',row['class_name'])] if row['module']=='tracking' else None))
    assert baseline_checked
    for module in ['detection','tracking']:
        for kind,key in [('false_negative','fn'),('false_positive','fp')]:
            assert sum(n for (m,k,b),n in counts.items() if (m,k)==(module,kind))==baseline['totals'][module][key]
    assert sum(n for (m,k,b),n in counts.items() if k=='id_switch')==46
    assert sum(n for (m,k,b),n in counts.items() if k=='gap_id_change')==38
    write_csv(out/'threshold_sweep.csv',sweeps);write_csv(out/'threshold_by_class.csv',per_class)
    metadata=dict(scope='scene-0061 / 39 keyframes / single teaching scene',baseline_score=.25,
        distance_gate_m=2,thresholds=THRESHOLDS,confidence_bin_edges=EDGES,
        track_protocol='filter existing tracker outputs and reevaluate; tracker is NOT rerun',
        fn_confidence=None,id_confidence='score of the current/new tracking ID at the switch frame',
        identity_events='id_switch (consecutive observations); gap_id_change separate',
        count_unit='object-frame failures and identity transitions, not aggregated event groups',
        lower_bound='stored CenterPoint predictions were already filtered at 0.1',
        baseline_reproduction='all cases and totals exactly match',
        source_sha256={str(p):sha256(p) for p in [dp,tp,ROOT/'manifests/frame_packets.json',
            ROOT/'reports/mini_evaluation/cases.json',ROOT/'reports/mini_evaluation/summary.json',
            ROOT/'tools/evaluate_mini.py',ROOT/'tools/evaluation_utils.py',Path(__file__)]},
        statistics='descriptive only; one correlated scene, no independent runs/seeds or significance/CI claims')
    (out/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    table=['| 阈值 ≥ | 检测 FN | 检测 FP | 跟踪 FN | 跟踪 FP | 连续 ID 切换 | 间隔后换 ID |',
           '|---:|---:|---:|---:|---:|---:|---:|']
    for t in THRESHOLDS:
        d=next(r for r in sweeps if r['threshold']==t and r['module']=='detection')
        tr=next(r for r in sweeps if r['threshold']==t and r['module']=='tracking')
        table.append(f"| {t:.2f} | {d['fn']} | {d['fp']} | {tr['fn']} | {tr['fp']} | {tr['id_switch']} | {tr['gap_id_change']} |")
    bin_table=['| 分数区间 | 检测 FP | 跟踪 FP | 连续 ID 切换 | 间隔后换 ID |',
               '|---|---:|---:|---:|---:|']
    for a,b in zip(EDGES,EDGES[1:]):
        label=score_bin(a)
        bin_table.append(f"| {label} | {counts[('detection','false_positive',label)]} | {counts[('tracking','false_positive',label)]} | {counts[('tracking','id_switch',label)]} | {counts[('tracking','gap_id_change',label)]} |")
    text=['# 置信度与失败数量','',
        '2026-09-29。scene-0061，39帧；原有模型与轨迹输出，未重新推理或训练。',
        '统计单位为逐目标逐帧失败及身份转换次数，不是1268个归并事件。检测10类、跟踪7类，分支之间不能直接用数量比较优劣。',
        '','## 不同筛选阈值下重新评估','',
        '每个阈值都重新执行原匹配和身份评估，不能把固定0.25阈值的失败简单筛选来替代。跟踪使用既有轨迹输出再筛分数，不是改变检测门限后重跑跟踪器；生产系统效果需另做完整跟踪重跑。',
        '',*table,'','## 固定阈值0.25下，按失败预测的分数分档','',
        'FN没有匹配预测，因此没有预测置信度。检测FN=75、跟踪FN=39，均单列N/A；不把FN当作0分预测，也不采用附近误匹配框的分数。',
        'ID切换用切换当帧新ID对应预测的score，不是ID关联可信度；旧ID分数不用于本次分档。区间左闭右开，最后一档含1。','',*bin_table,'',
        '分档表与阈值表含义不同：阈值提高后匹配关系可能变化，ID切换和断轨统计尤其不能通过简单删除低分事件得到。',
        '','## 复现','',
        '```bash','perception_lab/envs/mmdet3d/bin/python perception_lab/tools/analyze_failure_confidence.py','```','',
        '当前0.25基线逐条案例与总计完全复现，FP/FN/ID分档总和核对通过。原评估、事件和Rerun数据没有改写。',
        '','## 产物','',
        '- threshold_sweep.csv / threshold_by_class.csv：各阈值总计和类别统计。',
        '- confidence_bins.csv / confidence_bins_by_class.csv：基线失败的分数分档。',
        '- cases_with_confidence.json / csv：每条原始case_id及分数来源，可回溯原预测。',
        '- summary.json：口径、参数、哈希。',
        '- stats-appendix.md：统计限制；figure-catalog.md：图表说明。',
        '','![阈值与失败数量](figures/threshold_counts.png)']
    (out/'analysis-report.md').write_text('\n'.join(text)+'\n')
    (out/'stats-appendix.md').write_text('''# 统计口径
仅描述性计数。N=1个场景、39个相关关键帧；同一目标跨帧重复出现，不能当作独立样本。
未提供多次独立运行或独立场景抽样，因此不报告均值±标准差、置信区间、显著性或泛化最优阈值。
改变阈值改变可匹配目标和身份观测机会；ID切换次数下降不等于跟踪更好，必须同时看漏检/召回。
本数据为mini教学场景，存在预训练数据重叠风险。模型score不是经校准的真实正确概率。
最低已有预测门限0.1，无法用现有文件评估低于0.1的完整输出。
''')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for ax,metric,title in zip(axes,['fp','fn','id_switch'],['False positives','False negatives','Consecutive ID switches']):
        for module,color in [('detection','#2670b7'),('tracking','#b75320')]:
            if metric=='id_switch' and module=='detection':continue
            data=[r for r in sweeps if r['module']==module]
            ax.plot([r['threshold'] for r in data],[r[metric] for r in data],marker='o',label=module,color=color)
        ax.set(title=title,xlabel='Score threshold (>=)',ylabel='Count')
        ax.axvline(.25,color='gray',linestyle=':',linewidth=1);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('scene-0061: fixed predictions / existing tracks; 39 frames')
    fig.tight_layout();fig.savefig(out/'figures/threshold_counts.png',dpi=160);plt.close(fig)
    (out/'figure-catalog.md').write_text('''# 图表说明
figures/threshold_counts.png：对比筛选阈值与FP、FN、连续ID切换次数，虚线为当前0.25基线。
来源：threshold_sweep.csv；横轴为预测分数门限，纵轴为原始次数，不是百分比。
读取时同时观察FP减少和FN增加；ID切换依赖重评匹配与可见机会，不应单独用次数判断跟踪质量。
没有独立重复实验，因此无误差条。跟踪曲线是既有输出的后置筛选，不代表重跑跟踪器的曲线。
''')
    print('\n'.join(table));print('\n'.join(bin_table))
if __name__=='__main__':main()
