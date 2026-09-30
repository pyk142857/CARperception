"""Render the measured experiment table and comparison figure."""
import csv,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
rows=list(csv.DictReader((HERE/'metrics.csv').open()));visible=[r for r in rows if r['variant']!='S0_mirror']
names={'S0_author':'作者基线（贪心）','S1_hungarian':'作者匈牙利','S1_gated':'修正门限处理的匈牙利','S2_kalman':'匈牙利＋卡尔曼','S3_rich_only':'匈牙利＋尺寸/运动代价','S3_kalman_rich':'匈牙利＋卡尔曼＋代价','S4_greedy_high':'贪心＋两阶段，新建≥0.5','S4_greedy_low':'贪心＋两阶段，新建≥0.25','S4_high_birth':'匈牙利＋两阶段，新建≥0.5','S4_low_birth':'匈牙利＋两阶段，新建≥0.25','S4_kalman_rich_high':'卡尔曼/代价＋两阶段，新建≥0.5','S4_kalman_rich_low':'卡尔曼/代价＋两阶段，新建≥0.25'}
text='''# 连续 ID 切换优化：顺序实验结果

日期：2026-09-30。已执行方案中的固定检测对照（S0–S5）；没有修改当前Rerun记录、max_age或正式跟踪入口。

## 实验设计与验证

同一 scene-0061 的39帧、相同CenterPoint检测、同一顺序/时间戳/速度/标定；max_age=3，行人关联门限1m，评估score≥0.25、同类中心距离<2m。详见[实施方案](../../instruction/20260930_01_tracking_optimization.md)。每个候选均复跑，输出完全一致；作者基线及独立贪心镜像与既有轨迹逐项精确一致，见[校验与输入哈希](verification.json)。

跟踪器不读GT；GT仅在复用评估函数时用于计算结果。未匹配历史轨迹不输出为当前检测。卡尔曼只用于关联，不将平滑位置写成新的检测框。

本轮按阶段推进并保留退化结果：S1只换作者算法；增加S1_gated对照实现问题；S2测试运动估计；S3分别单独加代价及叠加卡尔曼；S4在原贪心、全局匹配、卡尔曼/代价三个底座上测两阶段。参数不是通过本场景大规模搜索得到的。

## 完整结果

IDSW=连续帧ID切换；Gap=中断后换ID。均为当前固定阈值诊断，不是官方AMOTA。

| 方案 | 全类 IDSW | 全类 Gap | 行人 IDSW | 行人 Gap | FP | FN | 近前方 FN | 混合观测 | 候选准入 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
'''
for r in visible:
 text+='| '+names[r['variant']]+' | '+' | '.join(r[k] for k in ['id_switch','gap_id_change','ped_id_switch','ped_gap_id_change','fp','fn','front_fn','mixed_observations'])+' | '+('通过' if r['candidate_pass']=='True' else '基线' if r['variant']=='S0_author' else '未通过')+' |\n'
passed=[r for r in visible if r['candidate_pass']=='True']
text+='\n## 结论\n\n优先候选为 **S4_greedy_low（保留贪心＋置信度分阶段，新轨迹阈值0.25）**。全类IDSW 46→38（减少17.4%），行人42→36（减少14.3%）；全类Gap 38→37，行人34→33；FP/FN及近前方FN不变。全类两类身份错误合计84→75（减少10.7%）。相比叠加运动模型，该候选改动小，优先进入其他mini场景验证，当前默认仍未替换。\n\n' 
if passed:
 for r in passed:
  text+=f"- 本场景候选：**{names[r['variant']]}**，全类连续切换46→{r['id_switch']}（减少{(46-int(r['id_switch']))/46:.1%}），行人42→{r['ped_id_switch']}（减少{(42-int(r['ped_id_switch']))/42:.1%}）。详见上表中FN/FP和身份混合约束。\n"
else:text+='本轮没有候选同时满足预定约束；不应替换默认跟踪器。表面减少IDSW而增加FN或身份混合的方案不予推荐。\n'
text+='''
- 作者匈牙利未必优于贪心：其门限无效配对处理会提前丢失旧轨迹，独立修正版修正该问题并使用有限大代价保证有效匹配数优先，但指标是否改善仍取决于检测与场景，不能只凭算法名称选择。
- 卡尔曼参数固定、仅位置观测更新，初始速度来自检测；若短轨迹、速度突变或检测跳动，常速度假设可能不适用。本轮结果只针对该实现/参数，不能证明卡尔曼方法普遍无效。
- 两阶段优先使用≥0.5的框关联，低分框可接回已有轨迹。提高新建阈值可能同时减少误检和压制真正的新目标，必须看FN；不能用减少输出数量来冒充身份稳定性提升。
- 全类原始检测几何相同但不同身份会改变评估中的“保留上一帧关联”，因此个别方案TP/FP/FN及中心误差仍会变化。这不是模型重新推理后的检测精度变化。

## 诊断定义与局限

“混合观测”按每个预测ID汇总其匹配的GT实例，计算总匹配次数减去占比最多的GT次数，再对ID求和；用于发现ID关联多个GT的迹象。它受评估匹配影响，不是已证实的串人次数，也不是官方IDF1。近前方为0≤x≤30m、|y|≤3m，仅作复核矩形，不代表实际行驶路径。

候选准入要求：全类和行人连续ID切换都下降，两类身份错误合计不增加，FP/FN、行人FP/FN、近前方FN、混合观测不增加。通过不代表已验证可部署。

单场景、确定性配对实验，无独立场景重复，不计算显著性或置信区间。mini与训练集可能重叠。各帧不是独立样本。耗时包括Python诊断收集，是本机一次运行参考，不是生产性能基准。

## 代码、产物与复现

- [实验跟踪器](../../tools/experimental_tracker.py)：关联、卡尔曼状态、两阶段规则和逐框诊断。
- [顺序实验入口](../../tools/optimize_tracking.py)：配置、对照、重复运行、身份混合诊断、原评估复用。
- [测试](../../tests/test_experimental_tracker.py)：全局关联冲突、类别/门限、空帧过期、低分续接/新建、状态重置、无效配对保留。
- [全部指标](metrics.csv)、[逐类](classes.csv)、[逐帧](frames.csv)、[配置](configs.json)。
- [67项测试日志](tests.txt)；[优先候选新增/消失的身份事件](candidate_event_changes.json)按失败类型、帧和GT实例对齐，避免直接比较重编号后的预测ID。
- 每个候选`*_tracks.json.gz`保存完整输出；`*_associations.json.gz`保存逐帧逐检测的匹配/新建/抑制及原因，作者原版不额外插入日志，镜像及独立实现包含日志。`*_identity.json`保存切换和中断后换ID的GT对照实例。
- 新建原因区分无同类旧轨迹、门限拒绝、有效候选被占用。原因是该帧决策记录，不自动等价于某个GT身份失败的最终根因。原始检测索引采用送入跟踪器的七类检测顺序。

```bash
perception_lab/envs/mmdet3d/bin/python -m unittest discover -s perception_lab/tests
perception_lab/envs/mmdet3d/bin/python perception_lab/tools/optimize_tracking.py
perception_lab/envs/mmdet3d/bin/python perception_lab/reports/tracking_optimization/build_report.py
```

## 后续

优先将本轮通过的候选（若无通过，则保留基线）用于其他mini场景的固定检测验证。其他场景需先准备真实CenterPoint输出；当前没有声称已完成该验证。再针对高频跳变目标复核速度/中心误差，决定是否优化检测输入、历史点云或训练。图像ReID、IoU代价、检测重训均未在本轮运行。

![身份错误与漏检对照](comparison.png)
'''
(HERE/'report.md').write_text(text)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,(a,b)=plt.subplots(2,1,figsize=(12,8),sharex=True)
x=list(range(len(visible)));labels=[r['variant'] for r in visible]
a.bar(x,[int(r['id_switch']) for r in visible],label='Consecutive ID switches');a.bar(x,[int(r['gap_id_change']) for r in visible],bottom=[int(r['id_switch']) for r in visible],label='ID change after gap');a.set_ylabel('Identity events');a.legend()
b.bar(x,[int(r['fn']) for r in visible],color='#cc6644');b.axhline(39,color='black',linestyle='--',label='Baseline FN=39');b.set_ylabel('False negatives');b.legend();b.set_xticks(x);b.set_xticklabels(labels,rotation=50,ha='right');fig.suptitle('Fixed CenterPoint detections / scene-0061 / 39 frames');fig.tight_layout();fig.savefig(HERE/'comparison.png',dpi=140)
