"""Additional score-stratified diagnosis; never changes fixed acceptance flags."""
import csv,json,math,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/deployment_4090_20261008/bevfusion_tensorrt'
FIELDS=['center_m','size_m','yaw_rad','score_abs','velocity_mps']
def stats(rows):
 d={'pairs':len(rows)}
 for k in FIELDS:
  values=[r[k] for r in rows];d[k+'_p50']=float(np.median(values)) if values else None;d[k+'_p95']=float(np.percentile(values,95)) if values else None
 return d

def main():
 summary={};table=[];cases=[]
 for split in ['scene0061','mini_val']:
  cmp=json.loads((OUT/split/'comparison.json').read_text());matches=cmp['matched'];summary[split]={'comparison_sha256':hashlib.sha256((OUT/split/'comparison.json').read_bytes()).hexdigest(),'comparison_input_hashes':cmp['input_hashes'],'fixed_geometry_flags':cmp['geometry_diagnostics'],'fixed_metric_budget_passed':cmp['metric_budget_passed'],'groups':{},'tail_composition':{},'symmetry_diagnostic':{}}
  for cut in [0,.25,.5]:
   group=[r for r in matches if min(r['reference_score'],r['candidate_score'])>=cut];summary[split]['groups'][str(cut)]=stats(group)
   for cls in sorted({r['class_name'] for r in matches}):
    table.append({'split':split,'both_scores_at_least':cut,'class_name':cls,**stats([r for r in group if r['class_name']==cls])})
  limits={'center_m':.2,'size_m':.2,'yaw_rad':.1,'score_abs':.03,'velocity_mps':.3}
  for k,limit in limits.items():
   tail=[r for r in matches if r[k]>limit];summary[split]['tail_composition'][k]={'threshold':limit,'tail_pairs':len(tail),'at_least_one_score_below_025':sum(min(r['reference_score'],r['candidate_score'])<.25 for r in tail),'both_scores_at_least_05':sum(min(r['reference_score'],r['candidate_score'])>=.5 for r in tail)}
  barrier=[r for r in matches if r['class_name']=='barrier'];norm=[min(r['yaw_rad'],abs(math.pi-r['yaw_rad'])) for r in barrier];summary[split]['symmetry_diagnostic']={'barrier_pairs':len(barrier),'barrier_raw_yaw_p95':float(np.percentile([r['yaw_rad'] for r in barrier],95)),'barrier_pi_period_yaw_p95':float(np.percentile(norm,95)),'traffic_cone_pairs':sum(r['class_name']=='traffic_cone' for r in matches),'note':'Additional interpretation only: barrier pi symmetry; traffic cone yaw not assessed as orientation quality. Fixed all-class yaw flags retained.'}
  ref={r['sample_token']:r for r in json.loads((ROOT/'reports/deployment_4090_20261008/bevfusion_reference'/split/'predictions.json').read_text())};cand={r['sample_token']:r for r in json.loads((ROOT/'outputs/bevfusion_tensorrt'/split/'predictions.json').read_text())}
  selections=[('high_score_center',lambda r:min(r['reference_score'],r['candidate_score'])>=.5,'center_m'),('barrier_yaw',lambda r:r['class_name']=='barrier','yaw_rad'),('high_score_vehicle_yaw',lambda r:r['class_name'] in ['car','truck','bus'] and min(r['reference_score'],r['candidate_score'])>=.5,'yaw_rad')]
  for tag,cond,key in selections:
   eligible=[r for r in matches if cond(r)]
   if not eligible:continue
   r=max(eligible,key=lambda x:x[key]);rr=next(b for b in ref[r['sample_token']]['boxes3d'] if b['class_name']==r['class_name'] and abs(b['score']-r['reference_score'])<1e-8);cc=[b for b in cand[r['sample_token']]['boxes3d'] if b['class_name']==r['class_name'] and abs(b['score']-r['candidate_score'])<1e-8];bb=min(cc,key=lambda b:abs(np.linalg.norm(np.array(b['center_xyz'])-rr['center_xyz'])-r['center_m']))
   nearref=sum(b['class_name']==r['class_name'] and np.linalg.norm(np.array(b['center_xyz'])-rr['center_xyz'])<2 for b in ref[r['sample_token']]['boxes3d']);nearcand=sum(b['class_name']==r['class_name'] and np.linalg.norm(np.array(b['center_xyz'])-rr['center_xyz'])<2 for b in cand[r['sample_token']]['boxes3d']);cases.append({'split':split,'case':tag,**r,'distance_ego_m':float(np.linalg.norm(rr['center_xyz'][:2])),'reference_center':rr['center_xyz'],'candidate_center':bb['center_xyz'],'same_class_reference_neighbours_2m':int(nearref),'same_class_candidate_neighbours_2m':int(nearcand),'interpretation':'Multiple neighbouring proposals can change Hungarian correspondence; not proof of root cause. Inspect cameras and proposal-level pre-decode outputs.'})
 (OUT/'geometry_diagnosis.json').write_text(json.dumps(summary,indent=2));(OUT/'geometry_failure_cases.json').write_text(json.dumps(cases[:5],indent=2))
 with (OUT/'geometry_by_class_score.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
 print(json.dumps(summary,indent=2))
def render_report():
 p=OUT;d=json.loads((p/'geometry_diagnosis.json').read_text());cases=json.loads((p/'geometry_failure_cases.json').read_text())
 s='# FP16 几何尾部诊断（不修改验收阈值）\n\n使用末区间修复后完整39/81帧的最新comparison.json；输入哈希记录在JSON。固定mAP/NDS预算通过，全200候选的几何诊断仍有失败项。按已确定Hungarian匹配结果分组，**没有重新匹配、删掉评估输入或放宽阈值**。分数下限要求双方均达到，不代表GT确认目标。\n\n|split|双方分数下限|匹配数|中心P95 m|尺寸P95 m|yaw P95 rad|分数差P95|速度P95 m/s|\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
 for split,x in d.items():
  for cut,g in x['groups'].items():
   s+=f'|{split}|{cut}|{g["pairs"]}|'+ '|'.join(f'{g[k+"_p95"]:.4f}' for k in ['center_m','size_m','yaw_rad','score_abs','velocity_mps'])+'|\n'
 s+='\n中心差>0.2m的配对中：\n\n'
 for split,x in d.items():
  t=x['tail_composition']['center_m'];s+=f'- {split}：{t["at_least_one_score_below_025"]}/{t["tail_pairs"]}（{100*t["at_least_one_score_below_025"]/t["tail_pairs"]:.2f}%）至少一侧分数低于0.25；双方≥0.5仍有{t["both_scores_at_least_05"]}对。\n'
 s+='\n不能把全部差异归结为低分候选。逐类统计见[geometry_by_class_score.csv](geometry_by_class_score.csv)，原始flags及完整尾部见[geometry_diagnosis.json](geometry_diagnosis.json)。barrier π对称性仅作解释，不改写yaw验收：\n\n'
 for split,x in d.items():
  t=x['symmetry_diagnostic'];s+=f'- {split} barrier yaw P95：{t["barrier_raw_yaw_p95"]:.4f} → {t["barrier_pi_period_yaw_p95"]:.4f} rad。\n'
 s+='\n邻近同类proposal及输出TopK变化可能改变Hungarian配对，但这是待验证解释，不是根因证明；此配置不执行后置NMS。\n\n## 五个实测复核案例\n\n|split/token|类别|距离m|参考/候选分数|中心差m|yaw差rad|参考/候选2m邻居数|\n|---|---|---:|---:|---:|---:|---:|\n'
 for c in cases:
  s+=f'|{c["split"]} / {c["sample_token"]}|{c["class_name"]}|{c["distance_ego_m"]:.2f}|{c["reference_score"]:.4f}/{c["candidate_score"]:.4f}|{c["center_m"]:.4f}|{c["yaw_rad"]:.4f}|{c["same_class_reference_neighbours_2m"]}/{c["same_class_candidate_neighbours_2m"]}|\n'
 s+='\n双方实际中心及尺寸/速度/分数差见[geometry_failure_cases.json](geometry_failure_cases.json)。这些是部署差异案例，并非GT确认的FP/FN。\n'
 (p/'geometry_diagnosis.md').write_text(s)

if __name__=='__main__':
 main()
 render_report()
