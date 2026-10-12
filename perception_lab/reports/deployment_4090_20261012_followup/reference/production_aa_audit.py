"""CPU-only independent AA/Hungarian and raster evidence audit. No model execution."""
import hashlib,itertools,json,pathlib,sys
import numpy as np
OUT=pathlib.Path(__file__).resolve().parent;ROOT=OUT.parents[2];sys.path.insert(0,str(ROOT/'tools'))
from bevfusion_compare import compare
from pyquaternion import Quaternion
KEYS=['center_m','size_m','yaw_rad','score_abs','velocity_mps']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def stats(rows):
 return {k:({'count':len(rows),'mean':float(np.mean([r[k] for r in rows])),'p50':float(np.percentile([r[k] for r in rows],50)),'p95':float(np.percentile([r[k] for r in rows],95)),'max':float(max(r[k] for r in rows))} if rows else None) for k in KEYS}
def groups(rows):return {name:{'matched_pairs':len(rs),'metrics':stats(rs),'worst_center':sorted(rs,key=lambda r:r['center_m'],reverse=True)[:2],'worst_yaw':sorted(rs,key=lambda r:r['yaw_rad'],reverse=True)[:2]} for name,rs in [('all',rows)]+[(f'both_score_ge_{v}',[r for r in rows if min(r['reference_score'],r['candidate_score'])>=v]) for v in [.25,.5]]}
config=json.loads((ROOT/'configs/deployment_4090.yaml').read_text())['tensorrt']['acceptance_fixed_before_run']
A={'scope':'reference/reference same-backend output diagnosis; all saved outputs retained; no new official evaluation','matching':'existing class-constrained Hungarian on 3D centers radius2m; all 10 within-group unordered pairs, all25 cross-group pairs; no row-identity assumptions','score_groups':'post-match explanatory groups only; matching and full-candidate thresholds unchanged','instrumentation_note':'production has no TopK identities; cross-group differences alone cannot separate hook perturbation from baseline non-determinism','thresholds':config,'comparison_code_sha256':sha(ROOT/'tools/bevfusion_compare.py'),'tokens':[]}
packet_by_token={x['sample_token']:x for split in ['scene0061','mini_val'] for x in json.loads((ROOT/f'reports/deployment_4090_20261008/inputs/{split}_packets.json').read_text())}
production_bitwise=[]
pooled={k:[] for k in ['production_AA','instrumented_AA','production_vs_instrumented']};collisions=[]
old_collision=json.loads((OUT/'raster_collision_evidence.json').read_text())['comparisons'];old_by_key={(r['token'],r['repeat']):r for r in old_collision}
for prodpath in sorted(OUT.glob('*.production.json')):
 token=prodpath.name[:-len('.production.json')];diagpath=OUT/(token+'.json');prod=json.loads(prodpath.read_text());diag=json.loads(diagpath.read_text());ps=prod['repeats'];ds=diag['repeat'];assert len(ps)==len(ds)==5
 row={'token':token,'production_json_sha256':sha(prodpath),'instrumented_json_sha256':sha(diagpath),'raw_verification':[],'comparisons':{}}
 for kind,entries in [('production',ps),('instrumented',ds)]:
  baseline_arrays=None
  for e in entries:
   raw=pathlib.Path(e['raw_path']);assert sha(raw)==e['raw_sha256']
   with np.load(raw) as z:
    names=['boxes','scores','labels'] if kind=='production' else ['output.native_boxes','output.scores','output.labels'];b,s,l=[z[k] for k in names]
    if kind=='production':assert np.array_equal(b,e['native_boxes']) and np.array_equal(s,e['scores']) and np.array_equal(l,e['labels'])
    else:assert np.array_equal(b,[x['tensor'] for x in e['native_boxes']]) and np.array_equal(s,[x['score'] for x in e['native_boxes']]) and np.array_equal(l,[x['label'] for x in e['native_boxes']])
    if baseline_arrays is None:baseline_arrays=[v.copy() for v in [b,s,l]]
    elif kind=='production':production_bitwise.append(all(np.array_equal(v,base) for v,base in zip([b,s,l],baseline_arrays)))
    mat=np.asarray(packet_by_token[token]['sensors']['LIDAR_TOP']['T_sensor_to_ego']);q=Quaternion(matrix=mat[:3,:3])
    for native,box in zip(b,e['standardized_boxes']):
     gravity=native[:3].astype(np.float64).copy();gravity[2]+=float(native[5])*.5
     expected_center=mat[:3,:3]@gravity+mat[:3,3]
     assert np.allclose(expected_center,box['center_xyz'],atol=1e-5,rtol=0)
     assert np.allclose(native[3:6],box['size_wlh'],atol=1e-7,rtol=0)
     rotation=q*Quaternion(axis=[0,0,1],radians=float(-native[6]-np.pi/2))
     assert np.allclose(rotation.rotation_matrix,Quaternion(box['rotation_wxyz']).rotation_matrix,atol=1e-6,rtol=0)
     assert np.allclose((mat[:3,:3]@np.r_[native[7:9],0])[:2],box['velocity_xy'],atol=1e-6,rtol=0)
    assert len(b)==len(e['standardized_boxes']);assert np.array_equal(s,[x['score'] for x in e['standardized_boxes']]);assert max(e['standardization_audit']['max_error'].values())<1e-7
   row['raw_verification'].append({'group':kind,'repeat':e['repeat'],'sha256':e['raw_sha256'],'native_and_standardized_scores_agree':True,'independent_standardization_center_size_rotation_velocity_checked':True})
 for name,left,right,pairs in [('production_AA',ps,ps,list(itertools.combinations(range(5),2))),('instrumented_AA',ds,ds,list(itertools.combinations(range(5),2))),('production_vs_instrumented',ps,ds,list(itertools.product(range(5),repeat=2)))]:
  merged=[];pairrows=[]
  for i,j in pairs:
   c=compare([{'sample_token':token,'boxes3d':left[i]['standardized_boxes']}],[{'sample_token':token,'boxes3d':right[j]['standardized_boxes']}],2.)
   for m in c['matched']:m.update(reference_repeat=i,candidate_repeat=j)
   merged+=c['matched'];f=c['frames'][0];pairrows.append({'reference_repeat':i,'candidate_repeat':j,'coverage':f,'groups':groups(c['matched'])})
  pooled[name]+=merged;row['comparisons'][name]={'pair_count':len(pairs),'groups':groups(merged),'unmatched_reference_total':sum(x['coverage']['unmatched_reference'] for x in pairrows),'unmatched_candidate_total':sum(x['coverage']['unmatched_candidate'] for x in pairrows),'pairs':pairrows}
 A['tokens'].append(row)
 # Independently map scatter destinations to contribution sets, all four comparisons.
 with np.load(pathlib.Path(ds[0]['raw_path'])) as z:
  base={k:z[k] for k in z.files if k.startswith('camera.raster.')};d0=z['camera.sparse_depth'];contributions=[]
  for cam in range(6):
   pre=f'camera.raster.{cam}';pixels=base[pre+'.rows']*d0.shape[-1]+base[pre+'.cols'];values=base[pre+'.depth_values'];mp={}
   for px,v in zip(pixels,values):mp.setdefault(int(px),[]).append(float(v))
   contributions.append(mp)
  for e in ds[1:]:
   with np.load(pathlib.Path(e['raw_path'])) as zz:
    equal=all(np.array_equal(v,zz[k]) for k,v in base.items());d1=zz['camera.sparse_depth'];changed=np.argwhere(d0!=d1);multi=member=0
    for cam,channel,y,x in changed:
     vals=contributions[cam][int(y*d0.shape[-1]+x)];multi+=len(vals)>1;member+=float(d0[cam,channel,y,x]) in vals and float(d1[cam,channel,y,x]) in vals
    c={'token':token,'repeat':e['repeat'],'actual_scatter_arguments_bitwise_equal':equal,'changed_sparse_depth_pixels':len(changed),'changed_pixels_with_multiple_contributions':int(multi),'changed_pixels_where_both_outputs_are_actual_contributions':int(member)}
    c['matches_existing_evidence']=all(old_by_key[(token,e['repeat'])][k]==v for k,v in c.items() if k!='matches_existing_evidence');assert c['matches_existing_evidence'];collisions.append(c)
 print('audited',token,flush=True)
A['aggregate']={name:groups(rows) for name,rows in pooled.items()};A['raster_collision_independent_recalculation']=collisions
A['conclusions']={'all40_raw_hashes_verified':True,'all16_collision_comparisons_reproduced':len(collisions)==16 and all(x['matches_existing_evidence'] for x in collisions),'all_changed_pixels_are_actual_duplicate_scatter_contributions':all(x['actual_scatter_arguments_bitwise_equal'] and x['changed_sparse_depth_pixels']==x['changed_pixels_with_multiple_contributions']==x['changed_pixels_where_both_outputs_are_actual_contributions'] for x in collisions),'production_bitwise_determinism':all(production_bitwise),'official_metric_budget':'not evaluated by this same-frame audit','hook_neutrality':'not established; cross-group distributions compared, individual difference not causal attribution'}
(OUT/'production_aa_audit.json').write_text(json.dumps(A,indent=2)+'\n')
lines=['# Reference AA输出与散射证据复核','', 'CPU复核40个原始NPZ及JSON对应，并用LiDAR→ego矩阵独立重算标准化中心、尺寸、旋转和速度；4个token每组5次。每token生产AA和插桩AA各比较全部10个重复对，跨组比较25对，共180对。沿用原same-class 3D center 2m Hungarian；未把行号当作候选身份。所有候选保留；双方分数≥0.25/0.5只作匹配后解释性分组。','', '|比较|分组|匹配对|center p95/max m|size p95/max m|wrapped yaw p95/max rad|score p95/max|velocity p95/max m/s|','|---|---|---:|---:|---:|---:|---:|---:|']
for name,g in A['aggregate'].items():
 for group,s in g.items():
  vals=['%.6g / %.6g'%(s['metrics'][k]['p95'],s['metrics'][k]['max']) for k in KEYS];lines.append('|'+name+'|'+group+'|'+str(s['matched_pairs'])+'|'+'|'.join(vals)+'|')
lines+=['','yaw采用旧比较器的相对四元数主值角（绝对值≤π），不会把±π边界当作6.28rad物理差异。汇总按匹配对加权，重复比较共享同一输入，不能当作独立样本置信区间。未匹配数、各token/各对/高分最坏案例和原始哈希见JSON。未计算官方mAP/NDS，不将AA与跨后端精度预算混用。','','独立重算全部16个散射比较，与旧raster_collision_evidence逐项一致：真实scatter输入逐位相同；每个发生改变的深度像素都具有多个真实贡献，两个输出值均出自这些贡献。它支持参考后端覆盖式多点散射的非确定性，不单独量化TRT FP16误差。','','未插桩输出已有变化，生产/插桩差异不能直接归因为hook；此处不宣称hook数值中性。捕获阶段来源ID对齐属另一诊断，未混入Hungarian统计。','','复现：`python perception_lab/reports/deployment_4090_20261012_followup/reference/production_aa_audit.py`。']
(OUT/'production_aa_audit.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(A['conclusions'],indent=2))
