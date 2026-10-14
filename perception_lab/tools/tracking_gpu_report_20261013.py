"""Assemble measured evidence without turning pending stages into successes."""
import json,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1]/'reports/tracking_gpu_20261013'
def read(p):
 f=R/p;return json.loads(f.read_text()) if f.exists() else None

def main():
 native={name:read(path) for name,path in [('grae','native_compare/grae/frozen_reference/metrics.json'),('optipmb','native_compare/optipmb/metrics.json')]}
 full=read('benchmark/full.json');micro=read('benchmark/micro.json');pool=read('bev_pooling/results.json');visual=read('visualization/browser_validation.json')
 status=read('status.json')
 stages=status['stages']
 for stage,path in [('tilelang_kernel_executed','kernel_correctness.json'),('kernel_equivalence','kernel_correctness.json'),('kernelbench_validation','kernelbench/typed_gpu_validation.json'),('grae_full_evaluation','grae_gpu/official_equivalence.json')]:
  doc=read(path);stages[stage]='passed' if doc and (doc.get('status')=='passed' or stage=='kernelbench_validation' and doc.get('gpu',{}).get('status')=='passed') else 'pending'
 stages['cpu_semantics_profile_complete']='passed' if read('cpu_segments.json') and read('kernel_boundary_checks.json') else 'pending'
 stages['paired_performance']='passed' if full and micro and full['status']=='passed' and micro['status']=='passed' else 'pending'
 stages['bev_pooling_research']='passed' if pool and pool.get('status') in ['measured','correctness_failed'] and (not pool.get('full_model_required',False) or pool.get('full_model_status')=='passed') else 'pending'
 # Visualization owns its separate per-split/browser receipts; preserve explicit root validation status.
 status['gpu_speedup_adopted']=bool(full and full.get('candidate_adopted') and stages['kernel_equivalence']=='passed' and stages['grae_full_evaluation']=='passed' and stages['paired_performance']=='passed')
 status['status']='completed' if all(v in ['passed','not_adopted'] for v in stages.values()) else 'running'
 text=['# 冻结BEVFusion跟踪对照与GPU几何优化','', '本轮固定39／81帧BEVFusion检测，不重跑或覆盖检测输入。两种原生跟踪、真实TileLang FP64几何／NMS及GPU版GRAE官方评价已执行。以下均为nuScenes mini教学诊断，不能作为完整val泛化成绩。','', '## 原生系统比较','', '| 系统 | 范围 | AMOTA | AMOTP | IDS | FP | FN | Recall | MOTA |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for split in ['scene0061','mini_val']:
  for name in ['grae','optipmb']:
   row=next(x for x in native[name] if x['split']==split)
   text.append(f"| {name} | {split} | {row['amota']:.6f} | {row['amotp']:.6f} | {int(row['ids'])} | {int(row['fp'])} | {int(row['fn'])} | {row['recall']:.6f} | {row['mota']:.6f} |")
 text += ['', 'OptiPMB在81帧AMOTA下降2.543个百分点，FN增加56；虽然IDS减少3、FP减少14，仍未通过替换门槛。39帧AMOTA提高但FN增加6。保留GRAE；[逐类比较](native_compare/per_class_comparison.csv)和两算法原始预测／官方每类指标均已保存。', '', '两系统读取完全相同的原始global检测（7789／16199框），各自保留原生类别、NMS、阈值和生命周期。OptiPMB使用作者CenterPoint默认配置，还实际读取对应LiDAR并执行框内点数、检测概率调整；因此是原生系统比较，不能把差异全部归因于关联模块。它没有待下载网络权重；FastMurty实际库和已知矩阵验证已留证。GRAE严格加载162个状态张量，原生输出与历史封存结果逐字节一致。GT只进入后续评价和绘图。', '', '[输入哈希／原始点云／变换清单](input_manifest.json) · [GRAE指标](native_compare/grae/frozen_reference/metrics.json) · [OptiPMB来源与在线性](native_compare/optipmb/report.md) · [官方协议绑定](native_compare/grae/frozen_reference/scene0061/official_eval/protocol.json)', '', '## 几何实现与正确性', '', '原生CPU剖析覆盖120帧，452829次IoU调用。在插桩诊断中，几何IoU的嵌套时间约占91%，主要是多边形构造、求交和角点计算。这不是纯kernel基准；[完整cProfile](cpu_cprofile.txt)、[逐帧数据](cpu_profile.json)、[分段归因](cpu_segments.json)同时保留原生与插桩边界，嵌套时间不能相加。', '', 'TileLang 0.1.15在独立环境编译SM89 FP64矩形裁剪／原式3D IoU及GPU贪心抑制，ctypes库直接在原GRAE进程运行，无逐帧进程或IPC。CPU仍执行原np.argsort和100m粗网格关系；先NMS再score过滤，保留特殊高度公式、严格大于0.1和原类顺序。原网络、权重与关联不变。', '', '120帧NMS indices／order／types、模型输入字节哈希全部一致；两份最终轨迹JSON逐字节一致。GPU重新执行TrackingEval，所有整体／每类指标、配置与meta均一致，仅eval_time不同。真实帧没有CPU阈值边界回退；合成阈值用例触发的复核单独计数。', '', 'KernelBench是固定版本的“项目任务扩展”，不是标准250题成绩。保留FP64／int32／int64；整数精确比较，FP64预算atol1e-8、rtol1e-7。9个真实首／中／末代表帧有实际kernel调用、输入输出hash，冷L2 helper诊断与暖缓存基准分开。', '', '[几何契约](geometry_contract.md) · [120帧与20组fixture](kernel_correctness.json) · [无效输入／reference_failure](kernel_boundary_checks.json) · [typed适配及diff](kernelbench/typed_adapter.md) · [GPU执行](kernelbench/typed_gpu_validation.json) · [轨迹一致性](grae_gpu/equivalence.json) · [完整官方一致性](grae_gpu/official_equivalence.json)', '', '限制：当前固定capacity256，实测最大199框，使用默认0.1／1.0／0.3阈值与FP64几何。非正尺寸／非有限值仍在原reader处拒绝；极小正尺寸导致上游自IoU不足阈值、原NMS不终止，已实际超时隔离并记录reference_failure，候选显式拒绝。不能把这类输入标为等价通过。', '', '## 30轮配对性能', '']
 if full:
  text += ['三个scene的9个代表帧分别做30轮AB／BA预处理对照，device-resident geometry／suppression只用CUDA events另列。完整GRAE直接使用全部39／81帧，按真实scene从相同起点重置；双方各预热2次，关闭所有神经模块Hook。每轮从冻结JSON读取到可消费tracks，包含全部预处理、H2D、原网络、关联、D2H及Python导出，不含GT评价、磁盘写输出、视频和冷加载。**不含BEVFusion前向，不能称整条感知系统FPS。**', '', '| 范围 | CPU mean ms/帧 | GPU mean ms/帧 | 节省 | CPU/GPU P95整序列ms | 更快轮次 | 门槛 |', '|---|---:|---:|---:|---:|---:|---|']
  for split,n in [('scene0061',39),('mini_val',81)]:
   v=full['summary'][split];text.append(f"| {split} | {v['cpu']['mean']/n:.3f} | {v['gpu']['mean']/n:.3f} | {100*v['mean_saving_fraction']:.2f}% | {v['cpu']['p95']:.2f} / {v['gpu']['p95']:.2f} | {v['wins']}/30 | {'通过' if v['adoption_gate'] else '未通过'} |")
  v=full['combined_summary'];text += ['',f"合并120帧平均节省 **{100*v['mean_saving_fraction']:.2f}%**，{v['wins']}/30轮更快；两个split与合并结果均独立检查AB／BA均值改善、至少24/30胜、P95不升和均值节省≥5%。完整链路速度门：**{'通过' if full['candidate_adopted'] else '未通过，保留CPU'}**。", '', 'GPU外部compute进程每2秒和每轮边界采样；被竞争污染的轮次保留且不纳入接受结果。记录Torch allocated／reserved、进程GPU总显存和CPU lifetime高水位。未声称整机CPU隔离：观察到无关CPU任务，见host_workload_observation；AB／BA和完整原始轮次用于呈现实际变动。冷导入、模型加载、编译分别记录。', '', '[三层汇总](paired_benchmark.json) · [完整每轮CSV](benchmark/full_pairs.csv) · [9帧逐轮CSV](benchmark/representative_pairs.csv) · [搬运诊断](benchmark/transfer_samples.csv) · [监控与完整边界](benchmark/full.json)', '', 'OptiPMB原生耗时包含必要LiDAR读取／变换和点数统计，39／81序列分别3346.898／10741.198ms；这一独立运行的系统耗时不能当成与GRAE的AB／BA算法速度对照。GRAE正式CPU／GPU时延见上表，Opti与GRAE的算法精度结论见前表。']
  (R/'paired_benchmark.json').write_text(json.dumps({'status':'passed' if micro else 'partial_micro_remeasurement_pending','full':full,'micro':micro},indent=2)+'\n')
 else:text+=['仍在执行，尚无完整30轮采用结论。']
 if full and not micro:text+=['','代表帧微基准末段出现外部GPU训练进程，整组已作废并保留在benchmark/micro_rejected_*；上述完整GRAE30轮在该进程启动前完成，独立有效。代表帧三层报告待GPU空闲后重测，整体任务尚未完成。']
 text+=['','## BEV pooling与可视化','']
 if pool:
  text += [f"独立研究方向为每次forward预计算feature offset，保持上游FP16布局、原FP32 FMA累加顺序和pack10线程分配。候选计时包含新预计算kernel，不跨帧摊销。实际状态：`{pool['status']}`；稳定局部收益：`{pool.get('stable_local_benefit',False)}`；最终采用：`{pool.get('optimization_adopted',False)}`。详见[pooling报告](bev_pooling/report.md)；冻结跟踪输入未改变。"]
 else:text+=['BEV pooling尚未完成实测；在B–E发布后，按既有真实capture研究一个索引预计算候选，结果另存，绝不覆盖冻结检测。']
 text += ['', '[39帧对照入口](http://127.0.0.1:9094/tracking_gpu_20261013/) · [81帧入口](http://127.0.0.1:9094/tracking_gpu_20261013/mini_val/) · [可视化验收与固定案例规则](visualization/report.md)。完整原始点云、六相机、BEV检测、GRAE／Opti轨迹及独立GT分层显示；GPU输出相同，不重复画同样框。浏览器和解码验证分别记录，以验收状态为准。', '', '## 来源、复现与发布', '', '[阶段状态](status.json) · [真实命令／轻量runner](run_commands.md) · [隔离环境](environment/README.md) · [运行时兼容说明](environment/runtime_initialization.md) · [编译时源码及库hash](kernelbench/compile_manifest.json)。原始源码／权重／数据／环境／RRD留本地，报告保留来源与hash。', '', '两份冻结检测SHA：`dd6df55652026f2b04c7b18f9adde44123b664e8cb3706e53d6d30faca344d2f`（39），`a6fb424ccbe629f3af27172202960c4533cd48ddd887cd201f0d464ee22c56a7`（81）。原生对照首批main@fda66d0，GPU等价／完整链路30轮已发布main@70420b8；后续发布状态按status与Git远端核验记录。']
 (R/'report.md').write_text('\n'.join(text)+'\n');(R/'status.json').write_text(json.dumps(status,indent=2)+'\n')
if __name__=='__main__':main()
