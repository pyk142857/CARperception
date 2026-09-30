"""Summarize synchronized real-entrypoint timings without treating normalized sums as end-to-end latency."""
from pathlib import Path
from collections import Counter,defaultdict
import json,csv,hashlib,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
data=json.loads((HERE/'jobs.json').read_text());jobs=data['jobs'];assert all(j['returncode']==0 for j in jobs)
by=defaultdict(Counter);stages=Counter();detail=[]
for j in jobs:
 p=json.loads((HERE/(j['name']+'_phases.json')).read_text());c=Counter()
 for e in p['events']:assert e['seconds']>=0;c[e['phase']]+=e['seconds']
 # Boundary timers cover worker imports/setup through final export, but not
 # interpreter startup/shutdown; report that residual explicitly.
 residual=j['wall_seconds']-sum(c.values());assert residual>=-.1
 c['process_residual']=max(0,residual)
 group=j['name'].rsplit('_',1)[0] if j['name'].startswith(('pointpillars_','centerpoint_')) else j['name']
 by[group].update(c);stages.update(c)
 for phase,seconds in c.items():detail.append(dict(job=j['name'],module=group,phase=phase,seconds=seconds))
with (HERE/'phases.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
labels={'imports_setup_weights':'导入、元数据、模型加载','input_decode':'图像读取/解码','adapter_validation':'适配、校验/哈希','input_preprocess':'输入读取与预处理/H2D','inference_api':'推理API（含内部预/后处理）','inference_api_including_input':'分割API（含读图/预后处理）','inference_api_with_point_decode':'LiDAR分割推理及点解码/D2H','postprocess_export':'后处理、校验与文件输出','postprocess':'结果适配','evaluation_export':'诊断评估与结果输出','render_export':'离线图像/视频绘制输出','recording_build':'Rerun记录构建','recording_flush':'Rerun刷新写盘','tracking_with_coordinate_adapters':'跟踪及坐标适配（含复跑）','finalize':'汇总、校验与收尾','process_residual':'解释器启动/退出等剩余开销'}
wall=sum(j['wall_seconds'] for j in jobs)
text=f'''# 全感知离线流程耗时剖析

日期：2026-09-30。实际运行全部{len(jobs)}个测量任务，均成功退出。任务进程总耗时 **{wall:.2f}秒**；控制脚本测得总墙钟 **{data['total_wall_seconds']:.2f}秒**。

## 口径：当前没有单一实时流水线

项目由独立进程/环境运行各模块，PointPillars/CenterPoint逐帧加载模型，相机脚本保存图片和数组，跟踪脚本包含一致性复跑及MP4输出，Rerun是另一个离线导出步骤。因此本报告区分真实离线任务总成本和模块推理API成本，不将归一化之和称为实测端到端延迟或FPS。

本轮GPU推理实测：前3个时刻，YOLO、米制深度、SegFormer各18张相机图；PointPillars、CenterPoint各3个独立进程；MapTR、Cylinder3D各3帧。跟踪和Rerun导出使用现有39帧结果单独测量：它们不是本轮所有推理结果串接的输出。全部模块联合启用属于教学工作量，PointPillars与CenterPoint是替代性3D检测方案，不意味着部署时必须双跑。

本轮视觉模块设备显式设置CUDA，与早期CPU smoke日志不同；没有用历史耗时凑数。数据/infos/权重已在本地，模型进程新启动，文件系统缓存未清空。没有浏览器前端渲染、网络传输、模型训练或官方评估计时。

## 实际离线任务时间占比

下表分母是本次混合测量工作量的全部进程墙钟时间，样本数不同，不能据此比较单帧性能。

| 模块 | 任务范围 | 实测秒数 | 总任务占比 |
|---|---|---:|---:|
'''
scopes={'yolo':'18图/3时刻','depth':'18图/3时刻','segformer':'18图/3时刻','pointpillars':'3帧、逐帧加载','centerpoint':'3帧、逐帧加载','maptr':'3帧/18图','lidarseg':'3帧','tracking_39':'39帧、复跑+视频','rerun_export_39':'39帧记录导出'}
for name,c in by.items():text+=f"| {name} | {scopes[name]} | {sum(c.values()):.3f} | {sum(c.values())/wall:.1%} |\n"
text+='\n## 各阶段总占比（相同混合工作量）\n\n| 阶段 | 秒数 | 占比 |\n|---|---:|---:|\n'
for phase,sec in stages.most_common():text+=f'| {labels[phase]} | {sec:.3f} | {sec/wall:.1%} |\n'
text+='\n## 推理API：按一个传感器时刻归一化\n\n一个时刻包含六路图像与一帧LiDAR。下表把相机的18张调用总时间除以3，其他推理按3帧计算。GPU在计时边界同步；不是仅测CUDA任务提交时间。\n\n| 模块 | 推理API ms/时刻 | 说明 |\n|---|---:|---|\n'
api={}
for name,c in by.items():
 v=sum(sec for key,sec in c.items() if key.startswith('inference_api'))
 if not v:continue
 api[name]=v/3*1000
 note='逐帧进程首次推理，含冷启动效应' if name in ['pointpillars','centerpoint'] else '含第一次调用；非纯稳态benchmark'
 text+=f'| {name} | {api[name]:.2f} | {note} |\n'
warm_rows=[]
for name in ['yolo','depth','segformer','maptr','lidarseg']:
 events=json.loads((HERE/(name+'_phases.json')).read_text())['events']
 values=[e['seconds']*1000 for e in events if e['phase'].startswith('inference_api')]
 assert len(values)==(18 if name in ['yolo','depth','segformer'] else 3)
 ms=sum(values[1:])/len(values[1:]);factor=6 if name in ['yolo','depth','segformer'] else 1
 warm_rows.append(dict(module=name,api_ms_per_call=ms,ms_per_sensor_tick=ms*factor,samples=len(values)-1,scope='exclude first call; consecutive real inputs'))
for name in ['centerpoint','pointpillars']:
 events=json.loads((HERE/(name+'_resident_phases.json')).read_text())['events']
 values=[e['seconds']*1000 for e in events if e['phase']=='resident_inference'];assert len(values)==20
 ms=sum(values)/len(values);warm_rows.append(dict(module=name,api_ms_per_call=ms,ms_per_sensor_tick=ms,samples=len(values),scope='same first frame repeated, 3 warmups, 20 calls'))
(HERE/'warm_api.json').write_text(json.dumps(warm_rows,indent=2)+'\n')
text+='\n## 去除首次调用后的API参考值\n\n| 模块 | API ms/次 | 六相机/单LiDAR时刻归一化 ms | 样本次数 |\n|---|---:|---:|---:|\n'
for r in warm_rows:text+=f"| {r['module']} | {r['api_ms_per_call']:.2f} | {r['ms_per_sensor_tick']:.2f} | {r['samples']} |\n"
text+='\n相机模块取后17次，MapTR/LiDARseg仅后2次真实输入；CenterPoint/PointPillars另行预热3次、同一首帧重复20次。样本/缓存条件不同，属于排除首次开销后的诊断参考，不是完整稳态多场景基准。常驻探针不计入上面的主任务占比；它们的执行记录见resident_jobs.json。\n'
text+='''
这些API并非裸网络forward：YOLO含预处理/NMS；SegFormer含读图、滑窗和输出恢复；Depth含图像预处理/缩放恢复；3D test_step含数据预处理、voxelization与解码/NMS；Cylinder3D本段含softmax、逐点取值及CPU传回。因此不能进一步把本表解释为各网络纯GPU算子速度。

## 建议优化顺序

1. **先把3D模型改为进程内加载一次、连续推理**：现有逐帧脚本将导入和权重加载反复计入；这是架构开销，通常不需要修改网络算子即可去除。
2. **将图片绘制、压缩保存、视频编码和证据哈希移出感知关键路径**：它们是当前教学导出的一部分，不能归咎于模型推理慢。
3. **再针对占时较多的推理API做内部profiler**：分别观察预处理、H2D、网络、解码/NMS、D2H；在本报告可分辨的边界内，按实测较大项优先，不凭模型名称排序。
4. **跟踪优化收益单列**：上一轮同输入精确一致的CPU跟踪平均0.77ms/帧；此处原track_mini还包括坐标变换、复跑、画图和视频，不能拿整个脚本时间与0.77ms直接比较。
5. **Rerun导出与页面渲染分开**：本轮测到的是记录构建和写盘，浏览器交互/渲染延迟仍需单独测量。

## 复现与证据

```bash
python3 perception_lab/tools/profile_perception_pipeline.py
python3 perception_lab/tools/profile_perception_pipeline.py --resident-probes
perception_lab/envs/mmdet3d/bin/python perception_lab/reports/pipeline_timing/build_report.py
```

脚本按原入口源码的固定锚点注入互斥阶段边界，运行时显式同步已初始化CUDA；不改模型、配置、输出算法或正式结果。完整调用、退出码和源码哈希见[jobs.json](jobs.json)，逐任务阶段见`*_phases.json`，汇总见[phases.csv](phases.csv)，日志见同目录`.log`。原始模型输出只保存在`outputs/pipeline_timing`及隔离的`runs/pipeline_timing`，不覆盖现有回放。

阶段之和与进程墙钟的剩余差作为解释器等开销单列。重复同步可能影响原本的CPU/GPU重叠，所以这些是同步诊断耗时；本轮没有CPU/GPU流水并发。样本较少，不报告稳定P95或生产吞吐量。

![离线阶段占比](phases.png)
'''
(HERE/'report.md').write_text(text)
(HERE/'summary.json').write_text(json.dumps(dict(task_wall_seconds=wall,controller_wall_seconds=data['total_wall_seconds'],phases_seconds=stages,module_seconds={k:sum(v.values()) for k,v in by.items()},inference_api_ms_per_sensor_tick=api,profiler_sha256=hashlib.sha256((ROOT/'tools/profile_perception_pipeline.py').read_bytes()).hexdigest(),checks=dict(all_jobs_passed=True,nonnegative_phases=True,wall_residual_accounted=True)),indent=2)+'\n')
try:
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 vals=stages.most_common();fig,ax=plt.subplots(figsize=(10,6));ax.barh([k for k,v in vals],[v for k,v in vals]);ax.invert_yaxis();ax.set(xlabel='Seconds (mixed offline diagnostic workload)',title='Measured phase costs / GPU-synchronized boundaries');fig.tight_layout();fig.savefig(HERE/'phases.png',dpi=140)
except ImportError:print('Run with mmdet3d Python to generate plot')
print(json.dumps(dict(total=wall,api_ms=api,phases=stages),indent=2))
