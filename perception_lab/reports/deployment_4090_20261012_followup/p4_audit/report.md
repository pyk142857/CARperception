# P4 既有证据复核（2026-10-12）

本次只读核查原 `deployment_4090_20261008` 证据，未启动 GPU、未重新推理或运行官方评估、未重新启动浏览器。脚本与机器结果见 [audit.py](audit.py)、[audit.json](audit.json)、[verification.log](verification.log)。此处不证明远端发布；远端状态由主任务单独验收。

|范围|帧|mAP|NDS|AMOTA|AMOTP/m|IDS|
|---|---:|---:|---:|---:|---:|---:|
|Sparse4D scene0061|39|0.5845790292|0.5453148423|0.6794865962|0.7779580852|9|
|Sparse4D mini_val|81|0.4616075236|0.4330716006|0.6375665930|0.8195541182|16|

这些数值来自已保存 DetectionEval/TrackingEval 原始 `metrics_summary.json`，是 mini 教学诊断，不是官方全量 val 指标。本次将压缩预测解压计算 SHA256，与各 split summary 一致；每帧300框，标准化 native_instance_id 与原生输出逐项相同，track_id 为 scene_token:原生ID。39/81帧 token 集合完整；mini_val 执行按 `(scene_token,timestamp)` 排序，两个场景的整体顺序与 preflight 文件不同，不能把此差别误报为丢帧。按实际执行顺序，缓存重置位置为0、41，cache_empty=true、prev_id=0；scene0061为0。原工具确实调用 InstanceBank.reset 并清 dn_metas。保存的状态恢复/首帧重置检查 ID/class 精确一致，但浮点非逐位一致，最初1e-5检查失败仍保留。整段复跑指标变化未被解释完，不能承诺确定性。

Sparse4D计时边界保持原口径：forward_with_decode_d2h 含模型、作者解码和D2H；e2e含六图读取/预处理、H2D、模型与标准化，不含最终JSON写盘、渲染和评估。mini_val模型P50/P95=24.463879/25.660879ms，端到端=217.020918/224.773215ms；这些为旧实测，本次未重新计时。

GaussianWorld的39个NPZ均实际读取并重新哈希：token/时间/前序history与清单一致，uint8 XYZ 200×200×16、类别计数与清单一致、变换矩阵有限4×4。保持25,600 Gaussian、FP32、scene-stream。固定真实anchors的label/None head审计最大误差0、类别差异0；整网重复审计11/640000体素类别不同，仍未证明来源。本次只复核保存结果。

**身份差异已追溯：** provenance中工具哈希 `9b99b79bba08c2838f7c57c8ec51a3ea840a7a788559d4df85782542221124c3`（10892字节），当前ce24482工具为 `9f4d16e013030db1e56fa713f5618732d4d87352f77df1542cf3482805c6c070`（10888字节）。从原执行会话中5条已审阅创建/编辑记录，仅在内存重建字符串，恢复的[源码快照](gaussianworld_4090_provenance_snapshot.py)与旧provenance哈希精确相同。完整diff仅第77行空白行的4个空格被删除，AST完全一致，故不影响推理语义。见[恢复步骤与diff](provenance_recovery.json)；其恢复脚本依赖本机原始会话，但提交的快照可独立重新哈希及比较。旧provenance保留，不把两个不同字节哈希改成相同。其他列出的权重、配置、patch、summary工具、run manifest/timing均匹配。

本地GT搜索遵循作者 `dataset_nusc_surroundocc_streamtest.py:63` 的真实约定：`data/surroundocc/<LIDAR_TOP原文件名>.npy`。逐token候选文件名列于audit.json。遍历项目data、作者data、/media/minglei、/mnt，未发现39帧任一精确文件名匹配；有一个挂载盘lost+found目录无权限，不能声称全机绝对不存在标签。未下载数据、未查询远端作者清单；LiDARSeg不视为占用真值。结论仍为 inference passed / evaluation blocked_data，IoU留空；即使之后找到GT，也需明确scene-stream与作者重叠30帧clip评估差别。

保存的12张模型浏览器截图、3张Rerun截图、12张模型首/中/末JPEG均解码成功并记录新SHA256。旧browser_qa首/中/末token/时间逐项与输入一致；Rerun旧QA读回frame=0/19/38、范围0..38且无panic。本次打开查看GaussianWorld中帧图，六相机、占用切片、3D视图及类别图例可见。时间轴样本如下：

|frame|sample token|timestamp/us|elapsed/s|
|---|---|---:|---:|
|0|ca9a282c9e77460f8360f564131a8af5|1532402927647951|0|
|19|b26e791522294bec90f86fd72226e35c|1532402937198682|9.550731|
|38|ed5fc18c31904f96a8f0dbb99ff069c0|1532402946797517|19.149566|

占用轴序从XYZ当前LIDAR_TOP，体素中心=min+(index+0.5)×0.5，经lidar2ego变换；BEV显示横向−ego_y、纵向−ego_x；z索引6对应−1.75m。图中完整语义图例对应类别1..16，0为ignore、17为empty。源代码每帧递归Clear模型/BEV及相机overlay，只重新绘制当前存在ID；场景切换清history，每轨迹保留最后12个观测。局限是内部history仍保留本场景失活ID，重现后会连接旧观测，未设置时间间隔截断；本次未新增动态浏览器清理测试。

4个模型MP4重新执行ffprobe计数与ffmpeg完整CPU解码，均39帧、1920×1080、2fps、解码退出0，SHA256均与旧记录一致。2fps是播放速率。RRD文件重新哈希与旧记录 `e3ec01057816ddc11e150cba3e620e6aa08e81d7cda9ecd68cb079726f09b114` 一致，未重新执行RRD runtime验证。localhost只是本机入口，GitHub以原目录代表图和复现命令为可读证据。

复现本审核（从仓库根目录，需要numpy/Pillow、ffmpeg/ffprobe；只写本目录）：

```bash
python perception_lab/reports/deployment_4090_20261012_followup/p4_audit/audit.py > perception_lab/reports/deployment_4090_20261012_followup/p4_audit/verification.log
```
