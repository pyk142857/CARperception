# GRAE 六相机框错位：冻结记录的中心高度兼容修正

## 结论

错误源于此前NVIDIA BEVFusion导出与其可视化约定不一致：原始tensor z在作者可视化中表示几何中心，而旧导出调用`output_to_nusc_box`，使用`gravity_center`再增加了半个框高。BEVFusion检测及基于其冻结输出的GRAE/OptiPMB框因此整体偏高。相机图像尺寸和标定匹配，同帧GT投影正常；无需调整相机内外参或平移图像。

本次只为该冻结实验增加显示兼容转换：global→参考ego后，减去`R_lidar_to_ego[:,2] * h/2`。沿LiDAR z轴处理，不能直接减global z；ID、类别、分数、尺寸和旋转保留。3D框、BEV、历史轨迹和六相机投影使用同一修正后的中心。GT未修正，点云仍全部加载。

## 根因证据

实际部署源码是`third_party/bevfusion_4090`，固定commit `db75150717a9462cb60241e36ba28d65f6908607`。其`tools/visualize.py`第118行先执行`bboxes[...,2] -= bboxes[...,5]/2`，再构造以底面为原点的LiDARInstance3DBoxes。因而该路径显示中心就是原始tensor xyz。此前`tools/bevfusion_4090.py:standardize`直接使用`output_to_nusc_box`的gravity_center，把同一原始z增加了h/2。GRAE读取旧冻结全局框，并继承了偏移。

采用作者约定，而非用GT拟合偏移。新函数仅用于`tracking_gpu_visualize_20261013.py`封存输入加载后的三个模型族，不适用于GT、其他模型或未来已修正的导出。

## 验证

- 回归测试先以缺失兼容转换失败，再通过：包含非零ego旋转、LiDAR俯仰、原始数据不变，以及拒绝对GT套用修正。
- 39／81帧显示检测中心与原始native tensor刚体变换独立比较，最大绝对误差约1.2e-7米，来源为历史float32加减的舍入；预算1e-6米。
- 两份新RRD均完整解码检查点云数量、帧/token顺序、模型框中心、递归清理和相机线段数量。
- 第19帧57组XY近邻匹配（距离<0.6m，预测score>0.4）的高度偏差中位数：旧0.6926m，新−0.00169m。只是诊断，不是官方指标或逐类匹配评价。
- 六相机各自使用采集时ego pose；采样异步和模型自身误差仍可能导致局部偏差。没有执行目标速度外推来掩盖残差。
- 浏览器检查与截图分别存于`scene0061/`和`mini_val/`；以其`rerun_browser_qa.json`实际状态为准。

数值、输入哈希见[alignment_audit.json](alignment_audit.json)。首／中／末六相机对照见本目录`*_before.jpg`、`*_after.jpg`。

## 回放与评测范围

[39帧对齐版](http://127.0.0.1:9094/tracking_gpu_center_aligned_20261019/) · [81帧对齐版](http://127.0.0.1:9094/tracking_gpu_center_aligned_20261019/mini_val/)。进入Rerun默认显示GRAE；背景和配色沿用原设置。

旧RRD、原始检测/跟踪JSON和历史官方评价未覆盖。已发表的GPU等价和速度结论仍针对原冻结输入，不能解释为本次修正坐标后重新跑过跟踪/官方评价；本次没有重跑模型或修改原始评测结果。旧的显示验收只验证了记录与导出的一致性，没有验证native中心语义，因此漏掉此问题。本次补上独立native坐标核验。

## 复现

从仓库根目录，沿用已安装环境和数据：

```bash
perception_lab/envs/grae3dmot_2025/bin/python -m pytest -q perception_lab/tests/test_tracking_gpu_visualize_20261013.py
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/rerun_alignment_audit_20261019.py
perception_lab/envs/rerun/bin/python perception_lab/tools/tracking_gpu_visualize_20261013.py --split scene0061
perception_lab/envs/rerun/bin/python perception_lab/tools/tracking_gpu_visualize_20261013.py --split mini_val
python perception_lab/tools/tracking_gpu_rerun_qa_20261013.py --headless --status-path perception_lab/reports/rerun_alignment_20261019/scene0061/status.json
python perception_lab/tools/tracking_gpu_rerun_qa_20261013.py --headless --split mini_val --status-path perception_lab/reports/rerun_alignment_20261019/mini_val/status.json
```

RRD和原始图像/点云继续留本地；修复代码、报告、审计和代表性截图随仓库发布。
