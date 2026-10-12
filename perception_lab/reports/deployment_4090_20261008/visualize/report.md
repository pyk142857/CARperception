# 单卡部署统一回放与真实渲染核验

4 个已实测模型，scene-0061 全39帧。导出/视频解码通过；HTML交互核验：passed；Rerun真实渲染及时间轴核验：passed。

本机入口：[同步逐帧复核页面](http://127.0.0.1:9094/)，页面有交互式Rerun和各模型MP4链接。原9092工作台保留。Rerun实体路径分别为 `models/<model>`，BEV与六路相机投影也按模型分开，可单独开关。没有将新预测混入历史模型默认记录。GT来自官方annotation，保存在独立ground_truth实体及单独tab；不进入模型视频或预测。

|模型|代表帧|视频实解码帧数|播放速度|
|---|---|---:|---:|
|bevfusion_reference|[首帧](bevfusion_reference/frame_00.jpg) · [中帧](bevfusion_reference/frame_19.jpg) · [末帧](bevfusion_reference/frame_38.jpg)|39|2 fps|
|bevfusion_tensorrt|[首帧](bevfusion_tensorrt/frame_00.jpg) · [中帧](bevfusion_tensorrt/frame_19.jpg) · [末帧](bevfusion_tensorrt/frame_38.jpg)|39|2 fps|
|sparse4d|[首帧](sparse4d/frame_00.jpg) · [中帧](sparse4d/frame_19.jpg) · [末帧](sparse4d/frame_38.jpg)|39|2 fps|
|gaussianworld|[首帧](gaussianworld/frame_00.jpg) · [中帧](gaussianworld/frame_19.jpg) · [末帧](gaussianworld/frame_38.jpg)|39|2 fps|

视频以ffprobe统计帧数，另用ffmpeg全片解码至null，真实覆盖39个固定sample_token；2fps是播放速度，不是推理速度。全部原生/标准化模型预测、官方评估仍保留完整分数；显示过滤仅score≥0.25。视频有六路相机、ego BEV、固定正交3D视角，Sparse4D框及轨迹按原生ID配色；GaussianWorld显示所有非empty/ignore的语义体素中心和16类图例，原网格未抽样。GaussianWorld视频左侧BEV是原LiDAR z第6层（中心-1.75m、厚0.5m）的语义切片，再转到ego；3D保留全部非空体素，Rerun还提供独立切片tab。

Rerun保存完整当前LiDAR作为上下文，camera-only模型不因此变成融合推理。每相机以自己的ego_pose/calibration投影到当前LiDAR时刻ego坐标；未额外补偿目标跨曝光时刻运动，也未做遮挡推理。框采用gravity中心/wlh/wxyz；轨迹先积累global坐标，再转当前ego，避免自车运动造成假轨迹。GaussianWorld XYZ体素中心按min+(index+0.5)*0.5计算，再由NPZ的lidar2ego转ego；0 ignore、17 empty不显示。每帧清理模型、BEV与相机投影实体；场景切换清空轨迹历史。

Browser插件未安装，使用系统Chrome与Playwright。HTML核验模型切换、首/中/末按钮、token与实际图片加载、桌面及较窄视口；截图和browser_qa.json保留。真正Rerun加载RRD并渲染6相机/3D/BEV，通过SDK附带WebHandle API选择frame时间轴并暂停/seek到0、19、38，检查读回值及panic状态，截图与rerun_browser_qa.json保留。该QA仅在浏览器HTML中暴露原WebHandle对象，未改数据/渲染算法。调试期间CPU SwiftShader的WebGL仅支持6个color attachments，WebGPU软件后端可渲染。最终核验使用物理RTX4090的WebGL2（8个attachments），对应正常交互式入口；记录实际renderer字符串。最终视觉验证与模型性能测量分开。

RRD执行官方CLI `rrd verify`，日志见rrd_verify.log。大型RRD、全部逐帧JPEG与MP4留本地outputs/rerun/deployment_4090；source/视频/RRD路径及SHA256见summary.json，提交代表图和截图。失败案例分别见[Sparse4D](../sparse4d/scene0061/failure_cases.json)、[BEVFusion](../bevfusion_reference/scene0061/failure_cases.json)、[GaussianWorld](../gaussianworld/failure_cases.md)；这些教学诊断不是官方IDS，原因未证实者标为假设。

## 复现

从仓库根目录执行；只传入已完成的真实预测路径：

```bash
perception_lab/envs/rerun/bin/python perception_lab/tools/deployment_visualize.py \
  --prediction sparse4d=perception_lab/outputs/sparse4d/scene0061/predictions.json \
  --prediction bevfusion_reference=perception_lab/reports/deployment_4090_20261008/bevfusion_reference/scene0061/predictions.json \
  --prediction bevfusion_tensorrt=perception_lab/outputs/bevfusion_tensorrt/scene0061/predictions.json --serve
/usr/bin/python3 perception_lab/tools/deployment_visualize_qa.py
/usr/bin/python3 perception_lab/tools/deployment_rerun_qa.py
perception_lab/envs/rerun/bin/rerun rrd verify perception_lab/outputs/rerun/deployment_4090/deployment_4090.rrd
python3 perception_lab/tools/deployment_visualize_report.py
```

只启动已有结果服务：`perception_lab/envs/rerun/bin/python perception_lab/tools/deployment_visualize.py --serve`。
