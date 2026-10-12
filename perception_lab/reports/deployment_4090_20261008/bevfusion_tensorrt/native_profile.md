# TensorRT 逐阶段计时诊断

scene-0061首帧：读取最新日志最后一个完整计时块，前3块为预热。逐段CUDA event同步会扰动流水线；本表用于定位耗时，不替代39/81帧正常模式计时。输入图像已在CPU预处理，Normalization跳过。

|阶段|毫秒|本表占比|
|---|---:|---:|
|[NoSt] CopyLidar|0.27904|3.61%|
|Lidar Backbone|1.61603|20.92%|
|Camera Depth|0.01603|0.21%|
|Camera Backbone|2.10864|27.30%|
|Camera Bevpool|0.22838|2.96%|
|VTransform|0.20368|2.64%|
|Transfusion|0.75878|9.82%|
|Head BoundingBox|2.51389|32.54%|

含CopyLidar合计 **7.72447 ms**；上游Total **7.445 ms** 排除标记NoSt的CopyLidar。图像传输发生在Python绑定进入core之前，不在本表；框回传及解码计入Head BoundingBox。CPU数据读取/预处理、几何更新和标准化见各split timings.csv，本表不代表完整端到端耗时。

相机backbone、检测head和LiDAR backbone是主要项；单独优化pooling不能按其kernel加速比等比例加速完整感知流程。

日志SHA256 `7f95d7f38038ee71ba132958496a1fa4e6c894a26f41e4d04a7fc8685bf170d3`；共4个完整块，使用第4块。

复现：
```bash
cd perception_lab
source tools/bevfusion_trt_env.sh
envs/bevfusion/bin/python tools/bevfusion_trt_4090.py --packets reports/deployment_4090_20261008/inputs/scene0061_packets.json --out runs/deployment_4090_20261008/trt_profile --limit 1 --warmup 3 --profile > reports/deployment_4090_20261008/bevfusion_tensorrt/native_profile.log 2>&1
python tools/bevfusion_native_profile_report.py
```
