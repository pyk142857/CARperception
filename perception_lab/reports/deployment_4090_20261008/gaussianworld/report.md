# GaussianWorld 原配置流式占用实测

scene-0061 的 **39/39 帧真实六相机占用推理通过**，使用官方 streaming 权重，严格加载无 missing/unexpected keys。R101-DCN、864×1600、25,600 Gaussian、200×200×16 网格、batch=1、FP32/amp=False 均保留。无 OOM，未缩小配置；没有 SurroundOcc 真值，精度评估为 `blocked_data`，不提供虚构 IoU。

| 指标 | 实测 |
|---|---:|
| 模型平均 / P50 / P95 | 220.374 / 217.023 / 226.696 ms |
| 模型首帧（独立预热后） | 218.358 ms |
| 图像读取、哈希与官方预处理平均 | 184.581 ms |
| H2D / 后处理+D2H 平均 | 7.784 / 0.363 ms |
| 输入到 CPU 类别网格平均 / P50 / P95 | 413.102 / 409.323 / 422.111 ms |
| 对应模型 / 输入到网格 FPS | 4.538 / 2.421 |
| PyTorch allocated / reserved 峰值 | 2347.300 / 9388.950 MB（十进制） |
| 帧边界 nvidia-smi 进程显存最大采样 | 9458 MiB |
| 模型构建+严格加载+GPU搬运 | 0.829 s（不含 Python 导入） |

模型常驻、eval、inference_mode；先独立首帧预热，再清空历史并按时间顺序运行。计时使用每段 CUDA 同步墙钟，模型边界包括模型内部 CPU 协方差求逆和 sparse 运算。端到端边界不含 NPZ 压缩写盘、nvidia-smi 查询或渲染。进程显存为帧末采样，不能当作连续采样的瞬时峰值。独立一帧、三帧和完整 39 帧均真实运行，日志保留。

使用逐帧 CPU→GPU 搬运，从开始便避免 eval.py 将整个 clip 搬到 GPU。首帧 history=None，后续保留官方 `history_anchor` 两个张量；几何采用每路相机自己采集时刻的 sensor→ego→global 标定，转换到当前 LiDAR。39 帧持续状态属于 **scene-stream**，不同于作者重叠 30 帧 clip 评估协议，不宣称作者官方验证指标。mini 可能与预训练集重叠。

## 数据、代码和无标签路径

- 官方仓库 commit：`b43629eaecffd5a7cbaac1a55517766e6263e4fc`。配置快照见 [official_config.py](official_config.py)。官方 streaming 权重 SHA256：`5477081191e6d7fb8ac880bf5b99230b944a7cea76b0d87175c789a51cb97be3`，298029831 bytes。[来源](https://cloud.tsinghua.edu.cn/f/4939dcc50b2a44c1b98d/)。
- adapter 直接读取 mini 官方元数据，使用固定 packets 的 sample 顺序；六相机顺序为 FRONT、FRONT_RIGHT、FRONT_LEFT、BACK、BACK_LEFT、BACK_RIGHT。官方测试图像变换：900×1600 下裁 36 像素到864×1600、无随机翻转，RGB mean/std 归一化，32倍数 padding。
- 环境 Python3.10、Torch2.1.2+cu118、MMCV2.1.0、MMDetection3.3.0、MMSeg1.2.2；相较作者 Torch2.0/MMCV2.0.1 是兼容环境变更，完整 freeze 和 SM8.9 CUDA11.8 构建日志已保留。基环境通过 .pth 只读共享，新增依赖只安装到 gaussianworld venv，未升级原环境。
- `unlabelled.patch` 只将 label.repeat、label.shape 和输出 ce_label 设为可选。shape 从原 head 注册的 gt_xyz 获取；其内容是规则网格坐标而非真值。预测仍执行原 backbone、lifter、temporal decoder、encoder、local aggregation。未创建伪标签；审计时的零 tensor 仅验证 label 分支依赖，未进入指标或产物。
- 输出占用是 uint8 XYZ 数组，范围 [-50,-50,-5] 到 [50,50,3] 米，0.5米体素，中心 min+(index+0.5)*0.5，坐标系 **当前 LIDAR_TOP**。NPZ 提供 lidar2ego/lidar2global，供共同可视化转换。类别0 noise/ignore、1..16语义、17 empty；完整类别见 summary。
- [grids_manifest.json](grids_manifest.json) 保存逐帧 token、时间戳、前序历史 token、实际文件 SHA256 和各类体素计数。39 个压缩类别网格总计不足 1 MB，已随报告发布于 `grids/`；本地 runs 保留原始输出和输入图像哈希。源码、权重、环境不提交。

## 验证与局限

三个代表帧的 adapter 几何与独立 preflight 标定契约一致，裁剪后相机内参更新检查通过；测试见 `tests/test_gaussianworld_geometry.py`，实际结果 [geometry_test.log](geometry_test.log)。39帧 grid shape/class range 均通过。首、中、末 BEV 已人工打开核对轴方向和类别图例：

[首帧](bev_00.png) · [中帧](bev_19.png) · [末帧](bev_38.png)

初次独立三帧运行与当时完整运行首三帧有3、275、1128个体素类别差异（各640000体素）；该项为保留的早期重复性审计。完整网络重跑尚不能宣称逐位确定性；浮点差异和 sparse 重复坐标行为是待验证原因，不能将其归因于标签路径。单独 label-vs-None 全网络初测 max_abs=0.667，不能将全网络非确定性混同标签影响。后续隔离审计已通过：同一批实际预测 anchors 分别送入 label/None head，logits 最大/平均差均为0、640000体素类别完全一致；固定anchors重跑head也完全一致。完整网络同输入重复则最大差1.292831、平均差0.00011209，11/640000体素类别不同，差异位于head之前，具体来源未证明。详见 [audit.json](audit.json)。

[复现命令](run_commands.md) · [摘要](summary.json) · [逐帧计时](timing.csv) · [案例](failure_cases.md)
