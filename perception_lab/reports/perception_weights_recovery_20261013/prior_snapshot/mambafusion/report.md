# MambaFusion：资源受阻，原生算子和 mini 输入已验证

状态 **blocked_resource**。实际检测推理 **0 帧**；没有 mAP、NDS、跟踪输入或模型加速成绩。完整作者检测权重为 **391,545,251 字节**，尚未获得。已有部分下载不能用于加载，入口按准确文件长度拒绝它。未运行随机权重检测、旧模型替代或非严格加载。

## 已实测完成

- 作者源码固定 `AutoLab-SAI-SJTU/MambaFusion@d85c2feeb19207b20e42d50421afb11c095f3efa`。完整源码包 SHA256 `47ba51695cfd030b4e78f745b11f309d41c82fc529a384eb4ceaa8a7557a8520`；550 个上游文件逐一核验，仅 `pcdet/version.py` 被作者构建脚本生成。详见 [source_manifest.json](source_manifest.json)，Apache-2.0 LICENSE 在本地作者源码根目录。
- 隔离环境 `envs/mambafusion_2025` 复用现有 Torch **2.1.2+cu118**，未改基础环境或旧 SpConv1/TensorRT。作者要求 Torch2.1/cu118，补丁版本并非其原始2.1.0；使用 CUDA11.8/GCC 构建作者原生扩展，完整版本见 [environment_freeze.txt](environment_freeze.txt)。
- 作者 OpenPCDet、窗口 CUDA、selective_scan core/ndstate/oflex、`mamba_diffv/mamba` 均实际编译成功。RTX4090 上 **10 项算子检查通过**：四种 scan 与解析递推一致、相同/相离框 IoU、点到框归属、自定义窗口执行与shape/dtype smoke、torch-scatter均值、SpConv稀疏卷积单元核和PointToVoxel体素内点数检查。窗口未进行独立参考数值对照，voxel未核验逐点身份映射；不能把这些smoke解释为完整几何等价验证。详见 [operator_smoke.json](operator_smoke.json)。完整模型 Python import 也已通过，未构造模型。算子测试不等于完整模型严格加载和模块 Hook 通过。
- 四份 Hilbert 模板就绪。rank7、rank10 下载原始作者文件；其完整可见张量与作者生成函数逐元素一致。rank8、rank9 使用**未修改的作者 `convert_to_index`** 按所需坐标分块生成，保留 rank8 的17层和 rank9 的41层、3D扫描次序；rank10为作者2D曲线。未分配、保存切片背后未使用的完整立方体存储。rank9 生成文件约86MB，不是直接把1GB作者文件改成2D模板。详见 [hilbert_manifest.json](hilbert_manifest.json)。rank8/9没有下载版逐元素对照，其依据是同一作者函数、保存范围以及rank7/10全张量对照。
- 从本轮新采集的原始路径/标定/位姿包生成 inference-only `infos_withimg`：scene-0061 **39 帧**、mini_val **81 帧**，无 GT 字段，不生成训练数据库。720个相机变换与作者 `obtain_sensor2top` 对比、1053个历史扫变换与共同采集包对比，最大误差约1e-12；详见 [geometry_check.json](geometry_check.json)。
- 在未修改的作者 `NuScenesDataset(training=False)` 上实际预处理两个范围各首/中/末帧，共 **6 帧**，六相机张量为6×3×256×704，点为N×5，全部有限，固定随机种子后的重复点张量相等，输入不含GT。详见 [cpu_pipeline.json](cpu_pipeline.json)。这6帧只有CPU数据预处理，没有检测推理。

## 原生输入口径

保留六相机顺序 CAM_FRONT、CAM_FRONT_RIGHT、CAM_FRONT_LEFT、CAM_BACK、CAM_BACK_LEFT、CAM_BACK_RIGHT；作者图像读取包含 RGB→BGR 转换，再执行原生 resize/crop/normalize。点特征为XYZ/intensity/**timestamp**，不把ring当timestamp。范围[-54,-54,-5,54,54,3]、体素[0.3,0.3,0.25]、10 sweeps、200 proposals；完整展开配置见 [resolved_scene0061.json](resolved_scene0061.json)、[resolved_mini_val.json](resolved_mini_val.json)。

历史不足时沿作者规则补到9条历史输入：有历史则重复最早条目，无历史则重复当前扫且时间差为0；历史扫读入仍执行作者近车体点移除。记录的真实历史数与填充后长度分别保留在 [mini_infos.json](mini_infos.json)。作者test shuffle和历史随机排列未静默关闭；验证固定种子保证复现，不声称不同种子严格不变。

## 完整权重阻塞

[作者公开资源目录](https://drive.google.com/drive/folders/1TqvpIHA7plzoFdnGWvFgVYr45bgz-nQ3) 列出 `checkpoint_epoch_10.pth`，ID `10-FBWHbsCMJseKovJwGKncJ96Flp3umM`。已实际尝试 gdown、HTTP代理/直连、usercontent下载/续传、Range请求及Google Drive连接器：

- 本地网络下载只获得不足完整大小的部分二进制，其文件名保留为`checkpoint.download`，没有冒充完成权重。
- Drive连接器内联方式在返回大文件时序列化失败，没有成功物化文件。
- 流式方式明确返回 **HTTP413：391545251 bytes exceeds the 268435456-byte limit**。

详见 [download_attempts.json](download_attempts.json)、[resource_manifest.json](resource_manifest.json)。本轮数据可用，主要阻塞是完整权重传输；strict load、关键模块Hook、39/81帧预测、官方DetectionEval、A/B优化和新模型回放均未完成。后续网络恢复或提供可读完整同一作者文件后，需从严格加载继续，不能把预处理和算子通过当部署通过。

## 复现

在项目根目录执行：

```bash
python3 perception_lab/tools/mambafusion_2025.py --phase prepare
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase hilbert
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase geometry-check
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase cpu-pipeline
# 独占GPU时执行算子检查：
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase operator-smoke
python3 -m pytest -q perception_lab/tests/test_mambafusion_2025.py
# 当前应返回非零 blocked_resource：
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase run
```

`--phase run` 的原生模型路径已编写，但因完整权重缺失未验证；即使将来生成预测，也会先标partial，要求独立官方评价及回放验收。`--phase optimize` 在原生正确性和评价通过前拒绝优化。本报告不包含官方全量评测结果。
