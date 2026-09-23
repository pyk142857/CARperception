# 兼容性与协议说明

1. **MMCV**：从OpenMMLab官方cu118/torch2.1预编译轮安装MMCV2.1.0，真实CUDA NMS通过。3D环境安装SpConv2.3.6-cu118，真实SubMConv3d CUDA张量输出有限。没有替换全局CUDA或驱动。
2. **缺失Lyft SDK**：MMDetection3D的全量registry导入会加载Lyft evaluator；隔离环境补装依赖后通过。原失败日志保存在logs/mmdet3d_import_ops.log，成功记录在logs/mmdet3d_import_ops_retry.log。
3. **历史sweep逆变换**：固定上游commit的LoadPointsFromMultiSweeps使用逆旋转处理点，却直接减去仍在源坐标系中的逆平移。修复为 `p @ R - t @ R`。保存补丁：configs/locked/patches/mmdet3d_sweep_inverse_translation.patch。真实第2帧第1个sweep的前100点，最大偏差由0.000382786m降至0.000000409m；含90°旋转和非零平移的回归测试先失败后通过。首帧无历史sweep，因此首帧预测未受该修复影响；第2帧PointPillars/CenterPoint已重新实际运行。未改网络、输入范围、类别或权重。
4. **CenterPoint sweeps**：官方配置要求9条历史sweeps，converter缓存含最多10条；保留所选官方配置自身的选取逻辑，NumPy/PyTorch seed均固定20260922。`input_sweeps_available=10`表示缓存可用数，不表示模型使用10条。正式协议复现前仍应审阅该版本test_mode的历史选取行为。
5. **ONNX Runtime**：GPU包安装前导入了基础环境的CPU包，执行器拒绝CPU回退。随后发现Ultralytics的CPU导出会设置进程级CUDA_VISIBLE_DEVICES=-1；将导出隔离至子进程。数值对齐时关闭TF32，并在相同图像张量上比较PyTorch GPU/ORT GPU。真实profiling事件只有CUDAExecutionProvider计算节点。原始失败记录保留；没有放宽预设allclose门槛。单图数值校验不是完整COCO精度回归，也不是TensorRT部署。
6. **Depth Anything**：作者image2tensor自动选CUDA，CPU诊断进程显式隐藏GPU以使模型与输入一致；保留作者原始预处理、518输入下限、VKITTI metric权重和max_depth=80m。稀疏深度误差只在mini首帧六相机22,055个固定有效GT像素上计算，不是nuScenes官方val，也未做中位数尺度校正。预测误差较大，不能声称迁移精度良好。
7. **未完成部分**：未创建旧版MMCV1.x/各legacy模型可运行环境，M06–M10未完成推理接入。BEVFormer tiny原配置注明至少约6700MB显存，当前外部任务占用后仅约7100MiB可用，缺乏稳定运行余量。其他模型的当前设备完整推理峰值尚未实测，不把文献显存当作本机结果。SurroundOcc完整有效权重及标签未就绪，CenterFusion官方链接404。M07/M09权重可反序列化不代表已运行。
