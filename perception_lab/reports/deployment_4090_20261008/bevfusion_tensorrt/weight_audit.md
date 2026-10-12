# ONNX 与官方 PyTorch 权重对应审计

按作者导出路径在CPU重建 Conv/BN 融合，普通权重对照原state_dict与融合state_dict，SparseConv权重按官方 `(4,0,1,2,3)` 转置到输出通道在前的布局，并允许官方导出的FP16舍入。匿名导出参数按形状和数值匹配，Linear权重也检查转置。未通过这种形状/数值匹配的条目不会当作匹配成功。

| ONNX | 参数映射/总initializer | 映射数值量 | 整个张量逐位相同或FP16舍入后相同数量 | 最大绝对差 |
|---|---:|---:|---:|---:|
| camera.backbone.onnx | 126/135 | 27081398 | 36 | 2.38e-07 |
| camera.vtransform.onnx | 6/6 | 173040 | 6 | 0 |
| fuser.onnx | 33/33 | 5349120 | 12 | 1.19e-07 |
| head.bbox.layernormplugin.onnx | 62/82 | 1034398 | 62 | 0 |
| head.bbox.onnx | 62/82 | 1034398 | 62 | 0 |
| lidar.backbone.xyz.onnx | 42/42 | 2693024 | 42 | 0 |

camera剩余9个、每个head剩余20个initializer为图形shape/索引/类别单位矩阵常量，不是未找到的训练参数。所有对应训练参数一致或最大误差≤2.39e-7；这支持发布ONNX与该checkpoint权重对应，不能证明TensorRT算子实现、坐标/预处理、浮点精度或图导出计算完全等价。默认head与pluginhead的图结构仍不同。

逐张量匹配来源、张量形状、元素数、最大/平均误差、逐位一致元素数量见 [weight_audit.json](weight_audit.json)。浮点融合在不同CPU/GPU/版本的操作顺序会有舍入差异；此处不将1e-7误差写成逐位相等。

复现（不使用GPU）：

```bash
perception_lab/envs/bevfusion/bin/python perception_lab/tools/bevfusion_fuse_audit_state.py --out /tmp/bevfusion_fused_audit.pth
perception_lab/envs/gaussianworld/bin/python perception_lab/tools/bevfusion_weight_audit.py --model-dir perception_lab/checkpoints/deployment_4090/nvidia/model/resnet50 --fused-state /tmp/bevfusion_fused_audit.pth --out perception_lab/reports/deployment_4090_20261008/bevfusion_tensorrt/weight_audit.json
```

原生框格式审计：CUDA `head-transbbox.cu` 与 pinned `TransFusionBBoxCoder.decode` 均按XY=reg×8×0.075−54、dims=exp(dim)、bottomZ=height−h/2、yaw=atan2(rot0,rot1)、速度XY解码。`python.cpp`返回[x,y,bottom_z,w,l,h,yaw,vx,vy,class,score]，root runner包装为默认bottom-origin LiDARInstance3DBoxes并调用同一个 `output_to_nusc_box`，恢复gravity centre、使用旧版规定yaw=−yaw−π/2，然后LiDAR→ego。未发现重复z补偿或额外yaw转换。
