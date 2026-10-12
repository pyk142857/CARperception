# 首帧真实 pooling 中间张量对照

使用之前真实执行保存的S1 `pool_capture_0.pt` 和S2 `frame_0.output.bin`，没有重新运行GPU。两者均是 camera BEV pooling 后、vtransform downsample 前的结果。源码索引核验：PyTorch B,C,D,X,Y 中D=1，按原实现collapse depth得到1×80×360×360；CUDA同为C,X,Y，不对空间轴做猜测转置。

10,368,000个数值无NaN/Inf；绝对误差平均 0.00039333、P95 0.00131989、P99 0.00405293、最大 1.450134；余弦相似度 0.99995285、归一化RMSE 0.00971368。参考|value|P99=0.213467。

预先固定atol=0.05、rtol=0.02下，**1580个数值超差（0.01524%），不是全部元素通过**。阈值不放宽。强行转置空间轴的对照MAE=0.050910，明显大于按源码定义对齐后的误差，支持当前轴序。

参考有效投影行数 1835478，CUDA interval累加贡献行数 1835477；非零BEV cell分别 97326 / 97325，support XOR=3。几何边界浮点运算和上游相机/depth的FP16均可能影响这些结果，不能凭该整体张量将差异归因于某个kernel。

独立CPU重建pinned参考预处理，与S2实际记录的第一帧FP32输入SHA256逐一一致：图像、点云、camera2lidar、intrinsics、lidar2image、img_aug_matrix全部匹配。S1首次运行没有保存输入tensor哈希，所以这是确定性管线重建核验，不能倒称首次tensor的直接历史哈希对照。见 [preprocess_audit.json](preprocess_audit.json)。

完整数值和输入捕获文件SHA256见 [pool_intermediate_frame_0.json](pool_intermediate_frame_0.json)。这是完整分支的实际中间结果，不是仅拿相同输入重复调用两个pooling算子的自我对照；后者由独立S3实验报告负责。

复现：

```bash
perception_lab/envs/gaussianworld/bin/python perception_lab/tools/bevfusion_pool_intermediate_audit.py
perception_lab/envs/bevfusion/bin/python perception_lab/tools/bevfusion_preprocess_audit.py
```

上游源码另有明确的末区间off-by-one：`camera-geometry.cu` 把最后区间终点设为`numel_geometry-1`，pooling按`i<interval.y`使用排他的终点，因此最后排序元素未累加。实际末区间`[1993722,1993727)`，排序总数1993728，遗漏的depth索引664541，其80通道贡献最大绝对值0.0494845，目标XY=(359,272)。此次未修改上游，保持S3参考累计/索引语义；这个一元素遗漏不能解释全部最大1.45误差或全部超差元素。
