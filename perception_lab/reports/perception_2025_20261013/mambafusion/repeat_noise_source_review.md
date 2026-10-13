# 原生重复推理噪声：源码检查

六帧各两次重复，数值输入 SHA256 相同；预测候选数、顺序及部分低分候选会变化。严格按候选顺序比较的 0.1/0.01 临时门未通过，证据保存在 repeat_diagnostic_baseline/ 和 repeat_noise_review.json。此结论不改写为通过。

固定作者源码下可能的来源（只读检查，尚无逐层因果证明）：

- `pcdet/models/backbones_3d/vfe/dynamic_voxel_vfe.py:77,79`：启用的 DynamicVoxelVFE 调用 GPU `torch_scatter.scatter_mean`，计算点均值与高度。已安装 torch_scatter 的 scatter.py:116–120 说明 GPU 原子浮点累加不确定性。
- `pcdet/models/backbones_3d/lion_backbone_one_stride.py:865,868`：scatter_add/mean；启用的 DIFFUSION 在750–760对特征均值执行 topk20%，可能把微小变化放大为不同稀疏坐标。
- TransFusion head 对局部非极大热图置零后 argsort/top200，未指定稳定排序，类别提案会影响类别嵌入；后续分数修正与NMS也可能放大变化。

没有把未执行路径当作原因：LION v4 的重复索引scatter不在当前v2配置路径；ingroup的整数原子排序后还有规范化步骤；BEV pooling前向是区间内顺序累加；检索到的selective_scan原子操作属于反向。

本轮继续原始模型的 mini 教学诊断评价，保留全部原生预测和作者算子，不以更换算子或裁低分框消除差异。指标绑定单次完整推理的预测哈希。该决定不证明确定性，也不批准优化等价或部署精度预算。进一步定位需要逐层保存 VFE 特征/坐标/高度、LION 选中坐标和提案前热图，比较首次分歧。
