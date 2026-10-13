# 2025 感知部署实测报告

本轮汇总状态 **partial**。以下指标只读取当前完成状态及匹配范围的评价证据；失败或未完成阶段的旧指标不会自动作为本次成功结果。

| 模块 | 当前状态 |
|---|---|
| deimv2 | partial |
| harpnext | blocked_resource |
| mambafusion | blocked_resource |
| grae3dmot | partial |

## 当前实测证据

- DEIMv2-S native: 5000 images; AP=0.5086100689592818
- DEIMv2-S onnx: 5000 images; AP=0.5086092492645437
- DEIMv2-S tensorrt_fp16: 5000 images; AP=0.5080690341495484
- GRAE frozen_reference/scene0061: 39 frames; AMOTA=0.8625687017631094; IDS=5.0
- GRAE frozen_reference/mini_val: 81 frames; AMOTA=0.8229635568204605; IDS=13.0

[完整指标 CSV](metrics.csv) · [JSON](metrics.json) · [状态](status.json) · [资源](resource_manifest.json) · [环境](environment.md)

## 阻塞与未完成项

- harpnext: blocked_resource; Matching complete author nuScenes32x480 checkpoint not downloaded; see resource_manifest.json
- mambafusion: blocked_resource; ["Complete author checkpoint unavailable: [Errno 2] No such file or directory: '/home/minglei/Desktop/CAR/perception_lab/checkpoints/mambafusion_2025/checkpoint_epoch_10.pth'", 'Drive streaming returned HTTP413: 391545251 bytes exceeds 268435456-byte connector limit; shell downloads incomplete.']
- GRAE MambaFusion input branch has no measured mini_val tracking output

## 方法、诊断与采用决定

[DEIM报告](deimv2/report.md) · [GRAE报告](grae3dmot/report.md) · [HARP报告](harpnext/report.md) · [Mamba报告](mambafusion/report.md)。阶段报告可能包含明确标注的历史执行快照；与当前运行状态分开理解。

DEIM失败案例采用IoU0.5/score0.25独立诊断，不是AP：[案例](deimv2/badcases/summary.json)。GRAE冻结检测输入属 reused_archive，跟踪器真实执行才计 new_measured；不得把检测输入变化归因于tracker。HARP预处理速度不代表整网FPS，点语义精度必须仅评价原始有效GT点。CUDA算子smoke不等于Mamba整网部署。AP预算通过不能代替几何与数值一致性验收。优化的配对范围、稳定性与采用决定以各阶段报告为准。

## 回放、复现与局限

回放当前状态：**partial**。[本地入口](http://127.0.0.1:9094/perception_2025/) · [验收记录](visualization/status.json)。缺少的预测应显式显示缺失，不能沿用前帧。

nuScenes仅mini教学诊断，预训练可能覆盖样本，不能视为全量独立泛化评价。COCO实际样本数见上表及image_ids。未测结果不填论文数字。原始数据、权重、第三方源码、环境及大型中间产物留本地。

[复现命令](run_commands.md) · [独立审查](review.md) · [大文件清单](artifact_receipt.json) · [原始计划](plan.md)
