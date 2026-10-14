# 冻结BEVFusion跟踪对照与GPU几何优化

当前阶段：两种原生跟踪39/81帧及官方TrackingEval已实际完成，GPU几何候选与后续pooling尚未验收。所有结果为nuScenes mini教学诊断，不能作为完整val泛化成绩。

## 原生对照

| 输入系统 | 范围 | AMOTA | AMOTP | IDS | FP | FN |
|---|---|---:|---:|---:|---:|---:|
| GRAE原生 | scene0061 39帧 | 0.862569 | 0.404881 | 5 | 67 | 104 |
| OptiPMB原生 | scene0061 39帧 | 0.888718 | 0.404887 | 1 | 39 | 110 |
| GRAE原生 | mini_val 81帧 | 0.822964 | 0.435186 | 13 | 299 | 400 |
| OptiPMB原生 | mini_val 81帧 | 0.797533 | 0.529484 | 10 | 285 | 456 |

OptiPMB在39帧上AMOTA升高且IDS/FP降低，但FN增加；81帧AMOTA下降约2.54个百分点、FN增加56，虽IDS/FP减少，仍不满足替换GRAE的收益门。保留原生GRAE，继续其GPU预处理优化。新GRAE输出与历史封存结果逐字节一致，指标来自本轮重新执行的TrackingEval，没有填用旧指标。

两系统读取完全相同的原始global检测文件（39/81帧，7789/16199框），各自保留作者类顺序、阈值、NMS和生命周期。OptiPMB还读取对应帧真实LiDAR，执行去除近车体点、框内点数和检测概率调整，不能把差异全部归因于关联模块。其配置是“作者CenterPoint默认配置应用于固定BEVFusion”，不是作者BEVFusion专用配置。GT仅进入评价与后续图示。

[输入清单](input_manifest.json) · [GRAE新指标](native_compare/grae/frozen_reference/metrics.json) · [GRAE历史一致性](native_compare/grae/historical_crosscheck.json) · [OptiPMB指标](native_compare/optipmb/metrics.json) · [OptiPMB来源](native_compare/optipmb/source_manifest.json)。各split目录含完整跟踪结果、官方每类指标；OptiPMB另含实际点数、时间戳、坐标/点云hash、FastMurty已知矩阵验证和在线状态验证。

## 执行边界

固定源检测和packet完整SHA与指令一致；对应真实原始点数为1354112/2812480，sensor→global→sensor误差小于1e-8米。主对照不重跑BEVFusion，不使用Mamba检测，不训练、不增数据。GRAE输出目录通过新增--output-dir独立指定，旧默认和历史文件不变。

CPU热点与几何语义按[契约](geometry_contract.md)推进。原高度公式、np.argsort同分规则、粗网格eligible关系、严格大于阈值、先NMS后score过滤均属于等价条件。Kernel、包含搬运的预处理、完整GRAE计时分开验收；局部更快不代表完整链路采用。

[分阶段状态](status.json) · [执行计划](execution_plan.md) · [复现命令](run_commands.md)。GPU测试串行，最终配对测试需避开外部任务；原生验证含Hook的耗时不用于最终性能结论。
