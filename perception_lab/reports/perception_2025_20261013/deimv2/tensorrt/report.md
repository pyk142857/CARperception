# DEIMv2-S TensorRT 10.6 FP16 实测

TensorRT 10.6 FP16 engine 已构建，完成 COCO val2017 同一5,000张图、1,500,000个完整候选的共同 pycocotools COCOeval，以及30轮交错原生/后端配对计时。计时前已对首/中/末每张图在两条路径各预热10次。AP下降低于0.005预算，配对端到端有稳定收益；FP16仍会改变候选置信度与阈值两侧的候选集合，不能声明输出严格等价。原生仍作为默认与参考。

| 指标 | 原生 FP32 | TensorRT FP16 |
|---|---:|---:|
| COCO AP | 0.508610069 | 0.508069034 |
| AP75 | 0.550600045 | 0.550246772 |
| APs | 0.313784110 | 0.312878515 |
| 配对 E2E mean / ms | 16.66472 | 7.52568 |
| 配对 E2E P50 / ms | 16.58987 | 7.52350 |
| 配对 E2E P95 / ms | 17.65991 | 8.14273 |
| 配对 E2E std / ms | 0.74830 | 0.40908 |
| GPU forward+post mean / ms | 10.91899 | 1.82671 |
| 直接测量 D2H wall mean / ms | 0.068355 | 0.084620 |

AP绝对下降0.000541035，即0.0541个百分点。30轮每一轮后端端到端均更快，平均节省9.13904ms，约2.21倍端到端速度。此计时仅覆盖首/中/末3张输入轮换、batch1、热文件缓存的配对实验，不把全量单次序列时间当作配对优化证据，也不推广到其他硬件、batch或全部场景。

精度检查包括输入位级一致、全部输出有限、框宽高非负、80类映射范围及score∈[0,1]。首/中/末高置信度框按类别匹配的最大xywh分量差为2.324/0.569/1.232px。全量高置信度按类别强制二分匹配的P95类别最大分量差1.207px；最大588.643px不能解释为同一物体发生该位移，因为TopK及阈值改变后强制匹配可能配到完全不同的候选。

最差3帧的独立IoU门控审计见 `worst_three_iou_audit.json`：

| COCO image_id | 原强制配对最大分量差 / px | 该配对IoU | IoU≥0.5对应数 | 对应框最大分量差 / px | 未配对数（两边合计） |
|---|---:|---:|---:|---:|---:|
| 198805 | 588.643 | 0 | 15 | 0.727 | 2 |
| 114907 | 489.949 | 0 | 42 | 0.797 | 3 |
| 475678 | 486.490 | 0 | 24 | 1.369 | 2 |

这些未配对的高置信度框，在另一端完整300候选中均能找到同类别、IoU≥0.5但score<0.25的对应框。差异也并非全是微小舍入：198805的某车score从0.2802降至0.1807，114907的某人从0.1127升至0.3203。该诊断不假设decoder query身份，也不等同于物理运动或严格模型等价。保留完整候选与置信度差异供复核；AP预算通过只是整体精度条件之一。

Engine构建实测371.37秒；ONNX、engine、依赖、输入哈希和版本均保存。TensorRT layer precision policy为启用FP16、禁用TF32；输入输出FP32/INT64，不能宣称全部层均FP16。作者默认部署式构建的LayerNorm FP16警告原样保留，没有强改LayerNorm数学。执行使用默认CUDA stream，TensorRT警告可能附加同步，现有测量包含这项开销。

完整结果：`status.json`、`official_eval/metrics.json`、`paired_benchmark.json`、`three_image_geometry.json`、`all_image_geometry.json.gz`、`geometry_summary.json`。环境、来源与复现步骤见 `environment.json`、`packages.json`、`build.json`、`reproduce.md`。显存采样含设备其他驻留进程；Torch统计不覆盖TensorRT内部申请，不能当作进程独占峰值。原生构造日志中的“Training ViT-Tiny from scratch”是作者初始化打印，后续严格加载完整checkpoint再推理，本实验没有训练。

本表使用完成预热与身份核验修正后的最新30轮，旧结果存于 `historical_first_image_warmup/`，中间复核运行存于 `historical_benchmark_guard_only/`。D2H现从GPU结束事件同步后、紧邻`.cpu()`调用前直接计时，包含转NumPy，不再通过wall time减GPU event推算。完整5,000图的历史 `timings.json` 未重跑，其旧decode_d2h_ms字段是wall减GPU event残差，仅作历史记录，不用于本表。

`benchmark_identity.json`核验了当前checkpoint、原生配置与155个固定源码/配置文件，以及作者/实际导出脚本哈希；这些分别与原生strict_load、resource_manifest和原始export_identity相符，当前ONNX及engine继续与build.json一致。这是明确标记的事后交叉核验：原始export_identity没有绑定启动时checkpoint/ONNX哈希，不能追写成原始receipt已有这些证据。以后新构建会把native输入身份写入build.json，加载评价/配对前比较；已有engine保持原始构建记录。新计时记录含当前运行代码SHA256，模型计算未改变，所以没有重跑完整5,000图评价。
