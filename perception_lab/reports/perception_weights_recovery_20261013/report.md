# 两份作者权重恢复与原生评价完成（2026-10-13）

两份完整作者权重均已在本机原子安装。HARP 404帧原始点评价、Mamba 39/81帧官方DetectionEval，以及新Mamba→GRAE 39/81帧TrackingEval与教学审计均已完成；39/81帧回放解码与真实浏览器检查也已通过。本指令的三项技术验收均达成，发布回执另行记录。

| 验收项 | 当前结论 |
| --- | --- |
| `weights_downloaded` | **true**：两份本地文件长度、完整 SHA256、CPU 结构解析及原子安装通过 |
| `native_evaluation_complete` | **true**：HARP、Mamba及新输入GRAE的实际推理与评价完成 |
| `visualization_complete` | **true**：39/81帧解码与浏览器首/中/末、5模型页全部通过 |

机器可读结论见 [status.json](status.json)。本报告是 nuScenes mini 教学诊断，不是官方全量独立泛化评测；作者预训练与 mini 重叠的可能性保留。

两份文件的来源和完整性如下。模型权重、数据、环境、第三方源码及临时下载块不提交 Git。

| 模型 | 最终本地路径（项目根目录相对） | 精确字节数 | 完整 SHA256 |
| --- | --- | ---: | --- |
| HARP-NeXt nuScenes32×480 | `perception_lab/checkpoints/harpnext_2025/ckpt_best.pth` | 64,865,210 | `6f7fc5a1c03e9123541458358b1c703babf37ffc0b8b4ea6beac6c9f8478a147` |
| MambaFusion | `perception_lab/checkpoints/mambafusion_2025/checkpoint_epoch_10.pth` | 391,545,251 | `d2454354b762a5a4b0b85af8b23c4b1f63da9357af7160aa313bc689a8d11ea6` |

HARP 使用 [作者关联的公开 HF 仓库](https://huggingface.co/samir-abh/HARP-NeXt)，固定 revision `c2cd3ce5940be23bfbc2d1cf22131fd2fd3dc149`、文件 `harpnext-nuscenes-32x480/ckpt_best.pth`；本地 SHA 与 Hub LFS SHA 完全相同。Mamba 使用 [作者 Drive 文件](https://drive.google.com/file/d/10-FBWHbsCMJseKovJwGKncJ96Flp3umM/view)，ID `10-FBWHbsCMJseKovJwGKncJ96Flp3umM`。**Mamba 没有获得作者预先公布的 checksum**：上述 SHA 是独立 GitHub Actions 节点完整下载作者文件后计算的源文件 hash，与本地完整文件逐字节哈希一致，不能称为作者公布 hash。

完整证据见 [下载清单](download_manifest.json)、[HARP 安装凭据](harpnext_atomic_receipt.json)、[Mamba 安装凭据](mambafusion_atomic_receipt.json)、[HARP 分块校验](harpnext_verified_blocks.json)、[Mamba 分块校验](mambafusion_verified_blocks.json)。CPU 检查分别确认非空 `net`（293 条）及 `model_state`（1,403 条）；模型适配器后续 `strict=True` 加载也已通过。CPU 结构解析与网络严格加载是两个独立验收步骤。

最初配置代理和系统 DNS 直连都失败；真实 DNS 的短 Range 探测曾成功，但整文件/TLS 连接仍不稳定。最终从固定作者源直接取得权重：HARP 经任务专用日本节点代理 `127.0.0.1:19098`；Mamba 经该节点完成部分块后遇到同段两次断连，切换任务专用新加坡节点 `127.0.0.1:19099` 完成余块。全部保持 TLS 证书验证，只使用任务进程配置，未修改全局 DNS/代理。私密节点配置、凭据及临时签名 URL 不发布。

下载使用精确 `206 start/end/total` 校验和完整块 SHA，拒绝错误页、错误范围、`416` 假成功及把 `200` 追加到旧文件。HARP 使用 4 个完整块，Mamba 使用 24 个完整块。Mamba 支持作者 legacy Torch 序列化头。最终合并文件通过完整长度和 SHA 检查，CPU 解析后同文件系统原子安装。旧 4,541,653 字节 `checkpoint.download` 未复用、未改动，见 [历史 partial 凭据](historical_partial_preserved.json)。并行第一版停止后，另有 5 个已完成且经过范围验证的块尚未写入总索引；只恢复这些规范命名的完整 `.block`，不接纳 `.part`，并以最终全文件源 SHA 为安装前必要条件，见 [恢复记录](unindexed_completed_blocks_recovered.json)。

[Actions 运行 38360862641](https://github.com/pyk142857/CARperception/actions/runs/38360862641) 在提交 `54acae3bcb4e326ceb9992d3043eaad4f7ab9229` 上完整取得两份作者文件，记录源 hash 并分块。最终本地载荷来自作者源的任务代理下载，**没有把 Actions artifact 载荷传完后作为安装文件**；该运行提供独立源节点全文件 hash，见 [源清单](relay_source_manifest.json)。本地双文件验收后，8 个临时 artifact 均以 HTTP 204 删除，再查询剩余数量为 0；见 [清理凭据](artifact_cleanup_receipt.json)。运行 ID、源码及清单保留。此前 Playwright 浏览器下载环境实际在本机，不能把它当作独立远端节点；真正的独立远端证据是此次 Actions runner。

恢复前基线提交为 `bddc28f51a101fd118b6e09e9f13e0bfd4f0478d`，已封存 158 份历史文本/指标文件及其哈希，见 [旧报告快照凭据](prior_snapshot/snapshot_receipt.json)。旧失败日志保留，新成功证据不会反向改写旧实验。

HARP 使用作者 CPU 预处理、batch=1、无 TTA、无训练增广，网络输入为原生 32×480、输出 16 类，严格载入 293 个状态张量。404 帧已保存原始点预测，各范围按原始有效 GT 点统计，排除追加插值点与 ignore=255，IoU 均值分母为 union>0 的类别；本次三个范围均为 16 类。

| 范围 | 帧数 | 有效原始点 | mIoU | 点准确率 |
| --- | ---: | ---: | ---: | ---: |
| scene0061 | 39 | 1,016,188 | 64.4201% | 94.4287% |
| mini_val | 81 | 2,087,603 | 55.2162% | 94.8077% |
| mini_all | 404 | 10,359,306 | 77.3849% | 94.7851% |

详见 [HARP 报告](../perception_2025_20261013/harpnext/report.md)、[原始指标](../perception_2025_20261013/harpnext/metrics.json)、[官方 lidarseg 函数交叉核对](../perception_2025_20261013/harpnext/official_lidarseg_crosscheck.json)。交叉核对复用 nuScenes-devkit 1.2.0 的 `ConfusionMatrix`，在同一混淆矩阵上的类别 IoU 最大差为 0；这不是官方全量数据提交。首/中/末的 zero/actual/permuted GT 检查保持网络输入与原始点预测一致，同时保存重复零标签控制的数值噪声。GPU 预处理候选仍未采用。

Mamba 已严格载入 1,403 个状态张量，保持作者原生六相机–LiDAR、256×704 图像、0.3×0.3×0.25m 体素与既有 sweeps 配置；必要 NumPy 兼容改动仅修复已移除别名与 ragged batch 数组构造，见 [兼容说明](mambafusion_numpy_compat.md)、[逐文件凭据](mambafusion_numpy_compat_receipt.json)。`scene0061_predictions.json` 和 `mini_val_predictions.json` 分别含39和81个准确token，原始global框直接完成本次官方评价；预测/指标哈希均绑定。

[重复噪声审查](../perception_2025_20261013/mambafusion/repeat_noise_review.json) 记录 6 个首/中/末帧、每帧 2 次重复、数值输入 hash 相同，但候选数量/排序及部分输出不同。原始按序重复误差门限未通过，**不能宣称确定性或优化等价性通过**。仅作诊断的类别感知 Hungarian 中心匹配使用距离<2m、任一分数≥0.25 定义高分框；匹配高分框中心最大差 0.624186m、分数最大绝对差 0.032042，未匹配候选仍保留统计。这些数值不是放宽后的验收阈值。已知代码中 CUDA scatter 浮点归约、LION diffusion top-k、TransFusion 提议排序可能相关，但未完成算子级因果定位。原生 mini 评价继续执行，并保留该局限，不改输入/算子、不过滤评价预测来掩盖差异。

从项目根目录可复现下载核心检查；须先提供自己可达且验证 TLS 的网络/任务代理，示例端口只是此次临时配置，私密节点信息不随报告发布：

```bash
python -m unittest discover -s perception_lab/tests -p test_recover_perception_weights_2025.py -v
# 公共固定源；代理由当前任务进程环境提供。
python perception_lab/tools/recover_perception_weights_2025.py --model harpnext --network proxy
# 本次成功并行脚本，导入同一已测试的范围校验与安装核心。
RECOVERY_PROXY=http://127.0.0.1:19099 python perception_lab/reports/perception_weights_recovery_20261013/executed_downloader.py
```

16 项离线测试全部通过，包括精确 206、错 total、200 不追加、416、断流、坏块、HTML/JSON、legacy 头、原子安装拒绝、路径越界、凭据日志脱敏、磁盘失败重试上限及 `PYTHONOPTIMIZE=1` 下结构检查，见 [测试日志](download_helper_tests.txt)。实际运行脚本两个版本及其原始/归档 SHA 见 [执行源码凭据](executed_downloader_provenance.json)，不以单元测试代替下载或模型执行。

模型步骤实际执行完成，复现命令如下（需准备固定来源权重/数据/环境）：

```bash
perception_lab/envs/harpnext_2025/bin/python perception_lab/tools/harpnext_2025.py --phase run --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/mambafusion_2025/bin/python perception_lab/tools/mambafusion_2025.py --phase run --allow-measured-repeat-noise --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/nuscenes_eval/bin/python perception_lab/tools/mambafusion_2025.py --phase evaluate --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/grae3dmot_2025.py --phase run --input mambafusion_native --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/sparse4d/bin/python perception_lab/tools/grae3dmot_2025.py --phase evaluate --input mambafusion_native --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/sparse4d/bin/python perception_lab/tools/grae3dmot_2025.py --phase audit --input mambafusion_native --config perception_lab/configs/perception_2025.yaml
```

`sparse4d` 此处仅复用已有 nuScenes 官方评价环境，不运行旧模型。本轮检测/跟踪评价与回放验收均已完成。旧 DEIM FP16 问题与未采用的 HARP GPU 预处理候选继续单独保留，不因此伪造整个项目 passed。


## 本轮检测与跟踪结果

| 范围 | Mamba mAP | NDS | 新GRAE AMOTA | AMOTP | IDS | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| scene0061 / 39 | 0.727975 | 0.673878 | 0.873912 | 0.338502 | 3 | 67 | 94 |
| mini_val / 81 | 0.588979 | 0.595775 | 0.849728 | 0.389175 | 19 | 405 | 344 |

完整各类AP、五类TP误差、失败案例及绑定证据见 [Mamba报告](../perception_2025_20261013/mambafusion/report.md)；两输入独立对照见 [GRAE报告](../perception_2025_20261013/grae3dmot/report.md)。相比封存输入，mini_val AMOTA提升约2.68个百分点，但IDS13→19、FP299→405，不能称全面改善或将检测变化全归因于跟踪器。GRAE GT sentinel、空帧、跨scene重置、前缀重放检查通过。时延含有限值Hook开销，并观察到其他项目占用GPU，不能当作独占速度基准。


## 回放、验证与交付

[回放验收报告](../perception_2025_20261013/visualization/report.md)包含39/81帧解码、真实浏览器各15个模型页/时间状态、截图及短片。完整原始点云/HARP/GT各覆盖1,354,112点（39帧）和2,812,480点（81帧）；原生Mamba显示框2976/3827、新GRAE1403/3763。全部模型层加载，逐帧清理、六相机标定投影与原始点映射核对通过。DEIM2D只覆盖既有39帧234图；81帧明确没有DEIM预测，不冒用其他帧。

[39帧本地入口](http://127.0.0.1:9094/perception_2025/) · [81帧入口](http://127.0.0.1:9094/perception_2025/mini_val/) · [19.5秒短片](../perception_2025_20261013/visualization/native_mambafusion_scene0061.mp4)。视频固定2fps仅为播放速度。

最终88项CPU测试与23项subtests通过，见[integration_tests.txt](integration_tests.txt)。独立审查核验了Mamba预测/官方指标/实际执行源码、两个GRAE family输入与指标hash；封存输出与bddc28f逐字节一致。代码、指标、失败案例、404帧压缩语义预测、截图与短片纳入项目；大型RRD按[清单](../perception_2025_20261013/artifact_receipt.json)本地重建。三项验收通过只针对本指令范围；原生重复噪声、未采用优化及mini口径限制仍然保留。
