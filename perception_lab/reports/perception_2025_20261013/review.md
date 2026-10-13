# 20261013_02 恢复任务最终审查

本次恢复后的独立审查核对了完整权重、严格加载、真实Mamba120/HARP404/GRAE120推理、官方检测/跟踪与原始点评价、当前预测/指标/实际执行脚本哈希。两个GRAE family分别绑定输入与结果；封存3类产物与bddc28f逐字节一致。最终88项CPU测试、23项subtests通过。

两组39/81回放均真实解码并在浏览器核验15个模型页/时间状态；完整点数、框数、六相机投影与逐帧清理一致，无未解释应用错误。主代理查看实际HARP截图，并复核RRD哈希与status。全部新增实验结果已纳入报告目录；权重/环境/源码/RRD大文件仍排除。

保留限制：Mamba原生按序重复门未通过，明确作为已测噪声的mini诊断继续，未声称确定性/优化等价。时延含Hook同步且观察到外部GPU进程，不能作独占基准。mini存在预训练重叠可能；官方评价协议不等于全量评测。DEIM与历史优化未重跑。源码/测试/配置的git diff whitespace检查通过；原始日志、patch上下文和历史快照保留原始空白，不为格式清理改变证据哈希。

以下为恢复前的历史审查，缺权重等旧结论只描述当时状态。

# 独立 CPU 集成审查（2026-10-13）

结论：现有 GRAE 学习式跟踪结果有对应冻结输入、预测和官方评估文件；HARP-NeXt、MambaFusion 的模型缺权重阻塞表述与实际产物一致。预处理、算子 smoke、库存点数均不能当作模型精度。此次审查未运行 GPU、模型推理、训练或权重下载。此结论不表示四模型完整部署通过。

## 范围与方法

读取项目 AGENTS.md 和 `instruction/20261013_01_2025_perception_deployment.md`；检查 `tools/{grae3dmot,harpnext,mambafusion}_2025.py`、runner、五个相关测试文件及现有报告。只对已有模型产物进行静态审查、哈希/JSON核对和 CPU 单元测试；不独立重算 GPU 预测或 TrackingEval。

## 发现与修复

1. **Mamba→GRAE 路径不一致（已修复）**：检测器输出 `{split}_predictions.json`，GRAE 原默认寻找 `{split}/official_predictions.json`。GRAE 默认值已与检测器一致；主配置亦显式指定两套输入路径。增加回归测试。
2. **Mamba partial 被 runner 误判失败（已修复）**：原 CLI 因待评价事项仍在 blockers 中返回 2；runner 正确要求退出码 2 对应 blocked_resource/blocked_data。现在 partial 返回 0、blocked 返回 2、failed 返回 1；保持 runner freshness 检查不变。增加三状态测试。
3. **HARP optimize 缺少新状态证据（已修复）**：资源齐全时原 optimize 仅写 `gpu_preprocessing_audit.json`，runner 期望 `optimization.json` 或新 status。现在额外写 `optimization.json`，保留 partial/不采用结论；CPU mock 测试验证发布内容。本机 checkpoint 仍缺失，未声称已实际验证网络优化。
4. **配置可能被静默忽略（限定并修复）**：Mamba 对不支持的源码、checkpoint、作者配置、数据、报告和 split 路径覆盖显式拒绝；GRAE 对不支持的源码、checkpoint、作者配置、报告路径覆盖显式拒绝；HARP 使用配置中的 data/report/checkpoint，拒绝不支持的源码与作者配置覆盖。这里只支持已固定的 mini 实验，非通用部署平台。

以上修改未改变网络、权重、类别、几何或关联数学。现有执行 manifest 中的历史脚本 hash 应保留为实际运行版本；当前集成修改没有对应新增 GPU 成绩。

## 结果证据核对

| 模块 | 核对结果 | 边界 |
|---|---|---|
| GRAE | 冻结检测文件 SHA256 与 `frozen_reference/input_manifest.json` 一致；输出 token 与 39/81 帧 packet 集合完全相同；汇总 AMOTA/AMOTP/IDS/FP/FN 与各 `official_eval/metrics_summary.json` 相等 | 新 tracker 推理，检测输入 reused_archive；Mamba 输入尚不可用；mini 教学范围，不能作全量官方榜单成绩 |
| HARP | status 为 blocked_resource、inference_frames=0；preflight 是404帧、14,026,208原始点库存/预处理审计 | 不存在模型 mIoU；GPU preprocessing audit 明确无模型且不采用 |
| Mamba | status 为 blocked_resource、frames_inferred=0；geometry/cpu_pipeline/operator_smoke 独立保存 | 算子 smoke 不等于完整模型；官方检测精度为空 |

GRAE scene0061 的 AMOTA=0.8625687017631094、AMOTP=0.4048811070336119、IDS=5；mini_val 的 AMOTA=0.8229635568204605、AMOTP=0.43518577589429613、IDS=13。这里只核对汇总与已保存官方评估输出一致，没有重跑评估。

## GT、类别、几何与原始点

- GRAE reader 仅复制预测框、分数、类别、时间与 scene/sample；tracking_id 固定为 sentinel，未来框不进入输入，输出 ID 来自 tracker 的 instance_inds。现有 state_checks 报告 sentinel 不变性、空检测和 prefix replay 通过；本次 CPU 测试验证字段 allowlist。真实 GT 仅在独立教学/官方评价路径读取。
- GRAE 使用作者七类，global translation、wlh、四元数和 xy 速度；每 scene 建立新 predictor。未从 GT 填充属性或关联标签。
- HARP 32→16 类映射和 ignore255、缺类 N/A、原始点恰好覆盖一次、排除插值点均有 CPU 测试。默认推理用零标签，真实标签在预测后评价；已有三帧预处理 GT 不变性记录仅证明输入特征，不证明缺权重模型的预测。
- HARP 已记录边界与 point-index0 差异：CPU/向量化插值点数不同，重排 index0 改变插值。GPU 预处理不可作为等价优化采用，当前报告已如此表述。
- Mamba packet_to_info 不复制 annotations；时间差由 LiDAR timestamps 计算，未来 sweep 拒绝，短历史显式补齐。现有 geometry_check 对720个相机矩阵和1053个 sweep 矩阵，与作者参考最大误差分别约 7.43e-13、1.11e-12；本次未重跑该核验。CPU pipeline 已保存6帧无 GT 输入和固定 seed 重复一致证据。

## Runner 恢复与剩余限制

CPU 测试验证旧 passed 文件不能掩盖新进程失败、退出0但无新状态不能冒充模型完成、blocked 必须有新证据、预测或 manifest 篡改阻止恢复，以及源码/checkpoint/环境代码变化使 fingerprint 改变。配置已补 Mamba source/checkpoint/config 路径。

没有完整且可验证的 artifact manifest 时，runner 保守重跑，不执行跳过式恢复；这不是已完成全模块恢复验收。指纹与输出 manifest 本身也不替代真实输入数据来源记录。完整模型权重获取之后仍须依次验证 strict load、全部必需内部模块调用、原始点输出、native评价和回放；尤其 Mamba 内部各 USE_* 分支的实际执行及预测几何尚不能由 CPU预处理通过推断。

## 本次 CPU 测试

原始审查29项通过；增加上述集成回归测试后，**35 passed in 0.32s**。

```bash
cd /home/minglei/Desktop/CAR
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 perception_lab/envs/grae3dmot_2025/bin/python -m pytest -q \
  perception_lab/tests/test_grae3dmot_2025.py \
  perception_lab/tests/test_harpnext_2025.py \
  perception_lab/tests/test_mambafusion_2025.py \
  perception_lab/tests/test_perception_2025_runner.py \
  perception_lab/tests/test_deimv2_2025.py
```

测试环境 Python 3.10.13、pytest 9.1.1。首次尝试 nuscenes_eval 环境缺 pytest；系统 Python 3.8 的29项中28通过、1项因不支持 ast.unparse 失败。随后使用配置对应的 GRAE Python3.10环境完成全部测试；未修改任何既有环境。CPU mock 验证不代表缺权重路径完成端到端部署。

## 第二轮交付入口审查

追加检查 DEIM native optimize入口、ONNX子集AP/TF32/显存记录、失败案例和汇总器。再次只运行CPU测试，未执行模型或GPU。

- **Runner预检证据路径已修复**：两步preflight分别核对新写出的 `inputs/summary.json`、`preflight.json`；summarize使用 `summarize/status.json`。所有阶段现在要求新证据，避免退出0但未产出状态被当作完成。新增路径和freshness测试。
- **汇总器硬编码成功叙述已移除**：原版本即使报告目录为空，也会宣称DEIM5000张、GRAE120帧及固定指标；并永久声称HARP/Mamba缺权重。现在根据当前状态和具体评价文件生成指标、样本数、阻塞和正文。DEIM需状态、唯一image_ids、评价protocol的数量一致；GRAE需inference与official_evaluation通过，且指标与原始官方文件及输出帧数一致。summarize完成状态最后写入。新增空实验、失败/错范围旧指标、GRAE评价未完成和指标冲突测试。
- **失败案例范围及空检测已修复**：原版本对原生 `--limit` 结果仍评价全部COCO图像，会把未处理图像计成FN；现在严格使用native的image_ids。所有预测低于阈值时构造合法空检测集，避免 `COCO.loadRes([])` 崩溃。CPU双图像fixture验证只对选定图像计漏检。
- ONNX AP下降比较仅在native/ONNX image_ids完全相同时成立，子集与全量比较置空；TF32通过进程启动前环境变量关闭，ORT不支持的provider选项按版本省略；显存明确为采样的设备总使用量，不冒充本模型独占allocator峰值。此审查不验证这些后端的实际数值精度。
- 已向主代理报告DEIM optimize先检查TRT环境会不必要阻止ONNX路径的问题，由主代理修复。GRAE `run` 本身仍需随后 `evaluate`；其官方评价入口不会自动把run状态改为完成。复现命令及汇总应保持这一区别，不能仅凭旧metrics宣称新run完成官方评价。

第二轮全部七个相关测试文件 **40 passed in 0.36s**。在前述命令追加 `perception_lab/tests/test_perception_2025_summarize.py` 和 `perception_lab/tests/test_deimv2_badcases_2025.py` 即可复现。汇总器对当时目录只读检查得到native AP0.5086100689592818和两套GRAE指标共22行；正在重跑的ONNX旧指标正确被排除。未调用汇总main覆盖主代理正在整理的报告。

## 第三轮：配对计时与GRAE完整入口

检查新 `deimv2_paired_onnx_2025.py`：控制器逐个发送并等待回复，30轮交替AB/BA，测量期间两个模型常驻而GPU推理串行；每对比较预处理张量和原图尺寸字节hash。每种后端的相同图像输出hash仅用于重复稳定性，不宣称native/ONNX数值相同。设备显存明确包含两worker及其他驻留分配，不是单模型峰值。冷启动字段明确包含import、加载和9次warmup；hash/IPC不计入模型wall time。

修复native计时边界：旧 `decode_d2h_wall_ms` 用wall时间减CUDA事件时间，会把host调度/同步混入D2H。现在直接在CUDA完成后、CPU回传前取时间戳，另存forward/postprocess wall范围，保留GPU event时间。增加输出目录创建。CPU mock控制器测试验证60次请求的交替顺序、相同三帧选择、统计和显存标签；未启动实际worker或CUDA。

GRAE完整入口进一步修复，替代上一轮的遗留说明：公共runner现顺序执行 `run → evaluate → audit`，后两步使用 `environments.grae3dmot_eval`，未配置时默认现有 `envs/sparse4d/bin/python`。前一步失败/阻塞即停止依赖步骤，不能评价旧预测。评价/教学各写独立新receipt；评价需要当前匹配input_family的inference通过，实际TrackingEval产生新metrics_summary，且预测与packet token完全一致，才生成metrics JSON/CSV并将official_evaluation置passed。评价失败明确置failed，旧metrics文件不能通过汇总门禁。

GRAE optimize的running/partial/blocked/failed状态改写独立 `optimization.json`，不再覆盖已经验证的native inference/evaluation状态；缺资源返回码2和blocked_resource一致。CPU stub测试覆盖评价成功/失败及优化缺资源时native状态保留，没有运行真实评价。

最终八个相关测试文件 **44 passed in 0.37s**；在第二轮命令再加入 `perception_lab/tests/test_deimv2_paired_onnx_2025.py`。这仍是入口/状态/统计的CPU验证，不能替代随后实际配对GPU计时及官方评价。

## 最后补充：恢复指纹覆盖

修复源码archive位于主项目Git内部且被ignore时，`git -C archive ls-files` 返回空成功结果而漏哈希的问题。现在要求 `rev-parse --show-toplevel` 恰好等于源码目录，否则逐文件内容扫描并排除Git元数据和Python缓存。真实临时Git仓库回归测试覆盖此情况。

输入指纹除packet/inventory JSON外，重新哈希其引用的实际相机、LiDAR/sweep、COCO图像、lidarseg原始点与标签，并覆盖COCO annotations和mini元数据。测试验证不修改manifest而改变image/point原始内容也使指纹变化。此检查会增加恢复前的磁盘读取时间，以避免缓存时间戳或库存hash掩盖实际数据变化。

未来DEIM native完整5000图成功执行后才自动发布完整artifact_manifest，覆盖status、原始/压缩COCO及mini预测、官方/作者指标与protocol、profile/timings、strict load、hooks、deploy equivalence、三图gate、配置及资源记录。缺任一文件不发布complete证明；mini预测损坏使proof失败。没有为旧历史执行补造manifest；其他partial模块保持保守不跳过恢复。

此补充后相关八文件共 **47项CPU测试通过**。未执行新的GPU推理或对当前结果发布恢复证明。


最终发布前根代理复核：增加后端身份变更拒绝的4项回归测试；完整9文件CPU测试51项通过，命令及输出见[tests.log](tests.log)。当前冻结权重/config/ONNX/原生预测已执行真实哈希核对。TRT独立审查见[trt_review.md](trt_review.md)。
