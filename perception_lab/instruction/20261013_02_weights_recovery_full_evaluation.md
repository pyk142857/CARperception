# 2026-10-13_02：恢复两份权重并完成整网评价

## 任务、范围与状态

BOSS授权：“派发指令打通两份权重下载，再完成整网评价。”

本指令补充 20261013_01_2025_perception_deployment.md，优先执行本轮权重恢复与原生评价。候选仍是2025方法，不换旧模型，不训练，不新增全量数据。
本地项目 /home/minglei/Desktop/CAR；单张RTX4090 24GB。参考当前报告提交 bddc28f51a101fd118b6e09e9f13e0bfd4f0478d。
本指令发布时只完成来源/传输探测，未完成完整权重下载或新模型实验。

最终目标：
1. 完整获取匹配的 HARP-NeXt nuScenes32×480 与 MambaFusion 作者权重，严格加载。
2. MambaFusion：39/81帧真实六相机–LiDAR推理及官方DetectionEval。
3. HARP-NeXt：39/81/404帧原始点级预测及共同语义评价。
4. GRAE消费新MambaFusion检测，完成39/81帧官方TrackingEval。
5. 更新同步回放、真实指标/报告/状态并发布。未采用的HARP GPU预处理和既有DEIM FP16问题不作为原生评价的前置阻塞。

先读AGENTS.md、TASK_STATE和两模块报告/下载日志。建议分支 codex/20261013_weights_recovery。
保留旧partial报告和原始失败日志；下载与GPU任务独立推进，一个权重到位即可先验收该模型。GPU运行串行。

## A. 已核实的下载路线

本轮只读核验记录：
[来源与传输探测](../reports/perception_weights_recovery_20261013/source_probe.json)。

### A1. HARP改走固定版本Hugging Face文件

模型仓库 https://huggingface.co/samir-abh/HARP-NeXt 的模型卡关联本论文与作者代码，仓库非gated，包含nuScenes和SemanticKITTI两种权重。只取以下nuScenes文件：

- revision：c2cd3ce5940be23bfbc2d1cf22131fd2fd3dc149
- 文件：harpnext-nuscenes-32x480/ckpt_best.pth
- 精确大小：64,865,210字节
- Hub LFS SHA256：6f7fc5a1c03e9123541458358b1c703babf37ffc0b8b4ea6beac6c9f8478a147
- 最终本地路径：perception_lab/checkpoints/harpnext_2025/ckpt_best.pth

在隔离下载环境检查 hf --help/版本；安装 huggingface_hub CLI时不要升级模型环境。公共文件先匿名下载，不把token写入日志。

从项目根目录：
~~~bash
hf download samir-abh/HARP-NeXt harpnext-nuscenes-32x480/ckpt_best.pth \
  --revision c2cd3ce5940be23bfbc2d1cf22131fd2fd3dc149 \
  --local-dir perception_lab/checkpoints/harpnext_2025/hf_source \
  --max-workers 1
~~~

下载得到 hf_source/harpnext-nuscenes-32x480/ckpt_best.pth，核对长度与SHA256后再原子安装到最终路径。
同一已核验文件可使用固定revision的 resolve URL：
https://huggingface.co/samir-abh/HARP-NeXt/resolve/c2cd3ce5940be23bfbc2d1cf22131fd2fd3dc149/harpnext-nuscenes-32x480/ckpt_best.pth?download=true
HF CLI或CDN失败时换普通HTTP客户端下载同一对象，校验条件不变。
不要继续猜Nextcloud目录名；不要下载SemanticKITTI文件改名；不要用HF网页/git-lfs指针当checkpoint。
原Nextcloud作者分享保留为来源交叉核对或备用；其完整包广告140,113,745字节不是nuScenes单文件大小。

### A2. Mamba使用实测可读的HTTP Range端点

- 原始文件ID：10-FBWHbsCMJseKovJwGKncJ96Flp3umM
- 作者文件名：checkpoint_epoch_10.pth
- 精确大小：391,545,251字节
- 作者目录：https://drive.google.com/drive/folders/1TqvpIHA7plzoFdnGWvFgVYr45bgz-nQ3
- 文件页：https://drive.google.com/file/d/10-FBWHbsCMJseKovJwGKncJ96Flp3umM/view
- 实测直接端点：https://drive.usercontent.google.com/download?id=10-FBWHbsCMJseKovJwGKncJ96Flp3umM&export=download&confirm=t
- 最终路径：perception_lab/checkpoints/mambafusion_2025/checkpoint_epoch_10.pth

本轮在云端网络节点实测：
- GET Range bytes=0-65535 返回206、Content-Range bytes 0-65535/391545251，读取65,536字节。
- GET Range bytes=391479715-391545250 返回206且范围/总长完全一致，读取65,536字节。
- 返回application/octet-stream及正确文件名。首字节为旧版PyTorch序列化头，不是ZIP；不得强制要求PK头。
这是首/尾探测，不是完整下载成功。metadata未返回可用作者MD5/SHA256；后续本地计算的SHA不得称为作者预先公布hash。

停止重试有256MiB硬上限的整文件连接器。采用普通HTTP客户端或gdown直接落盘，限制于本文件，不递归下载作者整个目录。
先测可达网络路径；已有HTTP代理发生过RemoteDisconnected/超时，按只读配置测试直连与已配置代理，不改全局网络或关闭TLS验证。
客户端按当前版本确认gdown --continue，或实现Range续传。示意直连探测：
~~~bash
curl -L --fail --connect-timeout 15 --max-time 45 \
  -H 'Range: bytes=0-65535' \
  'https://drive.usercontent.google.com/download?id=10-FBWHbsCMJseKovJwGKncJ96Flp3umM&export=download&confirm=t' \
  -D perception_lab/runs/weights_recovery_20261013/mamba_probe.headers \
  -o perception_lab/runs/weights_recovery_20261013/mamba_probe.bin
~~~
先创建目录；日志中的临时授权信息脱敏。代理路径有进度则继续，重复无进展应换路线，不无限循环同一超时。

### A3. 续传、中转与完整性

需要的下载辅助接口可新增 tools/recover_perception_weights_2025.py，明确标记待实现，避免把示意参数当现成命令。复用现有下载库，不开发通用下载平台。
- 下载先写.staging/.part，落盘进度、来源ID/revision、请求范围、响应状态/Content-Range、实际字节数与重试原因。
- 默认HTTP子块可取16MiB，遇到连接中断保留已验证块。每块只有完整且Content-Range匹配才登记完成。
- 206必须核对start/end/total。200忽略Range不能追加到已有part；重新以整文件模式写新staging。416先核对实际长度，不把它当成功。
- 旧 checkpoint.download 约4.5MB不是完成权重。若复用，先验证对应完整前缀内容；只看首尾或文件长度不足以证明旧partial内部无损。不一致则保留历史文件另起下载。
- 拒绝HTML/确认页/JSON错误响应，完成时全文件长度校验与SHA256，CPU解析checkpoint结构后原子安装。Mamba允许作者legacy torch.save格式；HARP必须与Hub LFS SHA256相等。
- 如作者/Drive提供额外可信checksum，记录并比对；否则保存整个源节点文件与接收端的相同SHA256，提供源ID、时间与大小证明，不伪造官方hash。
- 同一路径两次无字节进展后换可达网络节点，HTTP413硬上限立即换工具。不要把下载异常当模型失败。

若本机TLS/代理仍不可用，必须尝试不同网络节点中转，而不是重复本机失败：
1. 可使用已可用的远程下载节点；本轮云端Range探测证明该作者对象至少在一个网络节点可读。
2. 如无现成节点，允许在本公开仓库实现仅 workflow_dispatch 的临时GitHub Actions下载中转：标准Linux CPU runner、只下载固定作者公开文件、无训练/GPU、权限contents:read、不要求新增账号或付费服务。
3. 中转节点先取得完整源文件，核对391545251字节，计算全文件SHA256，再分成最多64MiB的顺序块：6块，前5块各67,108,864字节，末块56,000,931字节。
4. 每块一个artifact，附manifest（来源URL/ID、总长、完整hash、分块顺序/大小/hash、run ID）；不要把整份>256MiB ZIP再塞进原连接器。权重不提交Git、不创建GitHub Release、不上传数据集。
5. 用已授权的GitHub工具或gh按artifact逐块下载，本地核对每块后重组，最终长度和SHA256必须与中转节点一致。只完成中转而本机未落盘不算下载成功。
6. workflow只处理本轮固定公开权重，禁止任意URL输入、上传本机数据或私密凭据。下载已核验后删除/到期清理临时artifact，保留源码、manifest和日志来源。

下载与传输日志统一 reports/perception_weights_recovery_20261013/；大文件继续本地或短期中转，报告链接不含临时授权URL。

## B. 严格加载及单帧验证

完成一个权重就开始其原生验收；不能因为另一模型下载慢而停住全部任务。
复用现有代码、隔离环境、Hilbert模板和mini输入，不重做已经有真实哈希证明的准备阶段。
运行前清楚区分现有source/config/checkpoint与新下载对象，记录源码、环境、脚本和权重hash。

- HARP：Network('harpnext', nuScenes32×480配置)，checkpoint['net']，仅允许统一module前缀移除，strict=True；输出16类。
- Mamba：固定作者 mamba_fusion.yaml及完整 checkpoint['model_state']，strict=True，不删键/任意reshape/随机补参数。
- 被完整权重严格覆盖的训练预初始化下载不应阻塞推理；仅可跳过这类初始化入口，并证明全部相应参数被完整checkpoint覆盖。不能关闭实际Mamba/融合模块。
- 若源码来自作者完整archive，保留逐文件hash证明；git不能误读父项目HEAD当作作者版本。真实代码差异逐项解释，不能虚构本地clone。
- 先CPU解析和严格匹配，再GPU单帧、首/中/末。batch1，保留原生范围/分辨率/体素/sweeps与eval模式。
- Hook证据必须含Mamba高度编码、真实Mamba/scan、相机/点云及局部/全局融合；HARP含Conv-SE-NeXt、range/point融合及head。零调用不能判模型完成。
- 保存有限值、shape、显存、重复噪声、真实预测与3帧可视化。兼容失败自动定位依赖/原生源码接口，不以网络未部署为由切换旧模型。
- HARP保持作者CPU预处理参考；现有GPU候选有语义/非确定性问题，不能拿它生成本轮正式预测。

## C. MambaFusion完整推理与官方评价

现有 tools/mambafusion_2025.py --phase run 已编写但未实测，成功forward仍只写partial，缺独立evaluate阶段。补齐原生评价与runner依赖，不停在run退出0。

现有运行命令（项目根目录）：
~~~bash
perception_lab/envs/mambafusion_2025/bin/python \
  perception_lab/tools/mambafusion_2025.py --phase preflight \
  --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/mambafusion_2025/bin/python \
  perception_lab/tools/mambafusion_2025.py --phase run \
  --config perception_lab/configs/perception_2025.yaml
~~~

真实推理：scene0061=39、mini_val=81帧。最终预测token恰好覆盖，按时间序列记录，首/中/末检查不重复计入帧数。
native保存 scene0061_predictions.json、mini_val_predictions.json，已经是nuScenes global results/meta格式。

重要接口区别：
现有 bevfusion_evaluate.py/export_nuscenes_detection.py 接受的是项目ego标准框列表，并会旋转平移到global；不能把Mamba已经global的results字典直接塞进去，也不能再转换一次。
新增/扩展官方评价入口直接校验并消费Mamba nuScenes格式，复用 nuscenes-devkit DetectionEval/config_factory('detection_cvpr_2019')。
- version=v1.0-mini；scene0061用独立自定义scene-0061 split，mini_val用官方scene-0103/scene-0916。
- 同类、置信度排序、中心距离0.5/1/2/4m，原生完整预测，class range/max500，官方GT仅进入评价。
- 检查global translation、wlh、wxyz quaternion、xy velocity和10类。显示阈值不裁评价预测，不从GT填属性。
- 保存作者原始attributes；若与封存参考比较，生成独立同attribute策略副本并标注，不能静默覆盖原始预测。
- 保存官方mAP/NDS、各类AP、mATE/mASE/mAOE/mAVE/mAAE，eval config/version/命令、原始预测和GT/split/hash/token清单。
- 输出至 reports/perception_2025_20261013/mambafusion/<split>/official_eval/metrics_summary.json，或修复collector明确读取实际路径。不要忽略当前wrapper产生额外metrics/子目录与collector期待路径的差异。
- 只有本次新推理成功且本次DetectionEval成功、范围/hash一致才写official_evaluation=passed、official_metrics有效；旧指标不能掩盖新失败。

## D. HARP-NeXt整网及点级评价

~~~bash
perception_lab/envs/harpnext_2025/bin/python \
  perception_lab/tools/harpnext_2025.py --phase preflight \
  --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/harpnext_2025/bin/python \
  perception_lab/tools/harpnext_2025.py --phase run \
  --config perception_lab/configs/perception_2025.yaml
~~~

先严格加载与三帧门，再运行现存404帧，并分别汇总scene0061 39帧、mini_val81帧、mini_all404帧。
- 使用原生CPU投影/插值，关闭TTA和训练增广；不修改训练时的输入语义来追求速度。
- 每帧保存原始点数N、追加点数、预测映射。评价仅计原始有效GT点，排除插值点，noise/ignore不计。预测标签须覆盖原始点恰好一次。
- 默认网络输入不读取GT；同一原始输入的zero/actual/permuted标签验证预测独立性。先比较同输入重复噪声，再判断GT变动，不能把非确定性误判为泄漏或通过。
- 正式主指标按nuScenes lidarseg的16类映射与官方IoU定义，保存confusion/class union/IoU、缺类N/A与均值分母。原生manager包含追加点的统计只作作者辅助口径。
- 补足作者官方lidarseg评价器的mini/原始点格式适配，或与官方评价函数在相同16类混淆矩阵上交叉核对；不要新创指标替代IoU/mIoU。并保存有效点准确率。
- 39/81/404分别列出，不混均值；mini/预训练重叠可能性保留，不称独立全量泛化。
- 保存逐点预测/hash、module_hooks、gt_prediction_independence、metrics/profile和真实改善/失败图。

## E. 接通MambaFusion→GRAE及回放

固定新检测后运行作者GRAE，无新训练，保留现有frozen_reference结果，mambafusion_native单独存放。
已有命令：
~~~bash
perception_lab/envs/grae3dmot_2025/bin/python \
  perception_lab/tools/grae3dmot_2025.py --phase run \
  --input mambafusion_native --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/sparse4d/bin/python \
  perception_lab/tools/grae3dmot_2025.py --phase evaluate \
  --input mambafusion_native --config perception_lab/configs/perception_2025.yaml
perception_lab/envs/sparse4d/bin/python \
  perception_lab/tools/grae3dmot_2025.py --phase audit \
  --input mambafusion_native --config perception_lab/configs/perception_2025.yaml
~~~
此处sparse4d仅使用现有官方评价Python环境，不运行旧模型。执行前核对依赖，不改旧环境。

- 核对配置 mamba_inputs实际文件名；输入global框/分数/类别/速度/时间，GT ID为sentinel，跨scene reset。
- 分别保存39/81帧的AMOTA/AMOTP/IDS/FP/FN、各类详情、state checks、检测及跟踪输出hash。
- 新检测变化带来的指标差异不能全归因于GRAE；两输入families分表。
- 现有可视化脚本硬编码“Mamba/HARP缺失”和frozen tracking。改为按实际artifact读取，增加原生检测、两种GRAE输入选择及HARP原始点语义/独立GT。
- 复用9094/Rerun，首/中/末检查同步、坐标、点覆盖、颜色/ID及清理；每模型缺帧明确显示，不沿用前帧或旧网络假填。
- 39帧主回放必须包含新模型；81帧至少核对首/中/末和状态。保存代表截图与短片。

## F. 交付与验收

下载辅助、Mamba evaluate、runner run→evaluate、两个GRAE family状态/collector、回放读取属于必要实现；指令中的新增接口在完成代码前不得假称已存在。
新编排支持只执行本轮任务，不重跑DEIM全部5000图/已完成ONNX/TRT，也不启动旧候选或训练。
下载完成后更新resource_manifest，用本次新证据解除block；旧download_attempts保留历史，不永久硬编码“缺权重”。

reports/perception_weights_recovery_20261013/：
- report.md、status.json、download_manifest.json、attempts.json：实际完整文件与落盘、路线、失败/续传/中转记录。
- 源/接收端hash、Hub revision/LFS hash、原子安装receipt、archive/依赖变更及复现命令。
原模型目录继续保存本轮新推理、官方评价、profile、badcases、回放和root metrics/status；保存恢复前快照和旧hash，不抹除历史失败。

验收区分：
- weights_downloaded：两个完整文件已在本机，来源与长度/hash核对。
- native_evaluation_complete：两模型严格加载、真实39/81（HARP再404）、官方/点级评价和新Mamba→GRAE完成。
- visualization_complete：新结果真实回放及检查通过。
只有三者实际达成才将本指令范围标passed。已有FP16或GPU预处理候选仍可partial/not_adopted，不能为凑整个项目passed强行采用。

跑必要的不依赖外网的下载测试：206范围/错total/200忽略Range/断流/损坏块/HTML/旧Torch格式/原子安装拒绝；评价token、global避免重复变换、原始点映射与多family汇总测试。再跑实际3帧门、全范围评价、回放。单元测试不代替模型执行。
若再遇阻塞，先自行定位并尝试可实证的新路线；记录真实剩余问题，一个模型受阻仍完成另一个的评价。禁止无进展重复同一413或同一TLS超时。
按AGENTS既有授权更新README/TASK_STATE、提交必要代码/配置与可分享报告，推送并读回核对远端SHA。权重/原始数据/环境/大缓存不提交Git。

## 参考

- [HARP Hub模型卡](https://huggingface.co/samir-abh/HARP-NeXt)与固定版本Hub API：已核查文件/大小/LFS SHA256。
- [Mamba作者资源](https://github.com/AutoLab-SAI-SJTU/MambaFusion)与上述Google文件ID；本轮首尾Range实测见source_probe.json。
- [gdown官方用法](https://github.com/wkentaro/gdown)；[Actions artifact](https://github.com/actions/upload-artifact)。
