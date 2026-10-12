# TensorRT 8.6 / CUDA-BEVFusion 构建记录

已用官方 TensorRT release/8.6 的 trtexec/common 源码和 pip TensorRT8.6.1 动态库构建本机 `envs/tensorrt/bin/trtexec`。CUDA-BEVFusion `libbevfusion_core.so`、`libpybev.so`、`libcustom_layernorm.so` 使用 CUDA11.8、SM89、SpConv11.4、Python3.10 编译；libpybev 实际导入成功。原环境未全局升级。

官方 TensorRT include、ONNX parser include 来自本地锁定源码；protobuf3.6.1 的 headers/protoc 从 `.deb` 解包到 `envs/tensorrt/build-deps/extracted`，与系统 libprotobuf.so.17 一致。原 ONNX `.pb.cpp/.h` 由这套 protoc 重生成，因此不提交数万行机器生成差异，复现脚本保留生成命令。

[build_provenance.json](build_provenance.json) 记录上游 commit、编译产物与 `.deb` 哈希（初始构建、BEV pooling 候选修改之前）。编译日志见 [build_trtexec.log](build_trtexec.log)、[configure_pybev.log](configure_pybev.log)、[build_pybev.log](build_pybev.log)、[build_pybev_capacity500k.log](build_pybev_capacity500k.log)、[import_pybev.log](import_pybev.log)。

配置与兼容修改见 [build_compatibility.patch](build_compatibility.patch)：环境脚本改为加载项目内 `tools/bevfusion_trt_env.sh`，使用ResNet50/FP16、cuDNN8及正确的库路径；Python voxelization.max_points设500000，core容量读取该参数。scene0061实际输入点数244111–255918（中位250909），mini_val207998–264273（中位246377），容量覆盖所有实际帧。

复现 CPU 编译：

```bash
bash perception_lab/tools/bevfusion_trt_setup.sh
```

其组成命令已分别实际执行；整体脚本语法检查通过。依赖本项目已准备的 TensorRT env、官方 source/submodules、SpConv动态库及 `.deb` 文件。

构建 GPU engines：

```bash
cd perception_lab/third_party/Lidar_AI_Solution/CUDA-BEVFusion
bash tool/build_trt_engine.sh
```

原官方构建命令使用 --fp16、输入输出FP16 CHW、--memPoolSize=workspace:2048、detailed profiling。四个子网络都是固定shape；LiDAR点数由CUDA/SpConv路径处理，不是这四个TensorRT engine的动态点数profile。构建脚本内部自动随机输入测速仅验证engine运行；它不是本项目mini真实融合预测性能。

初次启动时 `model/` 尚未建立指向真实权重目录的symlink，导致ONNX文件缺失。失败日志 [failed_missing_model_link.log](failed_missing_model_link.log) 已保留；将该次只含日志的目录移为`model.failed_missing_link/`，建立指向`checkpoints/deployment_4090/nvidia/model`的symlink后重新构建。没有伪装失败为成功。

四个子网络 engine 构建及 trtexec 运行全部通过，返回码0；[engines_manifest.json](engines_manifest.json) 保存每个ONNX/plan/详细层信息/完整日志的大小与SHA256。体积较大的engine及完整verbose日志留在checkpoint目录，各build_tail.log保留末段验收证据。S2完整mini融合一致性由单独runner验证，不能由此构建通过推断。
