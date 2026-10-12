# 本机 TensorRT 部署依赖与复现

以下目录全部在项目内，未修改系统或既有模型环境。实际版本/哈希见 build_provenance.json、engines_manifest.json、weight_audit.json。

1. TensorRT 8.6.1 Python 3.10 环境：

```bash
perception_lab/envs/mmdet3d/bin/python -m venv perception_lab/envs/tensorrt
perception_lab/envs/tensorrt/bin/pip install tensorrt==8.6.1 onnx==1.16.2
```

pip可能下载CUDA12关联包；本次native运行通过 `tools/bevfusion_trt_env.sh` 显式使用 CUDA11.8、cuDNN8 和TensorRT库路径，不向其他环境添加这些pip包。运行数据适配器使用 `envs/bevfusion/bin/python`，其源码/依赖按 `tools/bevfusion_setup.sh` 恢复。

2. 官方源码：

```bash
git clone https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution.git perception_lab/third_party/Lidar_AI_Solution
git -C perception_lab/third_party/Lidar_AI_Solution checkout 7c1623f234d5a9794fd862c9456f68f87ae7b900
git -C perception_lab/third_party/Lidar_AI_Solution submodule update --init dependencies/pybind11 dependencies/stb dependencies/dlpack
git clone --branch release/8.6 https://github.com/NVIDIA/TensorRT.git perception_lab/third_party/TensorRT_8_6
git -C perception_lab/third_party/TensorRT_8_6 submodule update --init parsers/onnx
```

TensorRT实际commit以build_provenance记录为准，可执行`git checkout`到该commit以消除分支后续变化。SpConv预编译共享库在 `Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda11.4/libspconv.so`；须从同commit的Git LFS获取真实blob（本次13,593,896 bytes），不能把133字节LFS pointer当成库。可从 `https://media.githubusercontent.com/media/NVIDIA-AI-IOT/Lidar_AI_Solution/7c1623f234d5a9794fd862c9456f68f87ae7b900/libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda11.4/libspconv.so` 下载到该路径。

3. NVIDIA官方发布模型zip：[NVBox](https://nvidia.box.com/shared/static/vc1ezra9kw7gu7wg3v8cwuiqshwr8b39)。保存为 `perception_lab/checkpoints/deployment_4090/nvidia_model.zip`，解包至 `perception_lab/checkpoints/deployment_4090/nvidia/`，其中 `model/resnet50/` 含PyTorch、ONNX与配置。编译脚本建立CUDA-BEVFusion/model symlink。权重对应经过CPU定量审计，不能混用其他预训练pth。

4. Protobuf：在Ubuntu20.04兼容的apt源使用 `apt download libprotobuf-dev=3.6.1.3-2ubuntu5.2 libprotoc17=3.6.1.3-2ubuntu5.2 protobuf-compiler=3.6.1.3-2ubuntu5.2`，把.deb放入 `perception_lab/envs/tensorrt/build-deps/`。脚本只使用 `dpkg-deb -x` 解包，不需要root，不安装到系统。本机已有libprotobuf.so.17，必须与protoc3.6.1匹配。

5. 构建代码和engine：

```bash
# CPU编译：幂等应用pooling候选、geometry最后区间终点修复、max_points容量和set_timer绑定修改。
bash perception_lab/tools/bevfusion_trt_setup.sh
# 显式GPU构建：官方四个engine + 单独LayerNorm插件head。
bash perception_lab/tools/bevfusion_trt_setup.sh --build-engines
```

`--build-engines` 不在GPU忙于其他模型时执行。已存在plan时官方脚本跳过；配置/输入ONNX变化时应先核对保存的哈希并在独立目录重建，不能把旧plan声称为新配置结果。实际运行始终以runner记录的engine与ONNX哈希为准。

本次所有构建命令已实际执行，后续整理后的setup脚本通过shell语法检查；新增`--build-engines`完整组合未再次执行，以免干扰当前GPU测量。LayerNorm插件head构建命令与本次实际命令完全对应。
