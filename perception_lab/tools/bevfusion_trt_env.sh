#!/usr/bin/env bash
# Source this override instead of upstream environment.sh placeholders.
BEV_TRT_LAB=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export TensorRT_Lib="$BEV_TRT_LAB/envs/tensorrt/lib/python3.10/site-packages/tensorrt_libs"
export TensorRT_Inc="$BEV_TRT_LAB/third_party/TensorRT_8_6/include"
export TensorRT_Bin="$BEV_TRT_LAB/envs/tensorrt/bin"
export CUDA_HOME=/usr/local/cuda-11.8
export CUDA_Lib="$CUDA_HOME/lib64" CUDA_Inc="$CUDA_HOME/include" CUDA_Bin="$CUDA_HOME/bin"
export CUDNN_Lib=/home/minglei/anaconda3/envs/autolabe_cp/lib/python3.10/site-packages/torch/lib
export SPCONV_CUDA_VERSION=11.4 CUDASM=89 USE_Python=ON
export DEBUG_MODEL=resnet50 DEBUG_PRECISION=fp16 DEBUG_DATA=example-data
export Python_Inc=$("$TensorRT_Bin/python" -c "import sysconfig;print(sysconfig.get_path('include'))")
export Python_Lib=$("$TensorRT_Bin/python" -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR'))")
export Python_Soname=libpython3.10.so
export BEV_TRT_BUILD="$BEV_TRT_LAB/third_party/Lidar_AI_Solution/CUDA-BEVFusion/build"
export BEV_TRT_PROTOBUF="$BEV_TRT_LAB/envs/tensorrt/build-deps/extracted/usr"
export BEV_TRT_SPCONV="$BEV_TRT_LAB/third_party/Lidar_AI_Solution/libraries/3DSparseConvolution/libspconv/lib/x86_64_cuda11.4"
export PATH="$TensorRT_Bin:$CUDA_Bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:$BEV_TRT_BUILD:$BEV_TRT_SPCONV:$BEV_TRT_PROTOBUF/lib/x86_64-linux-gnu:$Python_Lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$BEV_TRT_BUILD:${PYTHONPATH:-}"
export ConfigurationStatus=Success
