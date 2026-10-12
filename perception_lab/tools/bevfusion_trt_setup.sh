#!/usr/bin/env bash
# CPU build by default. --build-engines additionally performs serialized GPU construction.
set -euo pipefail
BUILD_ENGINES=0
for option in "$@"; do
  case "$option" in
    --build-engines) BUILD_ENGINES=1 ;;
    *) echo "Usage: $0 [--build-engines]" >&2; exit 2 ;;
  esac
done
LAB=$(cd "$(dirname "$0")/.." && pwd)
source "$LAB/tools/bevfusion_trt_env.sh"
TRT_SRC="$LAB/third_party/TensorRT_8_6"
BEV_SRC="$LAB/third_party/Lidar_AI_Solution/CUDA-BEVFusion"
DEPS="$LAB/envs/tensorrt/build-deps"
REPORT="$LAB/reports/deployment_4090_20261008/bevfusion_tensorrt"
mkdir -p "$REPORT" "$DEPS/extracted"
if [[ ! -e "$BEV_SRC/model" ]]; then
  ln -s "$LAB/checkpoints/deployment_4090/nvidia/model" "$BEV_SRC/model"
fi
if [[ ! -f "$BEV_SRC/model/resnet50/camera.backbone.onnx" ]]; then
  echo "Missing NVIDIA R50 ONNX assets under $BEV_SRC/model; inspect model link." >&2
  exit 1
fi
for package in "$DEPS"/*.deb; do dpkg-deb -x "$package" "$DEPS/extracted"; done
for name in nvinfer nvinfer_plugin nvonnxparser nvparsers; do
  [[ -e "$TensorRT_Lib/lib$name.so" ]] || ln -s "lib$name.so.8" "$TensorRT_Lib/lib$name.so"
done
c++ -std=c++14 -O2 -fno-rtti -pthread \
  -I"$TRT_SRC/include" -I"$TRT_SRC/parsers/onnx" -I"$TRT_SRC/samples/common" -I"$CUDA_Inc" \
  "$TRT_SRC/samples/trtexec/trtexec.cpp" \
  "$TRT_SRC"/samples/common/{sampleEngines,sampleInference,sampleOptions,sampleReporting,sampleUtils,logger}.cpp \
  -L"$TensorRT_Lib" -L"$CUDA_Lib" -Wl,-rpath,"$TensorRT_Lib" -Wl,-rpath,"$CUDA_Lib" \
  -lnvinfer -lnvinfer_plugin -lnvonnxparser -lnvparsers -lcudart -lcublas -ldl -lrt \
  -o "$TensorRT_Bin/trtexec" > "$REPORT/build_trtexec.log" 2>&1
"$TensorRT_Bin/python" - "$BEV_SRC" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
p=root/'src/python.cpp';s=p.read_text().replace('voxelization.max_points = 300000;', 'voxelization.max_points = 500000;')
if 'void set_timer(bool enabled)' not in s:s=s.replace('  void print() { core_->print(); }','  void set_timer(bool enabled) { core_->set_timer(enabled); }\n  void print() { core_->print(); }')
if '.def("set_timer"' not in s:s=s.replace('.def("print", &BEVFusion::print)', '.def("print", &BEVFusion::print)\n      .def("set_timer", &BEVFusion::set_timer)')
p.write_text(s)
p=root/'src/bevfusion/bevfusion.cpp';s=p.read_text().replace('capacity_points_ = 300000;', 'capacity_points_ = param.lidar_scn.voxelization.max_points;');p.write_text(s)
p=root/'tool/environment.sh';p.write_text('#!/bin/bash\n# Workstation override; configuration lives in the tracked project tools.\nsource "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/tools/bevfusion_trt_env.sh"\n')
PY
(
  cd "$BEV_SRC/src/onnx"
  "$BEV_TRT_PROTOBUF/bin/protoc" onnx-ml.proto --cpp_out=.
  "$BEV_TRT_PROTOBUF/bin/protoc" onnx-operators-ml.proto --cpp_out=.
  mv onnx-ml.pb.cc onnx-ml.pb.cpp
  mv onnx-operators-ml.pb.cc onnx-operators-ml.pb.cpp
)
if git -C "$LAB/third_party/Lidar_AI_Solution" apply --check "$LAB/tools/bev_pooling_candidate.patch"; then
  git -C "$LAB/third_party/Lidar_AI_Solution" apply "$LAB/tools/bev_pooling_candidate.patch"
else
  git -C "$LAB/third_party/Lidar_AI_Solution" apply --reverse --check "$LAB/tools/bev_pooling_candidate.patch"
fi
if git -C "$LAB/third_party/Lidar_AI_Solution" apply --check "$LAB/tools/bevfusion_geometry_bounds.patch"; then
  git -C "$LAB/third_party/Lidar_AI_Solution" apply "$LAB/tools/bevfusion_geometry_bounds.patch"
else
  git -C "$LAB/third_party/Lidar_AI_Solution" apply --reverse --check "$LAB/tools/bevfusion_geometry_bounds.patch"
fi
cmake -S "$BEV_SRC" -B "$BEV_TRT_BUILD" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
  -DProtobuf_INCLUDE_DIR="$BEV_TRT_PROTOBUF/include" \
  -DProtobuf_LIBRARY=/lib/x86_64-linux-gnu/libprotobuf.so.17 \
  -DProtobuf_PROTOC_EXECUTABLE="$BEV_TRT_PROTOBUF/bin/protoc" > "$REPORT/configure_pybev.log" 2>&1
cmake --build "$BEV_TRT_BUILD" --target pybev custom_layernorm -j4 > "$REPORT/build_pybev.log" 2>&1
"$TensorRT_Bin/python" -c 'import libpybev; print(libpybev.__file__)' > "$REPORT/import_pybev.log" 2>&1

if [[ "$BUILD_ENGINES" == 1 ]]; then
  cd "$BEV_SRC"
  # Upstream skips an existing plan; remove only after independently validating
  # an intended config/weight change, never silently reuse different identities.
  bash tool/build_trt_engine.sh > "$REPORT/build_engines.log" 2>&1
  if [[ ! -f model/resnet50/build/head.bbox.layernormplugin.plan ]]; then
    trtexec --onnx=model/resnet50/head.bbox.layernormplugin.onnx --fp16 \
      --inputIOFormats=fp16:chw \
      --outputIOFormats=fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw,fp16:chw \
      --plugins=build/libcustom_layernorm.so \
      --saveEngine=model/resnet50/build/head.bbox.layernormplugin.plan \
      --memPoolSize=workspace:2048 --verbose --dumpLayerInfo --dumpProfile --separateProfileRun \
      --profilingVerbosity=detailed --exportLayerInfo=model/resnet50/build/head.bbox.layernormplugin.json \
      > model/resnet50/build/head.bbox.layernormplugin.log 2>&1
  fi
fi
