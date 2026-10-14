set -e
source perception_lab/tools/bevfusion_trt_env.sh
CAR_PREINDEX_SRC="$PWD/perception_lab/runs/tracking_gpu_20261013/bev_pooling/isolated/CUDA-BEVFusion"
CAR_PREINDEX_BUILD="$CAR_PREINDEX_SRC/build"
if [ ! -e "$CAR_PREINDEX_BUILD/libcustom_layernorm.so" ]; then ln -s "$BEV_TRT_BUILD/libcustom_layernorm.so" "$CAR_PREINDEX_BUILD/libcustom_layernorm.so"; fi
export LD_LIBRARY_PATH="$CAR_PREINDEX_BUILD:$LD_LIBRARY_PATH"
export PYTHONPATH="$CAR_PREINDEX_BUILD:${PYTHONPATH:-}"
exec perception_lab/envs/bevfusion/bin/python "$CAR_PREINDEX_SRC/tooling/bevfusion_preindex_run.py" "$@"
