# TensorRT 10.6 FP16 复现

工作目录：`perception_lab/`。单 GPU 串行运行，先完成相同权重的原生 COCO 5,000 图评价和作者 ONNX `--check` 导出。

```bash
envs/deimv2_compat_2025/bin/python -m venv --system-site-packages envs/deimv2_trt_2025
envs/deimv2_trt_2025/bin/python - <<'PYENV'
import pathlib, sysconfig
compat = pathlib.Path.cwd() / 'envs/deimv2_compat_2025/lib/python3.11/site-packages'
(pathlib.Path(sysconfig.get_paths()['purelib']) / 'deimv2_compat_parent.pth').write_text('import site; site.addsitedir(' + repr(str(compat)) + ')\n')
PYENV
envs/deimv2_trt_2025/bin/python -m pip install --no-deps --no-build-isolation checkpoints/perception_2025/tensorrt_packages/*.whl checkpoints/perception_2025/tensorrt_packages/*.tar.gz
envs/deimv2_trt_2025/bin/python tools/deimv2_trt_2025.py --phase build > reports/perception_2025_20261013/deimv2/tensorrt/build.log 2>&1
envs/deimv2_trt_2025/bin/python tools/deimv2_trt_2025.py --phase validate > reports/perception_2025_20261013/deimv2/tensorrt/validate.log 2>&1
envs/deimv2_trt_2025/bin/python tools/deimv2_trt_2025.py --phase benchmark > reports/perception_2025_20261013/deimv2/tensorrt/benchmark.log 2>&1
envs/deimv2_trt_2025/bin/python tools/deimv2_trt_2025.py --phase audit > reports/perception_2025_20261013/deimv2/tensorrt/audit.log 2>&1
```

环境版本见 `environment.json`。本机使用 Torch 2.6 / CUDA12.4 / Torchvision0.21 的既有兼容环境，并非作者 Torch2.5.1 环境。旧 TensorRT8.6 环境没有安装或覆盖。依赖文件来源、字节数和 SHA256 见 `packages.json`；两个 wheel 通过完整 ZIP CRC 检查；三个下载文件哈希与下载时保存的 index 相符。

Engine 保存在被忽略的 `checkpoints/perception_2025/deimv2/deimv2_s.fp16.engine`，其哈希/构建时间/参数见 `build.json`。FP16 为 TensorRT builder 精度许可，输入输出仍是 FP32 / INT64；并非声称每层均 FP16。TF32 禁用；静态优化 profile 为 batch1、640×640，workspace 上限4GiB。builder 原样保存 LayerNorm FP16 警告，不进行未经验证的数学修改。

输入预处理沿原生 Resize → ToTensor → ImageNet Normalize。Pillow版本与原生相同；`preprocess_identity.json` 为前3张输入哈希，`preprocess_first_middle_last.json` 为首/中/末输入跨环境位级相等检查。类别沿 COCO sparse category_id；完整300候选均参与评估，不用可视化阈值裁掉预测。

几何报告先对首/中/末图、随后对全部评价图像按类别和框距离进行二分匹配，保留全部候选和score≥0.25两组诊断；不假设TopK顺序恒定。每张预测检查非有限值、框宽高、类别范围及score∈[0,1]；全量诊断保存在all_image_geometry.json.gz和geometry_summary.json。FP16数值差异与候选增减应结合完整COCO AP审阅，不能仅靠行序差或AP声称几何严格等价。

性能测量独立于全量验证。30轮交错顺序配对，首/中/末3张图轮换，每条路径每次从原始图像文件读取、解码、预处理，经H2D、GPU推理及后处理，至D2H的可消费numpy数组。首/中/末每张图先在两条路径各预热10次。两条路径均使用同一环境、Torch CPU线程4、batch1、热OS缓存；JSON转换/写盘、可视化排除。GPUforward采用CUDA event，端到端采用wall clock；两种模型同时常驻以避免冷加载。Torch峰值不覆盖TensorRT分配；另存推理结束时采样的设备总已用显存，包含其他驻留进程，不是进程独占峰值。

配对开始前强制核验checkpoint、配置、155个固定源码文件及导出脚本。已有构建的身份关系属于事后交叉核验（见benchmark_identity.json）；未来新build.json直接保存输入身份，后续validate/benchmark必须匹配。D2H采用紧邻CPU拷贝前后的wall clock直接计时。
