# GaussianWorld commands

From project root, `bash perception_lab/tools/gaussianworld_setup.sh` automates the following setup (shell syntax checked; its constituent commands were executed). Create Python 3.10 venv, using read-only existing dependency layers to avoid multi-GB duplicate CUDA wheels:

```bash
/home/minglei/anaconda3/envs/autolabel/bin/python -m venv perception_lab/envs/gaussianworld
# base.pth contains, in order:
# /home/minglei/Desktop/CAR/perception_lab/envs/mmdet3d/lib/python3.10/site-packages
# /home/minglei/anaconda3/envs/autolabel/lib/python3.10/site-packages
# /home/minglei/Desktop/CAR/perception_lab/third_party/mmdetection3d
perception_lab/envs/gaussianworld/bin/pip install mmsegmentation==1.2.2 timm ninja
```

Official source commit `b43629eaecffd5a7cbaac1a55517766e6263e4fc`:

```bash
git clone https://github.com/zuosc19/GaussianWorld.git perception_lab/third_party/GaussianWorld
git -C perception_lab/third_party/GaussianWorld checkout b43629eaecffd5a7cbaac1a55517766e6263e4fc
```

Official streaming weight https://cloud.tsinghua.edu.cn/f/4939dcc50b2a44c1b98d/ (read current `rawPath` from page, whose signed URL expires). Save as `perception_lab/checkpoints/gaussianworld/ckpt_stream.pth`. SHA256 `5477081191e6d7fb8ac880bf5b99230b944a7cea76b0d87175c789a51cb97be3`, size 298029831 B.

In each of `model/encoder/gaussian_encoder/ops` and `model/head/localagg` under official source:

```bash
CUDA_HOME=/usr/local/cuda-11.8 TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 /home/minglei/Desktop/CAR/perception_lab/envs/gaussianworld/bin/python setup.py build_ext --inplace
```

No source modification required for CUDA build. The adapter applies the committed `unlabelled.patch` equivalently from the official two Python files. The full checkpoint strictly loads every model key; image-backbone pretrained initialization is omitted because the complete checkpoint already supplies it.

```bash
perception_lab/envs/gaussianworld/bin/python perception_lab/tools/gaussianworld_4090.py --packets /home/minglei/Desktop/CAR/perception_lab/reports/deployment_4090_20261008/inputs/scene0061_packets.json --out /home/minglei/Desktop/CAR/perception_lab/runs/deployment_4090_20261008/gaussianworld --limit 39
perception_lab/envs/gaussianworld/bin/python perception_lab/tools/gaussianworld_summarize.py --run perception_lab/runs/deployment_4090_20261008/gaussianworld --report perception_lab/reports/deployment_4090_20261008/gaussianworld
```

Run with a separate output directory and `--limit 1 --validate-label-independence` for label-dependency and repeatability audit. Never reuse an old output directory for a different checkpoint/configuration/input list.
