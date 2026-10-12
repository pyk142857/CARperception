"""Publish the final measured native timer block, excluding warmup blocks."""
import csv, hashlib, json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/deployment_4090_20261008/bevfusion_tensorrt'

def main():
    log = OUT / 'native_profile.log'
    blocks = re.findall(r'==================BEVFusion===================(.*?)Total:\s*([\d.]+) ms', log.read_text(), re.S)
    if not blocks:
        raise RuntimeError('No complete native timer block found')
    rows = [(name.strip(), float(ms)) for name, ms in re.findall(r'\[⏰ (.*?)\]:\s*([\d.]+) ms', blocks[-1][0])]
    if len(rows) != 8:
        raise RuntimeError(f'Expected eight timed stages, got {len(rows)}')
    total = sum(v for _, v in rows)
    with (OUT / 'native_profile.csv').open('w') as f:
        w = csv.writer(f); w.writerow(['stage', 'ms', 'share_percent'])
        w.writerows((n, v, 100 * v / total) for n, v in rows)
    text = '# TensorRT 逐阶段计时诊断\n\nscene-0061首帧：读取最新日志最后一个完整计时块，前3块为预热。逐段CUDA event同步会扰动流水线；本表用于定位耗时，不替代39/81帧正常模式计时。输入图像已在CPU预处理，Normalization跳过。\n\n|阶段|毫秒|本表占比|\n|---|---:|---:|\n'
    text += ''.join(f'|{n}|{v:.5f}|{100*v/total:.2f}%|\n' for n,v in rows)
    text += f'\n含CopyLidar合计 **{total:.5f} ms**；上游Total **{float(blocks[-1][1]):.3f} ms** 排除标记NoSt的CopyLidar。图像传输发生在Python绑定进入core之前，不在本表；框回传及解码计入Head BoundingBox。CPU数据读取/预处理、几何更新和标准化见各split timings.csv，本表不代表完整端到端耗时。\n\n相机backbone、检测head和LiDAR backbone是主要项；单独优化pooling不能按其kernel加速比等比例加速完整感知流程。\n\n'
    text += f'日志SHA256 `{hashlib.sha256(log.read_bytes()).hexdigest()}`；共{len(blocks)}个完整块，使用第{len(blocks)}块。\n\n复现：\n```bash\ncd perception_lab\nsource tools/bevfusion_trt_env.sh\nenvs/bevfusion/bin/python tools/bevfusion_trt_4090.py --packets reports/deployment_4090_20261008/inputs/scene0061_packets.json --out runs/deployment_4090_20261008/trt_profile --limit 1 --warmup 3 --profile > reports/deployment_4090_20261008/bevfusion_tensorrt/native_profile.log 2>&1\npython tools/bevfusion_native_profile_report.py\n```\n'
    (OUT / 'native_profile.md').write_text(text)

if __name__ == '__main__':
    main()
