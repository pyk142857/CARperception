"""Run MMSegmentation's Cityscapes SegFormer B0 with its native slide pipeline."""
import argparse
import json
from pathlib import Path
import os
import numpy as np
import torch
from evidence import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(2)
    from mmseg.apis import init_model, inference_model
    from mmseg.utils import get_classes, get_palette
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import cv2
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = root / 'third_party/mmsegmentation/configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py'
    weight = root / 'checkpoints/segformer_b0_cityscapes.pth'
    model = init_model(str(cfg), str(weight), device=args.device)
    state = torch.load(weight, map_location='cpu')['state_dict']
    print(model.load_state_dict(state, strict=True), flush=True)
    model.cfg.dump(str(out / 'resolved_config.py'))
    manifest = json.loads(Path(args.manifest).read_text())
    names = get_classes('cityscapes'); palette = np.array(get_palette('cityscapes'), dtype=np.uint8)
    records = []
    for index, frame in enumerate(manifest['images']):
        with torch.inference_mode():
            result = inference_model(model, frame['path'])
        seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
        image = cv2.cvtColor(cv2.imread(frame['path']), cv2.COLOR_BGR2RGB)
        if seg.shape != image.shape[:2] or seg.max() >= 19:
            raise ValueError('Invalid segmentation shape or class IDs')
        target = out / f'{index:06d}_trainid.npy'; np.save(target, seg)
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.imshow((image * .5 + palette[seg] * .5).astype(np.uint8)); ax.axis('off')
        ax.legend(handles=[Patch(color=palette[i]/255, label=n) for i,n in enumerate(names)],
                  loc='upper center', bbox_to_anchor=(.5, -.01), ncol=4, fontsize=8)
        fig.savefig(out / f'{index:06d}_overlay.png', bbox_inches='tight'); plt.close(fig)
        records.append(dict(run_id=out.parts[-3], module_id='M02', model_id='SegFormer_B0_Cityscapes',
            sample_token=frame.get('sample_token'), scene_token=frame.get('scene_token'),
            timestamp_us=frame.get('timestamp_us'), source='measured', scope=manifest['scope'], status='passed',
            native_coordinate_system='original_image_pixels', label_namespace='Cityscapes_trainId_19',
            checkpoint_sha256=sha256(weight), source_config_hash=sha256(out/'resolved_config.py'),
            input_path=frame['path'], input_sha256=sha256(frame['path']), camera_channel=frame.get('camera_channel'),
            segmentation={'path':str(target),'shape':list(seg.shape),'ignore_id':255}))
        print(index, frame['path'], 'passed', flush=True)
    (out/'labels.json').write_text(json.dumps({'classes':names,'palette':palette.tolist()},indent=2))
    (out/'predictions.json').write_text(json.dumps(records,indent=2))
    (out/'summary.json').write_text(json.dumps({'source':'measured','scope':manifest['scope'],
        'sample_count':len(records),'mIoU':None,'reason':'No pixel GT in this diagnostic input','device':args.device},indent=2))


if __name__ == '__main__':
    main()
