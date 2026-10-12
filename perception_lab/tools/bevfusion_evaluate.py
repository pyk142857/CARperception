"""Official nuScenes DetectionEval for measured camera-LiDAR predictions."""
import argparse
import csv
import json
from pathlib import Path
from export_nuscenes_detection import export_records
from evidence import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--dataroot', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--eval-set', required=True)
    p.add_argument('--scene', action='append')
    a = p.parse_args()
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import get_scenes_of_split, is_predefined_split
    from nuscenes.eval.common.loaders import get_samples_of_custom_split
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.detection.config import config_factory
    nusc = NuScenes(version='v1.0-mini', dataroot=str(a.dataroot), verbose=False)
    if a.scene:
        if is_predefined_split(a.eval_set):
            raise ValueError('Cannot redefine official split')
        path = a.dataroot / 'v1.0-mini' / 'splits.json'
        splits = json.loads(path.read_text()) if path.exists() else {}
        if a.eval_set in splits and splits[a.eval_set] != a.scene:
            raise ValueError('Conflicting existing split')
        splits[a.eval_set] = a.scene
        path.write_text(json.dumps(splits, indent=2) + '\n')
    if is_predefined_split(a.eval_set):
        scenes = set(get_scenes_of_split(a.eval_set, nusc))
        tokens = [s['token'] for s in nusc.sample if nusc.get('scene', s['scene_token'])['name'] in scenes]
    else:
        tokens = get_samples_of_custom_split(a.eval_set, nusc)
    poses = {}
    for token in tokens:
        sd = nusc.get('sample_data', nusc.get('sample', token)['data']['LIDAR_TOP'])
        poses[token] = nusc.get('ego_pose', sd['ego_pose_token'])
    result, audit = export_records(json.loads(a.predictions.read_text()), poses, tokens)
    result['meta']['use_camera'] = True
    a.out.mkdir(parents=True, exist_ok=True)
    output = a.out / 'official_predictions.json'
    output.write_text(json.dumps(result, allow_nan=False))
    audit.update(input_sha256=sha256(a.predictions), output_sha256=sha256(output),
                 evaluation='DetectionEval/detection_cvpr_2019', eval_set=a.eval_set,
                 sample_tokens=tokens, sensor_meta=result['meta'], scope='mini teaching diagnostic, not full validation')
    (a.out / 'export_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    evaluator = DetectionEval(nusc, config_factory('detection_cvpr_2019'), str(output),
                             a.eval_set, output_dir=str(a.out / 'metrics'), verbose=True)
    evaluator.main(plot_examples=0, render_curves=False)
    metrics = json.loads((a.out / 'metrics/metrics_summary.json').read_text())
    rows = [dict(class_name=name, **aps, mean_ap=metrics['mean_dist_aps'][name]) for name, aps in metrics['label_aps'].items()]
    with (a.out / 'class_ap.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(dict(frames=len(tokens), mAP=metrics['mean_ap'], NDS=metrics['nd_score'])))


if __name__ == '__main__':
    main()
