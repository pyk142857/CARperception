"""Evaluate native, already-global MambaFusion predictions without reprojecting boxes."""
import hashlib
import json
from pathlib import Path
import numpy as np

NAMES = {'car','truck','bus','trailer','construction_vehicle','pedestrian','motorcycle','bicycle','traffic_cone','barrier'}
ATTRIBUTES = {'','cycle.with_rider','cycle.without_rider','pedestrian.moving','pedestrian.standing','pedestrian.sitting_lying_down','vehicle.moving','vehicle.parked','vehicle.stopped'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_global(data, tokens):
    results = data.get('results', {})
    if len(tokens) != len(set(tokens)) or set(results) != set(tokens):
        raise ValueError('Exact unique token coverage required')
    count = 0
    for token, boxes in results.items():
        if len(boxes) > 500:
            raise ValueError('Native predictions exceed official max500')
        for box in boxes:
            if box.get('sample_token') != token or box.get('detection_name') not in NAMES:
                raise ValueError('Invalid sample/class')
            arrays = {}
            for key, size in [('translation',3),('size',3),('rotation',4),('velocity',2)]:
                value = np.asarray(box.get(key), dtype=float)
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError('Invalid global '+key)
                arrays[key] = value
            if np.any(arrays['size'] <= 0) or not np.isclose(np.linalg.norm(arrays['rotation']),1,atol=1e-5):
                raise ValueError('Invalid wlh/quaternion')
            score = float(box['detection_score'])
            if not np.isfinite(score) or not 0 <= score <= 1 or box.get('attribute_name') not in ATTRIBUTES:
                raise ValueError('Invalid score/author attribute')
            count += 1
    return {'frames':len(tokens),'boxes':count,'coordinate_system':'nuScenes global',
            'geometry_transform':'none; consume author exported global boxes unchanged',
            'score_filter':None,'attributes':'author original, unchanged'}


def evaluate(lab, report):
    import sys
    def save(path, value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    status_path = report/'status.json'
    status = json.loads(status_path.read_text())
    status.update(official_evaluation='running',official_metrics=None)
    save(status_path,status)
    try:
        if status.get('inference') != 'passed' or status.get('frames_inferred') != 120 or status.get('strict_load') != 'passed':
            raise ValueError('Current successful strict native 39+81 inference required')
        receipt = json.loads((report/'native_receipt.json').read_text())
        for split in ['scene0061','mini_val']:
            if sha(report/(split+'_predictions.json')) != receipt['predictions'][split]['sha256']:
                raise ValueError('Predictions differ from current inference receipt')
        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.eval.detection.config import config_factory
        from importlib.metadata import version
        data = lab/'data/nuscenes'
        nusc = NuScenes(version='v1.0-mini',dataroot=str(data),verbose=False)
        split_file = data/'v1.0-mini/splits.json'
        splits = json.loads(split_file.read_text()) if split_file.exists() else {}
        custom = 'mini_scene_0061'
        if custom in splits and splits[custom] != ['scene-0061']:
            raise ValueError('Conflicting custom scene split')
        splits[custom] = ['scene-0061']
        save(split_file,splits)
        summaries = {}
        for split, eval_set, scenes, count in [('scene0061',custom,{'scene-0061'},39),('mini_val','mini_val',{'scene-0103','scene-0916'},81)]:
            tokens = [s['token'] for s in sorted(nusc.sample,key=lambda s:(s['scene_token'],s['timestamp'])) if nusc.get('scene',s['scene_token'])['name'] in scenes]
            if len(tokens) != count: raise ValueError('Unexpected split size')
            predictions = report/(split+'_predictions.json')
            audit = validate_global(json.loads(predictions.read_text()),tokens)
            folder = report/split/'official_eval'
            cfg = config_factory('detection_cvpr_2019')
            audit.update(prediction_sha256=sha(predictions),native_receipt_sha256=sha(report/'native_receipt.json'),
                sample_tokens=tokens,eval_set=eval_set,version='v1.0-mini',devkit_version=version('nuscenes-devkit'),
                config=cfg.serialize(),command=sys.argv,gt_hashes={p.name:sha(p) for p in [data/'v1.0-mini/sample_annotation.json',data/'v1.0-mini/sample.json',data/'v1.0-mini/scene.json']},
                scope='mini teaching evaluation using official DetectionEval; not full validation')
            save(folder/'protocol.json',audit)
            DetectionEval(nusc,cfg,str(predictions),eval_set,output_dir=str(folder),verbose=True).main(plot_examples=0,render_curves=False)
            metrics_path = folder/'metrics_summary.json'
            metrics = json.loads(metrics_path.read_text())
            summaries[split] = {'frames':count,'prediction_sha256':sha(predictions),'metrics_sha256':sha(metrics_path),
                'mean_ap':metrics['mean_ap'],'nd_score':metrics['nd_score'],'tp_errors':metrics['tp_errors']}
        status.update(status='partial',official_evaluation='passed',official_metrics=summaries,
                      blockers=['Synchronized replay acceptance pending'],adoption='native mini evaluation complete; repeat nondeterminism retained; no optimization adoption')
        save(status_path,status)
        return status
    except Exception as exc:
        status.update(status='failed',official_evaluation='failed',official_metrics=None,blockers=[str(exc)])
        save(status_path,status)
        raise
