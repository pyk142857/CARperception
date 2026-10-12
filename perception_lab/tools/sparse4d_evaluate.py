"""Unmodified official devkit DetectionEval and TrackingEval on measured predictions."""
import argparse
from pathlib import Path
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.tracking.evaluate import TrackingEval

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--split',choices=['mini_scene_0061','mini_val'],required=True)
p.add_argument('--dataroot',type=Path,default=Path(__file__).resolve().parents[1]/'data/nuscenes')
a=p.parse_args()
nusc=NuScenes(version='v1.0-mini',dataroot=str(a.dataroot),verbose=False)
DetectionEval(nusc,config_factory('detection_cvpr_2019'),str(a.out/'detection.json'),a.split,str(a.out/'detection_eval'),verbose=True).main(plot_examples=0,render_curves=False)
TrackingEval(config_factory('tracking_nips_2019'),str(a.out/'tracking.json'),a.split,str(a.out/'tracking_eval'),'v1.0-mini',str(a.dataroot),verbose=True).main(render_curves=False)
