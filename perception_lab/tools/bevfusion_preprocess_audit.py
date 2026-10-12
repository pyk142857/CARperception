"""CPU regenerate pinned S1 preprocessing and compare with measured S2 hashes."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from bevfusion_4090 import packet_input,make_pipeline
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'third_party/bevfusion_4090'))
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.utils import recursive_eval
from mmdet3d.core.bbox import LiDARInstance3DBoxes,Box3DMode
configs.load(str(ROOT/'third_party/bevfusion_4090/configs/nuscenes/det/transfusion/secfpn/camera+lidar/resnet50/convfuser.yaml'),recursive=True);cfg=Config(recursive_eval(configs));pipeline,_=make_pipeline(cfg)
p=json.loads((ROOT/'reports/deployment_4090_20261008/inputs/scene0061_packets.json').read_text())[0];item=packet_input(p);item.update(box_type_3d=LiDARInstance3DBoxes,box_mode_3d=Box3DMode.LIDAR);item=pipeline(item)
def array(k):
 x=item[k];x=x.data if hasattr(x,'data') else x;x=x.numpy() if hasattr(x,'numpy') else x;return np.ascontiguousarray(x,dtype=np.float32)
def sha(x):return hashlib.sha256(x.tobytes()).hexdigest()
actual=dict(sample_token=p['sample_token'],fp32_image_sha256=sha(array('img')[None]),fp32_points_sha256=sha(array('points')),matrices_sha256=[sha(array(k)[None]) for k in ['camera2lidar','camera_intrinsics','lidar2image','img_aug_matrix']]);candidate=json.loads((ROOT/'outputs/bevfusion_tensorrt/scene0061/actual_inputs.json').read_text())[0];matched={k:actual[k]==candidate[k] for k in actual};result=dict(scope='CPU reconstruction using same pinned reference config/pipeline, compared against recorded TRT pre-FP16 input hashes; not a retroactive hash of uncaptured S1 input tensors',reference_regenerated=actual,candidate_recorded=candidate,equal=matched,passed=all(matched.values()));(ROOT/'reports/deployment_4090_20261008/bevfusion_tensorrt/preprocess_audit.json').write_text(json.dumps(result,indent=2));print(result)
