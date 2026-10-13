#!/usr/bin/env python3
"""MambaFusion deployment resource checks. Never substitute an older detector."""
from __future__ import annotations
import argparse
import hashlib
import json
import copy
import subprocess
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
PIN = "d85c2feeb19207b20e42d50421afb11c095f3efa"
SOURCE = LAB / "third_party/MambaFusion2025"
REPORT = LAB / "reports/perception_2025_20261013/mambafusion"
WEIGHT = LAB / "checkpoints/mambafusion_2025/checkpoint_epoch_10.pth"
CHECKPOINT_BYTES = 391545251


CAMERA_ORDER = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


def resolved_config(split="mini_val"):
    import yaml

    source = SOURCE / "tools/cfgs/mambafusion_models/mamba_fusion.yaml"
    cfg = yaml.safe_load(source.read_text())
    base = yaml.safe_load(
        (SOURCE / "tools" / cfg["DATA_CONFIG"].pop("_BASE_CONFIG_")).read_text()
    )

    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)
        return dst

    cfg["DATA_CONFIG"] = merge(base, cfg["DATA_CONFIG"])
    cfg["DATA_CONFIG"]["VERSION"] = "v1.0-mini"
    cfg["DATA_CONFIG"]["INFO_PATH"]["test"] = [
        str(
            LAB
            / "runs/perception_2025/mambafusion"
            / ("nuscenes_infos_10sweeps_" + split + "_withimg.pkl")
        )
    ]
    cfg["DATA_CONFIG"]["DATA_PATH"] = str(LAB / "data/nuscenes")
    validate_author_config(cfg)
    return cfg


def packet_to_info(packet):
    """Build inference fields only; reproduce author's nine-entry history padding.

    Uses raw paths and time-bound calibration, never old model tensors or GT.
    Full author Dataset/extension integration remains a separate runtime gate.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation

    ref = packet["sensors"]["LIDAR_TOP"]
    l2e = np.asarray(ref["T_sensor_to_ego"])
    e2g = np.asarray(ref["T_ego_to_global"])
    global_to_ref = np.linalg.inv(e2g @ l2e)
    info = {
        "lidar_path": ref["path"],
        "token": packet["sample_token"],
        "timestamp": ref["timestamp_us"] * 1e-6,
        "ref_from_car": np.linalg.inv(l2e),
        "car_from_global": np.linalg.inv(e2g),
        "cams": {},
        "sweeps": [],
    }
    for name in CAMERA_ORDER:
        cam = packet["sensors"][name]
        c2e = np.asarray(cam["T_sensor_to_ego"])
        ce2g = np.asarray(cam["T_ego_to_global"])
        c2l = global_to_ref @ ce2g @ c2e

        def quat(matrix):
            q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
            return [float(q[3]), *q[:3].tolist()]

        info["cams"][name] = {
            "data_path": cam["path"],
            "type": name,
            "sample_data_token": cam["sample_data_token"],
            "timestamp": cam["timestamp_us"],
            "camera_intrinsics": np.asarray(cam["K"]),
            "sensor2ego_translation": c2e[:3, 3],
            "sensor2ego_rotation": quat(c2e),
            "ego2global_translation": ce2g[:3, 3],
            "ego2global_rotation": quat(ce2g),
            "sensor2lidar_rotation": c2l[:3, :3],
            "sensor2lidar_translation": c2l[:3, 3],
        }
    for sweep in packet["lidar_sweeps"][:9]:
        dt = (ref["timestamp_us"] - sweep["timestamp_us"]) * 1e-6
        if dt < 0:
            raise ValueError("future sweep detected")
        transform = (
            global_to_ref
            @ np.asarray(sweep["T_ego_to_global"])
            @ np.asarray(sweep["T_sensor_to_ego"])
        )
        info["sweeps"].append(
            {
                "lidar_path": sweep["path"],
                "sample_data_token": sweep["sample_data_token"],
                "transform_matrix": transform,
                "time_lag": dt,
            }
        )
    if not info["sweeps"]:
        info["sweeps"].append(
            {
                "lidar_path": ref["path"],
                "sample_data_token": ref["sample_data_token"],
                "transform_matrix": None,
                "time_lag": 0.0,
            }
        )
    while len(info["sweeps"]) < 9:
        info["sweeps"].append(dict(info["sweeps"][-1]))
    return info


def prepare_infos():
    import pickle
    from collections import Counter

    out = LAB / "runs/perception_2025/mambafusion"
    out.mkdir(parents=True, exist_ok=True)
    inventory = {}
    for split, count in [("scene0061", 39), ("mini_val", 81)]:
        path = REPORT.parent / "inputs" / (split + "_packets.json")
        packets = json.loads(path.read_text())
        if len(packets) != count:
            raise ValueError("mini packet count mismatch")
        infos = [packet_to_info(p) for p in packets]
        tokens = [i["token"] for i in infos]
        if len(tokens) != len(set(tokens)):
            raise ValueError("duplicate sample token")
        dst = out / ("nuscenes_infos_10sweeps_" + split + "_withimg.pkl")
        with dst.open("wb") as f:
            pickle.dump(infos, f)
        inventory[split] = {
            "frames": len(infos),
            "path": str(dst),
            "sha256": sha256(dst),
            "input_packet_sha256": sha256(path),
            "camera_order": CAMERA_ORDER,
            "actual_history_histogram": dict(
                Counter(min(len(p["lidar_sweeps"]), 9) for p in packets)
            ),
            "padded_history_entries": 9,
            "contains_annotations": False,
            "native_dataset_integration_validated": False,
            "sweeps": check_sweep_inventory(infos),
        }
    dump("mini_infos.json", inventory)
    for split in inventory:
        dump("resolved_" + split + ".json", resolved_config(split))
    return inventory


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_checkpoint_file(path, expected_bytes=None):
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(512).lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")) or b"<html" in head:
        raise ValueError("HTML response is not a checkpoint")
    size = path.stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"incomplete checkpoint: {size} != {expected_bytes} bytes")
    if size < 100_000_000:
        raise ValueError(f"incomplete full detector checkpoint: only {size} bytes")
    return {"path": str(path), "bytes": size, "sha256": sha256(path)}


def validate_author_config(cfg):
    data = cfg["DATA_CONFIG"]
    used = data.get("POINT_FEATURE_ENCODING", {}).get("used_feature_list", [])
    if used != ["x", "y", "z", "intensity", "timestamp"]:
        raise ValueError(
            "author features must be XYZ/intensity/timestamp; ring is not timestamp"
        )
    if data["MAX_SWEEPS"] != 10:
        raise ValueError("author requires 10 sweeps")
    if data["POINT_CLOUD_RANGE"] != [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]:
        raise ValueError("point cloud range differs from author config")
    if data["CAMERA_CONFIG"]["IMAGE"]["FINAL_DIM"] != [256, 704]:
        raise ValueError("author six camera resolution must be 256x704")
    if (
        cfg["MODEL"]["NAME"] != "MambaFusion"
        or cfg["MODEL"]["DENSE_HEAD"]["NUM_PROPOSALS"] != 200
    ):
        raise ValueError("model/proposal configuration differs from author")
    flags = {
        "USE_CAMERA": data["CAMERA_CONFIG"].get("USE_CAMERA"),
        "USE_VOXEL_MAMBA": cfg["MODEL"].get("USE_VOXEL_MAMBA"),
        "USE_DOW5": cfg["MODEL"].get("BACKBONE_3D", {}).get("USE_DOW5"),
        "USE_MAMBA": cfg["MODEL"].get("VTRANSFORM", {}).get("USE_MAMBA"),
        **{
            name: cfg["MODEL"].get("MM_BACKBONE", {}).get(name)
            for name in [
                "USE_WINMAMBA",
                "USE_ALL_MAMBA",
                "USE_MAMBA_INTER2",
                "USE_VMAMBA_PRETRAIN",
            ]
        },
    }
    for name, value in flags.items():
        if value is not True:
            raise ValueError("author enabled flag changed: " + name)
    voxel = [
        x["VOXEL_SIZE"]
        for x in data["DATA_PROCESSOR"]
        if x["NAME"] == "transform_points_to_voxels_placeholder"
    ]
    if voxel != [[0.3, 0.3, 0.25]]:
        raise ValueError("author actual voxel parameters differ")


def check_sweep_inventory(infos):
    result = []
    for info in infos:
        lags = [float(s["time_lag"]) for s in info["sweeps"]]
        if any(t < 0 for t in lags):
            raise ValueError("future sweep detected")
        if len(lags) > 9:
            raise ValueError("more than nine historical sweeps")
        result.append(
            {
                "sample_token": info["token"],
                "available_history": len(lags),
                "time_lags_seconds": lags,
            }
        )
    return result


def dump(name, data):
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def preflight():
    blockers = []
    resources = {
        "model": "MambaFusion",
        "source_url": "https://github.com/AutoLab-SAI-SJTU/MambaFusion",
        "source_commit": PIN,
        "checkpoint_url": "https://drive.google.com/file/d/10-FBWHbsCMJseKovJwGKncJ96Flp3umM/view",
        "checkpoint": None,
    }
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        resources["actual_commit"] = actual
        if actual != PIN:
            blockers.append("source commit differs from pinned author commit")
    except (subprocess.CalledProcessError, FileNotFoundError):
        blockers.append("complete pinned author source checkout unavailable")
    config = SOURCE / "tools/cfgs/mambafusion_models/mamba_fusion.yaml"
    if config.exists():
        resources["author_config_sha256"] = sha256(config)
    license_path = SOURCE / "LICENSE"
    if license_path.exists():
        resources["license"] = {
            "path": str(license_path),
            "sha256": sha256(license_path),
        }
    try:
        resources["checkpoint"] = validate_checkpoint_file(WEIGHT, CHECKPOINT_BYTES)
    except (OSError, ValueError) as e:
        blockers.append("Complete author checkpoint unavailable: " + str(e))
        evidence = REPORT / "download_attempts.json"
        if evidence.exists():
            blockers.append(
                "Drive streaming returned HTTP413: 391545251 bytes exceeds 268435456-byte connector limit; shell downloads incomplete."
            )
    resources["hilbert"] = []
    for rank in (7, 8, 9, 10):
        path = WEIGHT.parent / f"curve_template_3d_rank_{rank}.pth"
        if not path.exists():
            blockers.append(f"Hilbert rank {rank} unavailable")
        else:
            resources["hilbert"].append(
                {
                    "rank": rank,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    partial = WEIGHT.parent / "checkpoint.download"
    if partial.exists():
        resources["incomplete_checkpoint"] = {
            "bytes": partial.stat().st_size,
            "sha256": sha256(partial),
            "usable": False,
        }
    dump("resource_manifest.json", resources)
    status = {
        "model": "MambaFusion",
        "status": "blocked_resource" if blockers else "partial",
        "phase": "preflight",
        "source": "new_measured",
        "frames_inferred": 0,
        "strict_load": None,
        "official_metrics": None,
        "blockers": blockers,
        "gpu_inference_executed": False,
        "gpu_operator_smoke": (
            json.loads((REPORT / "operator_smoke.json").read_text())
            if (REPORT / "operator_smoke.json").exists()
            else None
        ),
        "frames_preprocessed": (
            json.loads((REPORT / "cpu_pipeline.json").read_text()).get(
                "frames_preprocessed", 0
            )
            if (REPORT / "cpu_pipeline.json").exists()
            else 0
        ),
        "adoption": "not deployed; native model requires strict load and operator/frame validation",
    }
    dump("status.json", status)
    return status


def native_run(gate_only=False, allow_measured_repeat_noise=False):
    """Run the actual author detector; unreachable until complete resource gates pass.

    GPU scheduling belongs to the common orchestrator. This function does not
    translate or silently drop state keys, or replace unavailable native ops.
    """
    import os, sys, time, logging, random
    import numpy as np
    import torch
    from easydict import EasyDict
    torch.set_num_threads(4)
    torch.manual_seed(20261013)

    sys.path[:0] = [
        str(SOURCE),
        str(SOURCE / "selective_scan"),
        str(SOURCE / "mamba_diffv/mamba"),
    ]
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.datasets.nuscenes import nuscenes_utils
    from nuscenes.nuscenes import NuScenes

    if allow_measured_repeat_noise:
        review=json.loads((REPORT/'repeat_noise_review.json').read_text())
        if review.get('status')!='reviewed_native_nondeterminism' or len(review.get('rows',[]))!=12:
            raise ValueError('Complete measured repeat review required')
        for row in review['rows']:
            if sha256(REPORT/row['artifact'])!=row['sha256']:
                raise ValueError('Repeat diagnostic review hash mismatch')
    prepare_infos()
    template_dir = SOURCE / "ckpts/hilbert_template"
    template_dir.parent.mkdir(parents=True, exist_ok=True)
    if not template_dir.exists():
        template_dir.symlink_to(WEIGHT.parent, target_is_directory=True)
    os.chdir(SOURCE / "tools")
    cfg = EasyDict(resolved_config("scene0061"))
    logger = logging.getLogger("MambaFusion2025")
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)
    dataset = NuScenesDataset(
        cfg.DATA_CONFIG,
        cfg.CLASS_NAMES,
        training=False,
        root_path=LAB / "data/nuscenes",
        logger=logger,
    )
    checkpoint = torch.load(WEIGHT, map_location="cpu")
    if "model_state" not in checkpoint:
        raise ValueError("complete author detector model_state missing")
    model = build_network(
        model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset
    )
    loaded = model.load_state_dict(checkpoint["model_state"], strict=True)
    dump(
        "strict_load.json",
        {
            "missing_keys": list(loaded.missing_keys),
            "unexpected_keys": list(loaded.unexpected_keys),
            "state_tensors": len(checkpoint["model_state"]),
            "checkpoint_sha256": sha256(WEIGHT),
        },
    )
    model = model.cuda().eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hooks = {}
    handles = []
    scan_originals = []
    for module_name in ['selective_scan_cuda_core','selective_scan_cuda_oflex','selective_scan_cuda_ndstate','selective_scan_cuda']:
        import importlib
        extension = importlib.import_module(module_name)
        original = extension.fwd
        def observed_scan(*args, _original=original, _name=module_name, **kwargs):
            row=hooks.setdefault('cuda.'+_name,{'calls':0,'type':'author CUDA selective scan'})
            row['calls']+=1
            row['input_shapes']=[list(x.shape) for x in args if isinstance(x,torch.Tensor)]
            return _original(*args,**kwargs)
        extension.fwd=observed_scan
        scan_originals.append((extension,original))

    def hook_for(name):
        def hook(module, inputs, outputs):
            row = hooks.setdefault(name, {"calls": 0, "type": type(module).__name__})
            row["calls"] += 1
            tensors = {}
            def visit(value,path='output'):
                if isinstance(value,torch.Tensor):tensors[path]=value
                elif isinstance(value,dict):
                    for key,item in value.items():visit(item,path+'.'+str(key))
                elif isinstance(value,(list,tuple)):
                    for index,item in enumerate(value):visit(item,path+'.'+str(index))
                elif isinstance(getattr(value,'features',None),torch.Tensor):
                    tensors[path+'.features']=value.features
            visit(outputs)
            row["outputs"] = {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in tensors.items()
            }
            for value in tensors.values():
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError("nonfinite module output " + name)

        return hook

    for name, module in model.named_modules():
        if name in model.module_topology or "mamba" in type(module).__name__.lower() or type(module).__name__ in ('SS2D','Backbone_VSSM'):
            handles.append(module.register_forward_hook(hook_for(name)))
    nusc = NuScenes(
        version="v1.0-mini", dataroot=str(LAB / "data/nuscenes"), verbose=False
    )
    total = 0
    for split, count in [("scene0061", 39), ("mini_val", 81)]:
        cfg = EasyDict(resolved_config(split))
        dataset = NuScenesDataset(
            cfg.DATA_CONFIG,
            cfg.CLASS_NAMES,
            training=False,
            root_path=LAB / "data/nuscenes",
            logger=logger,
        )
        if len(dataset) != count:
            raise ValueError("native Dataset frame count mismatch")
        predictions = []
        profile = []
        gate = []
        torch.cuda.reset_peak_memory_stats()
        order = list(dict.fromkeys([0, count // 2, count - 1] + list(range(count))))
        for idx in (order[:3] if gate_only else order):
            hook_before={k:v['calls'] for k,v in hooks.items()}
            np.random.seed(2025 + idx)
            random.seed(2025 + idx)
            torch.manual_seed(2025 + idx)
            start = time.perf_counter()
            batch = dataset.collate_batch([dataset[idx]])
            if "gt_boxes" in batch:
                raise ValueError("annotations entered inference batch")
            input_hash={k: hashlib.sha256((v.numpy() if isinstance(v,torch.Tensor) else v).tobytes()).hexdigest() for k,v in batch.items() if isinstance(v,torch.Tensor) or (isinstance(v,np.ndarray) and v.dtype!=object)}
            load_data_to_gpu(batch)
            event1 = torch.cuda.Event(enable_timing=True)
            event2 = torch.cuda.Event(enable_timing=True)
            with torch.inference_mode():
                event1.record()
                pred, _ = model(batch)
                event2.record()
                torch.cuda.synchronize()
            annos = dataset.generate_prediction_dicts(batch, pred, cfg.CLASS_NAMES)
            for anno in annos:
                if (
                    not np.isfinite(anno["boxes_lidar"]).all()
                    or not np.isfinite(anno["score"]).all()
                ):
                    raise ValueError("nonfinite predicted box or score")
            predictions.extend(annos)
            profile.append(
                {
                    "sample_token": dataset.infos[idx]["token"],
                    "end_to_end_ms": (time.perf_counter() - start) * 1000,
                    "gpu_forward_ms": event1.elapsed_time(event2),
                }
            )
            if idx in [0,count//2,count-1]:
                # Rebuild the same seeded native batch: author forward mutates its dictionary.
                repeats=[]
                for repeat in range(2):
                    np.random.seed(2025+idx)
                    random.seed(2025+idx)
                    torch.manual_seed(2025+idx)
                    repeated=dataset.collate_batch([dataset[idx]])
                    repeat_input_hash={k: hashlib.sha256((v.numpy() if isinstance(v,torch.Tensor) else v).tobytes()).hexdigest() for k,v in repeated.items() if isinstance(v,torch.Tensor) or (isinstance(v,np.ndarray) and v.dtype!=object)}
                    if repeat_input_hash!=input_hash:raise ValueError("Native repeated input differs before GPU")
                    load_data_to_gpu(repeated)
                    with torch.inference_mode():
                        other,_=model(repeated)
                    other_ann=dataset.generate_prediction_dicts(repeated,other,cfg.CLASS_NAMES)[0]
                    a,b=annos[0],other_ann
                    np.savez_compressed(REPORT / (split+'_gate_'+str(idx)+'_repeat_'+str(repeat)+'.npz'), base_boxes=a['boxes_lidar'],base_scores=a['score'],base_names=a['name'], repeat_boxes=b['boxes_lidar'],repeat_scores=b['score'],repeat_names=b['name'])
                    aligned=a['boxes_lidar'].shape==b['boxes_lidar'].shape and np.array_equal(a['name'],b['name'])
                    repeats.append({'candidate_order_and_shape_equal':bool(aligned),
                        'box_max_abs':float(np.max(np.abs(a['boxes_lidar']-b['boxes_lidar']))) if aligned and len(a['name']) else None,
                        'score_max_abs':float(np.max(np.abs(a['score']-b['score']))) if aligned and len(a['name']) else None})
                gate.append({'sample_token':dataset.infos[idx]['token'],'index':idx,'finite':True,'repeats':repeats,
                             'input_sha256':input_hash,'repeat_input_identical':True,'note':'same raw frame and seeded native eval preprocessing; candidate order may vary'})
                frame_hooks={name:dict(evidence,calls=evidence['calls']-hook_before.get(name,0)) for name,evidence in hooks.items() if evidence['calls']>hook_before.get(name,0)}
                gate[-1]['module_hooks']=frame_hooks
                gate[-1]['repeat_budget']={'box_component_max_abs':.1,'score_max_abs':.01,'aligned_classes_and_shape_required':True}
                gate[-1]['passed']=all(r['candidate_order_and_shape_equal'] and (r['box_max_abs'] is None or r['box_max_abs']<=.1) and (r['score_max_abs'] is None or r['score_max_abs']<=.01) for r in repeats)
                dump(split+'_first_middle_last.json',gate)
                if any(frame_hooks.get(name,{}).get('calls',0)==0 for name in model.module_topology):
                    raise ValueError('Frame-local enabled module gate failed before expansion')
                if not any(name.startswith('cuda.') for name in frame_hooks):
                    raise ValueError('Frame-local CUDA scan gate failed before expansion')
                if not any(v['type']=='SS2D' for v in frame_hooks.values()):
                    raise ValueError('Frame-local camera scan/fusion gate failed before expansion')
                gate[-1]['native_diagnostic_continuation']=bool(allow_measured_repeat_noise and not gate[-1]['passed'])
                dump(split+'_first_middle_last.json',gate)
                if not gate[-1]['passed'] and not gate_only and not allow_measured_repeat_noise:
                    raise ValueError('Native repeat noise exceeds declared gate; inspect three-frame evidence before expansion')
        if gate_only:
            continue
        required = [name for name in model.module_topology]
        if any(hooks.get(name, {}).get("calls", 0) == 0 for name in required):
            raise ValueError("enabled topology module had zero calls")
        if not any(v['calls']>0 for k,v in hooks.items() if k.startswith('cuda.')):
            raise ValueError('No real selective scan CUDA calls observed')
        dump(split+'_first_middle_last.json',gate)
        by_token = {p['metadata']['token']: p for p in predictions}
        predictions = [by_token[info['token']] for info in dataset.infos]
        results = nuscenes_utils.transform_det_annos_to_nusc_annos(predictions, nusc)
        results["meta"] = {
            "use_camera": True,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        dump(split + "_predictions.json", results)
        dump(
            split + "_profile.json",
            {
                "frames": count,
                "profile": profile,
                "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                "dtype": "float32",
                "tf32": False,
                "warmup": "first/middle/last gate included; not benchmark",
            },
        )
        total += count
    if gate_only:
        result={'status':'partial','phase':'gate','inference':'diagnostic_only','inference_frames':0,'blockers':['Repeat diagnostic only; full expansion not yet accepted']}
        dump('status.json',result)
        return result
    dump("hooks.json", hooks)
    dump("native_receipt.json", {
        "checkpoint_sha256": sha256(WEIGHT),
        "repeat_noise_review_sha256": sha256(REPORT/"repeat_noise_review.json") if allow_measured_repeat_noise else None,
        "source_commit": PIN,
        "script_sha256": sha256(Path(__file__)),
        "predictions": {split: {"sha256":sha256(REPORT/(split+"_predictions.json")), "frames":count}
                        for split,count in [("scene0061",39),("mini_val",81)]},
    })
    for handle in handles:
        handle.remove()
    for extension,original in scan_originals:
        extension.fwd=original
    status = {
        "model": "MambaFusion",
        "status": "partial",
        "phase": "run",
        "source": "new_measured",
        "frames_inferred": total,
        "strict_load": "passed",
        "inference": "passed",
        "official_metrics": None,
        "blockers": ["Official DetectionEval and replay acceptance pending"],
        "gpu_inference_executed": True,
        "adoption": "native predictions pending evaluation; no optimization claim",
    }
    dump("status.json", status)
    return status


def generate_hilbert():
    import importlib.util, torch, pathlib, hashlib, json, time

    root = LAB
    script = (
        root
        / "third_party/MambaFusion2025/tools/hilbert_curves/create_hilbert_curve_template.py"
    )
    spec = importlib.util.spec_from_file_location("author_hilbert", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    out = root / "checkpoints/mambafusion_2025"
    results = {}
    for rank, zmax, dim in [(7, 9, 3), (10, 1, 2), (8, 17, 3), (9, 41, 3)]:
        t = time.monotonic()
        n = 2**rank
        count = n * n * zmax
        values = []
        for start in range(0, count, 262144):
            idx = torch.arange(start, min(start + 262144, count), dtype=torch.int64)
            coords = (
                torch.stack([idx // (n * n), idx // n % n, idx % n], dim=1)
                if dim == 3
                else torch.stack([idx // n, idx % n], dim=1)
            )
            values.append(module.convert_to_index(coords, dim, rank))
        result = torch.cat(values)
        path = out / f"curve_template_3d_rank_{rank}.pth"
        if path.exists():
            old = torch.load(path, map_location="cpu")
            if isinstance(old, dict):
                old = old["data"]
            eq = torch.equal(old.reshape(-1), result)
            print(rank, old.shape, result.shape, "equal", eq, flush=True)
            if not eq:
                raise RuntimeError("author downloadable template differs")
        else:
            torch.save(result, path)
        results[rank] = {
            "rank": rank,
            "dimensions": dim,
            "z_max": zmax,
            "shape": list(result.shape),
            "dtype": str(result.dtype),
            "bytes": path.stat().st_size,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tensor_sha256": hashlib.sha256(result.numpy().tobytes()).hexdigest(),
            "author_download_equal": (
                True
                if hashlib.sha256(path.read_bytes()).hexdigest()
                in {
                    "dcb00b6eafe3ec0414760196475db2d044ea4ea0ecde82ba426bc9c88e0e954f",
                    "2032f33fd571d740785055f80ee49ef14db8c9e2f80e794ec17c599aa3ea7441",
                }
                else None
            ),
            "generation_seconds": time.monotonic() - t,
            "generation_method": "unmodified author convert_to_index, CPU chunks, only coordinates in saved slice",
        }
        print(rank, "complete", results[rank], flush=True)
    (
        root / "reports/perception_2025_20261013/mambafusion/hilbert_manifest.json"
    ).write_text(json.dumps(results, indent=2) + "\n")


def operator_smoke():
    import sys, pathlib, json, torch, importlib, math, time

    root = LAB
    src = root / "third_party/MambaFusion2025"
    sys.path[:0] = [
        str(src),
        str(src / "selective_scan"),
        str(src / "mamba_diffv/mamba"),
    ]
    res = {
        "torch": torch.__version__,
        "runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "operators": {},
    }
    u = torch.ones(1, 4, 16, device="cuda")
    A = -torch.ones(4, 1, device="cuda")
    B = torch.ones(1, 1, 1, 16, device="cuda")
    expected = torch.tensor(
        [(1 - math.exp(-(i + 1))) / (1 - math.exp(-1)) for i in range(16)],
        device="cuda",
    ).view(1, 1, 16)
    for name in [
        "selective_scan_cuda_core",
        "selective_scan_cuda_oflex",
        "selective_scan_cuda_ndstate",
        "selective_scan_cuda",
    ]:
        m = importlib.import_module(name)
        args = [u, u, A, B, B, None, None, False, 1]
        if name == "selective_scan_cuda":
            args = [u, u, A, B, B, None, None, None, False]
        if name.endswith("oflex"):
            args.append(True)
        if name.endswith("ndstate"):
            args = [u, u, A.flatten(), B.squeeze(2), B.squeeze(2), None, None, False, 1]
        out = m.fwd(*args)[0]
        torch.cuda.synchronize()
        err = (out - expected).abs().max().item()
        assert err < 1e-5, (name, err)
        res["operators"][name] = {
            "passed": True,
            "shape": list(out.shape),
            "max_abs_error_to_analytic_recurrence": err,
        }
    iou = importlib.import_module("pcdet.ops.iou3d_nms.iou3d_nms_cuda")
    boxes = torch.tensor(
        [[0, 0, 0, 2, 2, 2, 0], [10, 0, 0, 2, 2, 2, 0]],
        device="cuda",
        dtype=torch.float32,
    )
    out = torch.zeros(2, 2, device="cuda")
    iou.boxes_iou_bev_gpu(boxes, boxes, out)
    torch.cuda.synchronize()
    assert torch.allclose(out, torch.eye(2, device="cuda"))
    res["operators"]["iou3d_nms_cuda"] = {"passed": True, "iou": out.cpu().tolist()}
    roi = importlib.import_module("pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda")
    pts = torch.tensor(
        [[[0, 0, 0], [10, 0, 0], [20, 0, 0]]], device="cuda", dtype=torch.float32
    )
    idx = torch.full((1, 3), -1, device="cuda", dtype=torch.int32)
    roi.points_in_boxes_gpu(boxes.unsqueeze(0), pts, idx)
    torch.cuda.synchronize()
    assert idx.cpu().tolist() == [[0, 1, -1]]
    res["operators"]["roiaware_pool3d_cuda"] = {
        "passed": True,
        "indices": idx.cpu().tolist(),
    }
    window = importlib.import_module("pcdet.ops.win_coors.flattened_window_cuda")
    coords = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 359, 359]], device="cuda", dtype=torch.int64
    )
    out = window.get_window_coors_shift_v2(coords, [360, 360, 1], [30, 30, 1], False)
    torch.cuda.synchronize()
    res["operators"]["flattened_window_cuda"] = {
        "passed": True,
        "outputs": [{"shape": list(x.shape), "dtype": str(x.dtype)} for x in out],
    }
    import torch_scatter
    import spconv.pytorch as spconv
    from spconv.pytorch.utils import PointToVoxel

    scatter = torch_scatter.scatter_mean(
        torch.tensor([1.0, 3.0, 5.0], device="cuda"),
        torch.tensor([0, 0, 1], device="cuda"),
    )
    assert torch.equal(scatter, torch.tensor([2.0, 5.0], device="cuda"))
    res["operators"]["torch_scatter_mean"] = {
        "passed": True,
        "output": scatter.cpu().tolist(),
    }
    sparse = spconv.SparseConvTensor(
        torch.tensor([[1.0], [2.0]], device="cuda"),
        torch.tensor([[0, 1, 1, 1], [0, 1, 1, 2]], device="cuda", dtype=torch.int32),
        [4, 4, 4],
        1,
    )
    conv = spconv.SubMConv3d(1, 1, 3, bias=False).cuda()
    with torch.no_grad():
        conv.weight.fill_(1.0)
        sparse_out = conv(sparse).features
    assert torch.allclose(sparse_out, torch.tensor([[3.0], [3.0]], device="cuda"))
    res["operators"]["spconv_submconv3d"] = {
        "passed": True,
        "output": sparse_out.cpu().tolist(),
        "scope": "unit operator with explicit all-one kernel, not detector inference",
    }
    voxelizer = PointToVoxel(
        vsize_xyz=[0.3, 0.3, 0.25],
        coors_range_xyz=[-54, -54, -5, 54, 54, 3],
        num_point_features=5,
        max_num_voxels=10,
        max_num_points_per_voxel=8,
        device=torch.device("cuda"),
    )
    voxel_input = torch.tensor(
        [[0, 0, 0, 1, 0], [0.01, 0, 0, 2, 0], [1, 0, 0, 3, 0.1]],
        device="cuda",
        dtype=torch.float32,
    )
    voxels, coords, counts = voxelizer(voxel_input)
    assert sorted(counts.cpu().tolist()) == [1, 2]
    res["operators"]["spconv_point_to_voxel"] = {
        "passed": True,
        "coordinates_zyx": coords.cpu().tolist(),
        "point_counts": counts.cpu().tolist(),
    }
    res["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    (
        root / "reports/perception_2025_20261013/mambafusion/operator_smoke.json"
    ).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


def geometry_check():
    import sys, json, pathlib, importlib.util, numpy as np

    root = LAB
    sys.path.insert(0, str(root / "tools"))
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion

    spec = importlib.util.spec_from_file_location(
        "author_nuscenes_utils",
        root / "third_party/MambaFusion2025/pcdet/datasets/nuscenes/nuscenes_utils.py",
    )
    u = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(u)
    nusc = NuScenes(
        version="v1.0-mini", dataroot=str(root / "data/nuscenes"), verbose=False
    )
    errors = []
    sweeps = []
    for split in ["scene0061", "mini_val"]:
        packets = json.load(
            open(root / f"reports/perception_2025_20261013/inputs/{split}_packets.json")
        )
        for p in packets:
            info = packet_to_info(p)
            lid = nusc.get(
                "sample_data", p["sensors"]["LIDAR_TOP"]["sample_data_token"]
            )
            cs = nusc.get("calibrated_sensor", lid["calibrated_sensor_token"])
            pose = nusc.get("ego_pose", lid["ego_pose_token"])
            for camera in CAMERA_ORDER:
                other = u.obtain_sensor2top(
                    nusc,
                    p["sensors"][camera]["sample_data_token"],
                    np.asarray([cs["translation"]]),
                    Quaternion(cs["rotation"]).rotation_matrix,
                    np.asarray(pose["translation"]),
                    Quaternion(pose["rotation"]).rotation_matrix,
                    camera,
                )
                error = max(
                    np.max(
                        np.abs(
                            info["cams"][camera]["sensor2lidar_rotation"]
                            - other["sensor2lidar_rotation"]
                        )
                    ),
                    np.max(
                        np.abs(
                            info["cams"][camera]["sensor2lidar_translation"]
                            - other["sensor2lidar_translation"]
                        )
                    ),
                )
                errors.append(float(error))
            for sw in p["lidar_sweeps"][:9]:
                matrix = np.linalg.inv(
                    np.asarray(p["sensors"]["LIDAR_TOP"]["T_sensor_to_ego"])
                ) @ np.asarray(sw["T_sensor_to_ego_ref"])
                match = next(
                    x
                    for x in info["sweeps"]
                    if x["sample_data_token"] == sw["sample_data_token"]
                )
                sweeps.append(float(np.max(np.abs(matrix - match["transform_matrix"]))))
    assert max(errors) < 1e-9 and max(sweeps) < 1e-9
    result = {
        "status": "passed",
        "camera_matrices_compared": len(errors),
        "camera_max_abs_error": max(errors),
        "sweep_matrices_compared": len(sweeps),
        "sweep_max_abs_error": max(sweeps),
        "camera_reference": "unmodified pinned author obtain_sensor2top",
        "annotation_fields_in_infos": False,
        "native_dataset_integration_validated": False,
    }
    (
        root / "reports/perception_2025_20261013/mambafusion/geometry_check.json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(result)


def cpu_pipeline_check():
    import sys, pathlib, json, logging, time, numpy as np

    root = LAB
    sys.path[:0] = [str(root / "tools"), str(root / "third_party/MambaFusion2025")]
    from easydict import EasyDict
    from pcdet.datasets.nuscenes.nuscenes_dataset import NuScenesDataset

    rows = []
    for split, count in [("scene0061", 39), ("mini_val", 81)]:
        cfg = EasyDict(resolved_config(split))
        ds = NuScenesDataset(
            cfg.DATA_CONFIG,
            cfg.CLASS_NAMES,
            training=False,
            root_path=root / "data/nuscenes",
            logger=logging.getLogger("cpu"),
        )
        assert len(ds) == count
        for idx in [0, count // 2, count - 1]:
            np.random.seed(2025 + idx)
            random.seed(2025 + idx)
            torch.manual_seed(2025 + idx)
            start = time.perf_counter()
            x = ds[idx]
            assert "gt_boxes" not in x and "gt_names" not in x
            assert np.isfinite(x["points"]).all() and x["points"].shape[1] == 5
            np.random.seed(2025 + idx)
            random.seed(2025 + idx)
            torch.manual_seed(2025 + idx)
            y = ds[idx]
            assert np.array_equal(x["points"], y["points"])
            images = np.stack(x["camera_imgs"])
            assert tuple(images.shape) == (6, 3, 256, 704), images.shape
            rows.append(
                {
                    "split": split,
                    "index": idx,
                    "sample_token": x["metadata"]["token"],
                    "points": list(x["points"].shape),
                    "camera_imgs": list(images.shape),
                    "keys": list(x),
                    "cpu_ms_including_repeat_check": (time.perf_counter() - start)
                    * 1000,
                    "seeded_repeat_equal": True,
                    "annotations_absent": True,
                }
            )
    result = {
        "status": "passed",
        "frames_preprocessed": len(rows),
        "inference_frames": 0,
        "pipeline": "unmodified pinned author NuScenesDataset(training=False)",
        "rows": rows,
    }
    (
        root / "reports/perception_2025_20261013/mambafusion/cpu_pipeline.json"
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def validate_runtime_config(config):
    expected = {"source": SOURCE, "checkpoint": WEIGHT,
                "config": SOURCE / "tools/cfgs/mambafusion_models/mamba_fusion.yaml"}
    for key, value in config.get("models", {}).get("mambafusion", {}).items():
        if key in expected:
            base = SOURCE if key == "config" else LAB
            if (base / value).resolve() != expected[key].resolve():
                raise ValueError("Unsupported MambaFusion config override: " + key)
    for key, expected_path in [("data", LAB / "data/nuscenes"), ("report", REPORT.parent)]:
        if key in config and (LAB / config[key]).resolve() != expected_path.resolve():
            raise ValueError("Unsupported MambaFusion config override: " + key)
    for split, path in config.get("splits", {}).items():
        if (LAB / path).resolve() != (REPORT.parent / "inputs" / (split + "_packets.json")).resolve():
            raise ValueError("Unsupported MambaFusion split override: " + split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=[
            "preflight",
            "prepare",
            "hilbert",
            "operator-smoke",
            "geometry-check",
            "cpu-pipeline",
            "run",
            "gate",
            "evaluate",
            "optimize",
        ],
        default="preflight",
    )
    parser.add_argument("--allow-measured-repeat-noise", action="store_true", help="Continue native diagnostic evaluation after hash-bound six-frame noise review; does not pass equivalence gate")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.config:
        validate_runtime_config(json.loads(args.config.read_text()))
    if args.phase == "evaluate":
        from mambafusion_evaluate_2025 import evaluate
        print(json.dumps(evaluate(LAB, REPORT), indent=2))
        return 0
    if args.phase == "prepare":
        print(json.dumps(prepare_infos(), indent=2))
        return 0
    if args.phase in ["hilbert", "operator-smoke"]:
        (generate_hilbert if args.phase == "hilbert" else operator_smoke)()
        return 0
    if args.phase in ["geometry-check", "cpu-pipeline"]:
        (geometry_check if args.phase == "geometry-check" else cpu_pipeline_check)()
        return 0
    status = preflight()
    status["phase"] = args.phase
    dump("status.json", status)
    if args.phase in ("run", "gate") and not status["blockers"]:
        try:
            status = native_run(gate_only=args.phase == "gate", allow_measured_repeat_noise=args.allow_measured_repeat_noise)
        except Exception as exc:
            import traceback

            (REPORT / "native_failure.log").write_text(traceback.format_exc())
            status.update(status="failed", phase="run", blockers=[str(exc)])
            dump("status.json", status)
    elif args.phase == "optimize":
        status["blockers"].append(
            "Native strict-load and 39/81-frame evaluation must pass before optimization."
        )
        dump("status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    # Pending evaluation is a partial result, not a failed execution or missing resource.
    if status["status"] == "failed":
        return 1
    return 2 if status["status"] in ("blocked_resource", "blocked_data") else 0


if __name__ == "__main__":
    raise SystemExit(main())
