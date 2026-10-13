"""Safety gates for a genuine MambaFusion deployment, with no model substitution."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from mambafusion_2025 import (
    validate_checkpoint_file,
    validate_author_config,
    check_sweep_inventory,
)


def test_html_is_never_a_checkpoint(tmp_path):
    p = tmp_path / "weights.pth"
    p.write_text("<!DOCTYPE html><html>download warning</html>")
    with pytest.raises(ValueError, match="HTML"):
        validate_checkpoint_file(p)


def test_partial_file_rejected_even_if_pickle_header(tmp_path):
    p = tmp_path / "weights.pth"
    p.write_bytes(b"\x80\x02" + b"0" * 1024)
    with pytest.raises(ValueError, match="incomplete"):
        validate_checkpoint_file(p, expected_bytes=391568021)


def test_timestamp_is_not_ring():
    cfg = {
        "DATA_CONFIG": {
            "POINT_CLOUD_RANGE": [-54, -54, -5, 54, 54, 3],
            "CAMERA_CONFIG": {"USE_CAMERA": True, "IMAGE": {"FINAL_DIM": [256, 704]}},
            "POINT_FEATURE_ENCODING": {
                "used_feature_list": ["x", "y", "z", "intensity", "ring"]
            },
            "MAX_SWEEPS": 10,
            "DATA_PROCESSOR": [
                {
                    "NAME": "transform_points_to_voxels_placeholder",
                    "VOXEL_SIZE": [0.3, 0.3, 0.25],
                }
            ],
        },
        "MODEL": {"NAME": "MambaFusion", "DENSE_HEAD": {"NUM_PROPOSALS": 200}},
    }
    with pytest.raises(ValueError, match="timestamp"):
        validate_author_config(cfg)


def test_sweeps_preserve_short_history_and_reject_future():
    rows = [{"token": "a", "sweeps": [{"time_lag": 0.0}, {"time_lag": 0.1}]}]
    assert check_sweep_inventory(rows)[0]["available_history"] == 2
    rows[0]["sweeps"].append({"time_lag": -0.1})
    with pytest.raises(ValueError, match="future"):
        check_sweep_inventory(rows)


def test_inference_infos_pad_author_history_without_gt():
    import numpy as np
    from mambafusion_2025 import packet_to_info, CAMERA_ORDER

    I = np.eye(4).tolist()
    sensor = {
        "path": "/data/current.bin",
        "sample_data_token": "lidar",
        "timestamp_us": 1000000,
        "T_sensor_to_ego": I,
        "T_ego_to_global": I,
        "K": np.eye(3).tolist(),
    }
    p = {"sample_token": "frame", "sensors": {"LIDAR_TOP": sensor}, "lidar_sweeps": []}
    for c in CAMERA_ORDER:
        p["sensors"][c] = dict(sensor, path="/data/" + c + ".jpg")
    p["annotations"] = [{"tracking_id": "DO_NOT_USE"}]
    info = packet_to_info(p)
    assert len(info["sweeps"]) == 9
    assert all(
        s["time_lag"] == 0 and s["transform_matrix"] is None for s in info["sweeps"]
    )
    assert not any(k.startswith("gt") for k in info)
    assert list(info["cams"]) == CAMERA_ORDER
    assert np.allclose(info["cams"]["CAM_FRONT"]["sensor2lidar_rotation"], np.eye(3))


def test_enabled_author_fusion_cannot_be_silently_disabled():
    import json

    cfg = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "reports/perception_2025_20261013/mambafusion/resolved_mini_val.json"
        ).read_text()
    )
    cfg["MODEL"]["MM_BACKBONE"]["USE_ALL_MAMBA"] = False
    with pytest.raises(ValueError, match="USE_ALL_MAMBA"):
        validate_author_config(cfg)


@pytest.mark.parametrize('state,expected', [('partial', 0), ('blocked_resource', 2), ('failed', 1)])
def test_runner_receives_status_consistent_exit_code(monkeypatch, state, expected):
    import mambafusion_2025 as module
    monkeypatch.setattr(sys, 'argv', ['mamba', '--phase', 'run'])
    monkeypatch.setattr(module, 'preflight', lambda: {'status': state, 'blockers': ['evaluation or resource pending']})
    monkeypatch.setattr(module, 'dump', lambda *args: None)
    assert module.main() == expected


def test_runtime_config_rejects_silently_ignored_paths():
    import json
    import mambafusion_2025 as module
    cfg = json.loads((module.LAB/'configs/perception_2025.yaml').read_text())
    module.validate_runtime_config(cfg)
    cfg['models']['mambafusion']['checkpoint'] = 'different.pth'
    with pytest.raises(ValueError, match='checkpoint'):
        module.validate_runtime_config(cfg)
