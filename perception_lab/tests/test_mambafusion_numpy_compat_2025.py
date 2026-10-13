"""CPU-only proof of exact patch scope, idempotence, and ragged metadata semantics."""
import ast
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from mambafusion_numpy_compat_2025 import LAB, PIN, FILES, apply, digest, transform


@pytest.fixture
def upstream_tree(tmp_path):
    source = LAB/'third_party/MambaFusion2025'
    for name, (original, _) in FILES.items():
        data = subprocess.check_output(['git', '-C', str(source), 'show', PIN+':'+name])
        assert digest(data) == original
        destination = tmp_path/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    return tmp_path


def test_exact_patch_and_second_application_is_noop(upstream_tree):
    before = {name:(upstream_tree/name).read_bytes() for name in FILES}
    first = apply(upstream_tree)
    second = apply(upstream_tree)
    assert all(x['action']=='applied' for x in first['files'])
    assert all(x['action']=='already_applied' for x in second['files'])
    for name, (_, expected) in FILES.items():
        current = (upstream_tree/name).read_bytes()
        assert current == transform(name,before[name])
        assert digest(current) == expected
    backbone = before['pcdet/models/backbones_2d/base_bev_backbone.py']
    assert backbone.count(b'astype(np.int)') == 3
    for reciprocal_stride in [1,2,4,8]:
        assert np.round(reciprocal_stride).astype(int) == reciprocal_stride


def test_unknown_second_file_rejected_without_partial_edit(upstream_tree):
    backbone = upstream_tree/'pcdet/models/backbones_2d/base_bev_backbone.py'
    original = backbone.read_bytes()
    dataset = upstream_tree/'pcdet/datasets/dataset.py'
    dataset.write_bytes(dataset.read_bytes()+b'\n# unrelated edit\n')
    with pytest.raises(ValueError,match='Unknown source hash'):
        apply(upstream_tree)
    assert backbone.read_bytes() == original


def test_real_collate_preserves_six_camera_metadata_and_other_fields(upstream_tree):
    apply(upstream_tree)
    tree = ast.parse((upstream_tree/'pcdet/datasets/dataset.py').read_text())
    function = next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='collate_batch')
    function.decorator_list = []
    namespace = {'np':np,'defaultdict':defaultdict}
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<author collate>', 'exec'),namespace)
    metadata = [[.5+i*.01,(0,8,704,264),False,0] for i in range(6)]
    points = np.array([[1.,2.,3.,.7]],dtype=np.float32)
    result = namespace['collate_batch']([{'img_process_infos':metadata,'points':points,'frame_id':'fixture'}])
    assert result['img_process_infos'].shape == (1,6,4)
    assert result['img_process_infos'].dtype == object
    assert result['img_process_infos'][0].tolist() == metadata
    assert np.array_equal(result['points'],np.array([[0.,1.,2.,3.,.7]],dtype=np.float32))
    assert result['frame_id'].tolist() == ['fixture']
    assert result['batch_size'] == 1
