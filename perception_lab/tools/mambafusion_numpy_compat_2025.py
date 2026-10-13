"""Apply only the two audited NumPy compatibility edits to pinned MambaFusion.

Accepts exact upstream or exact patched file hashes; rejects unknown source edits.
No Torch imports, GPU execution, environment changes, or Git index operations.
"""
import argparse
import hashlib
import json
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
PIN = 'd85c2feeb19207b20e42d50421afb11c095f3efa'
FILES = {
    'pcdet/models/backbones_2d/base_bev_backbone.py': (
        '49923055c5cec910fee86a559623b6aaf7d227e9a6b7bd21a23d086534ab2cc8',
        '45b84de996bf35ca611825d137585d97995dfae36deb553687d1083a02307a40'),
    'pcdet/datasets/dataset.py': (
        'ae0c9902e34f667718845992d24c57934949f395465d1848306e092c9f4f9ece',
        '1a0ef5f6135a4bbc81ad85d6220e2a5384b2602eb734110057562d62bae55e8c'),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(name, data):
    if name.endswith('base_bev_backbone.py'):
        if data.count(b'astype(np.int)') != 3:
            raise ValueError('Expected exactly three deprecated integer aliases')
        return data.replace(b'astype(np.int)', b'astype(int)')
    if name == 'pcdet/datasets/dataset.py':
        anchor = b"                elif key in ['calib']:\n"
        if data.count(anchor) != 1:
            raise ValueError('Expected exactly one collate insertion anchor')
        branch = (b"                elif key == 'img_process_infos':\n"
                  b"                    ret[key] = np.stack([np.asarray(item, dtype=object) for item in val], axis=0)\n")
        return data.replace(anchor, branch + anchor)
    raise ValueError('Unsupported source file')


def apply(source):
    changes = []
    # Validate every file before writing either, so unknown edits fail closed.
    for name, (original, patched) in FILES.items():
        path = source/name
        data = path.read_bytes()
        actual = digest(data)
        if actual == patched:
            changes.append((path, data, 'already_applied', original, patched))
        elif actual == original:
            result = transform(name, data)
            if digest(result) != patched:
                raise ValueError('Unexpected transformed source hash: ' + name)
            changes.append((path, result, 'applied', original, patched))
        else:
            raise ValueError('Unknown source hash; refusing overwrite: ' + name)
    for path, data, action, _, _ in changes:
        if action == 'applied':
            temporary = path.with_name(path.name + '.numpy_compat.tmp')
            temporary.write_bytes(data)
            temporary.replace(path)
    return {'status': 'passed', 'upstream_commit': PIN, 'files': [
        {'path': str(path.relative_to(source)), 'action': action,
         'upstream_sha256': original, 'patched_sha256': patched}
        for path, _, action, original, patched in changes]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=LAB/'third_party/MambaFusion2025')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = apply(args.source)
    rendered = json.dumps(result, indent=2) + '\n'
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()
