#!/usr/bin/env python3
"""Recover only the two pinned author checkpoints; never mutate historical partials."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'perception_lab/reports/perception_weights_recovery_20261013'
SOURCES = {
    'harpnext': {'url': 'https://huggingface.co/samir-abh/HARP-NeXt/resolve/c2cd3ce5940be23bfbc2d1cf22131fd2fd3dc149/harpnext-nuscenes-32x480/ckpt_best.pth?download=true', 'revision': 'c2cd3ce5940be23bfbc2d1cf22131fd2fd3dc149', 'size': 64865210, 'sha256': '6f7fc5a1c03e9123541458358b1c703babf37ffc0b8b4ea6beac6c9f8478a147', 'filename': 'ckpt_best.pth', 'key': 'net'},
    'mambafusion': {'url': 'https://drive.usercontent.google.com/download?id=10-FBWHbsCMJseKovJwGKncJ96Flp3umM&export=download&confirm=t', 'file_id': '10-FBWHbsCMJseKovJwGKncJ96Flp3umM', 'size': 391545251, 'sha256': None, 'filename': 'checkpoint_epoch_10.pth', 'key': 'model_state'},
}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def binary_prefix(data, content_type=''):
    prefix = data.lstrip().lower()
    if any(t in content_type.lower() for t in ('html', 'json', 'text/')) or prefix.startswith((b'<', b'{', b'[')):
        raise ValueError('non-checkpoint response')
    if not data.startswith((b'PK\x03\x04', b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05')):
        raise ValueError('unrecognized torch serialization prefix')


def receive(response, output, start, end, total, event):
    """A Range-ignoring 200 writes a NEW whole-file staging object, never appends."""
    status = response.status
    event.update(status=status if isinstance(status, int) else None, bytes_received=0)
    if status == 206:
        if response.headers.get('Content-Range') != f'bytes {start}-{end}/{total}':
            raise ValueError('Content-Range mismatch')
        event['content_range'] = {'start': start, 'end': end, 'total': total}
        expected = end - start + 1
        whole = False
    elif status == 200:
        expected, whole = total, True
    else:
        raise ValueError(f'HTTP status {status} is not completion')
    length = response.headers.get('Content-Length')
    if length is not None and int(length) != expected:
        raise ValueError('Content-Length mismatch')
    content_type = response.headers.get('Content-Type', '')
    if any(t in content_type.lower() for t in ('html', 'json', 'text/')):
        raise ValueError('non-binary content type')
    with open(output, 'wb') as f:
        while True:
            data = response.read(min(1024 * 1024, expected + 1 - event['bytes_received']))
            if not data:
                break
            if event['bytes_received'] == 0 and (start == 0 or whole):
                binary_prefix(data, content_type)
            f.write(data)
            event['bytes_received'] += len(data)
            if event['bytes_received'] > expected:
                raise ValueError('oversized response')
        if event['bytes_received'] != expected:
            raise ValueError('incomplete response')
        f.flush()
        os.fsync(f.fileno())
    return whole


def verified_blocks(directory, entries, total=None, block_size=None):
    """Treat persisted index as untrusted; only canonical contained block files qualify."""
    directory = Path(directory).resolve()
    valid = []
    occupied = []
    if not isinstance(entries, list):
        return valid
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        start, end, size = (entry.get(k) for k in ('start', 'end', 'size'))
        if any(type(v) is not int for v in (start, end, size)):
            continue
        if start < 0 or end < start or size != end - start + 1:
            continue
        if total is not None and end >= total:
            continue
        if block_size is not None and (start % block_size or end != min(start + block_size, total) - 1):
            continue
        filename = entry.get('filename')
        if filename != f'{start:012d}-{end:012d}.block':
            continue
        sha = entry.get('sha256')
        if not isinstance(sha, str) or not re.fullmatch(r'[0-9a-f]{64}', sha):
            continue
        path = directory / filename
        if path.is_symlink() or path.resolve().parent != directory:
            continue
        if any(start <= old_end and end >= old_start for old_start, old_end in occupied):
            continue
        if path.is_file() and path.stat().st_size == size and digest(path) == sha:
            valid.append(entry)
            occupied.append((start, end))
    return valid


def download(name, network, block_size=16 * 1024 * 1024):
    spec = SOURCES[name]
    directory = ROOT / f'perception_lab/checkpoints/{name}_2025/recovery_staging'
    directory.mkdir(parents=True, exist_ok=True)
    index = directory / 'blocks.json'
    entries = verified_blocks(directory, json.loads(index.read_text()) if index.exists() else [], spec['size'], block_size)
    save_json(index, entries)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}) if network == 'direct' else urllib.request.ProxyHandler())
    log_path = REPORT / f'{name}_attempts.json'
    logs = json.loads(log_path.read_text()) if log_path.exists() else []
    failures = 0
    for start in range(0, spec['size'], block_size):
        end = min(start + block_size, spec['size']) - 1
        if any(e['start'] == start and e['end'] == end for e in entries):
            continue
        while True:
            event = {'time_unix': time.time(), 'model': name, 'network': network, 'request_range': f'bytes={start}-{end}'}
            temporary = directory / 'incoming.part'
            try:
                request = urllib.request.Request(spec['url'], headers={'Range': event['request_range'], 'Accept-Encoding': 'identity'})
                with opener.open(request, timeout=30) as response:
                    whole = receive(response, temporary, start, end, spec['size'], event)
                if whole:
                    path = directory / 'complete.staging'
                    os.replace(temporary, path)
                    event['result'] = 'whole_file_received_without_append'
                    logs.append(event)
                    save_json(log_path, logs)
                    return path
                path = directory / f'{start:012d}-{end:012d}.block'
                os.replace(temporary, path)
                entry = {'start': start, 'end': end, 'size': end-start+1, 'filename': path.name, 'sha256': digest(path)}
                committed_entries = [e for e in entries if e['start'] != start] + [entry]
                save_json(index, committed_entries)
                entries = committed_entries
                event['result'] = 'verified_complete_block'
                logs.append(event)
                save_json(log_path, logs)
                failures = 0
                break
            except Exception as exc:
                # Exception messages can contain signed redirect URLs. Record type only.
                event.update(result='failed', error_type=type(exc).__name__)
                if isinstance(exc, urllib.error.HTTPError):
                    event['status'] = exc.code
                if isinstance(exc, ValueError):
                    event['reason'] = 'validation_error'
                logs.append(event)
                save_json(log_path, logs)
                failures += 1
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 413:
                    raise RuntimeError('HTTP413 hard limit; switch transport immediately') from None
                if failures >= 2:
                    raise RuntimeError('two attempts without verified progress; switch network node') from None
    stage = directory / 'complete.staging'
    with stage.open('wb') as output:
        for entry in sorted(verified_blocks(directory, entries, spec['size'], block_size), key=lambda e: e['start']):
            with (directory / entry['filename']).open('rb') as source:
                for data in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(data)
        output.flush()
        os.fsync(output.fileno())
    return stage


def cpu_parse(path, name):
    python = ROOT / f'perception_lab/envs/{name}_2025/bin/python'
    code = """import torch, sys, json
p = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
if not isinstance(p, dict):
    raise ValueError('checkpoint root must be a dictionary')
s = p.get(sys.argv[2])
if not isinstance(s, dict) or not s:
    raise ValueError('checkpoint state must be a nonempty dictionary')
print(json.dumps({'root_keys': list(p), 'state_entries': len(s), 'device': 'cpu'}))
"""
    result = subprocess.run([str(python), '-c', code, str(path), SOURCES[name]['key']], capture_output=True, text=True)
    if result.returncode:
        raise ValueError('CPU checkpoint structure parse failed')
    return json.loads(result.stdout.strip().splitlines()[-1])


def install(stage, final, size, expected_sha, parser):
    stage, final = Path(stage), Path(final)
    if stage.stat().st_size != size:
        raise ValueError('complete file size mismatch')
    with stage.open('rb') as f:
        binary_prefix(f.read(32))
    sha = digest(stage)
    if expected_sha and sha != expected_sha:
        raise ValueError('complete SHA256 mismatch')
    parsed = parser(stage)
    final.parent.mkdir(parents=True, exist_ok=True)
    if stage.stat().st_dev != final.parent.stat().st_dev:
        raise ValueError('atomic installation requires same filesystem')
    os.replace(stage, final)
    fd = os.open(str(final.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'path': os.path.relpath(final, ROOT), 'bytes': size, 'sha256': sha, 'cpu_parse': parsed, 'atomic_install': True, 'installed_unix': time.time()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=SOURCES, required=True)
    ap.add_argument('--network', choices=['proxy','direct'], default='proxy')
    ap.add_argument('--install-from', type=Path, help='validated relay or download staging file on checkpoint filesystem')
    ap.add_argument('--relay-sha256', help='full file hash computed at relay node; not an author hash')
    args = ap.parse_args()
    if args.model == 'mambafusion' and args.install_from and not args.relay_sha256:
        ap.error('Mamba imported artifact requires --relay-sha256 from source node')
    if args.relay_sha256 and not re.fullmatch(r'[0-9a-f]{64}', args.relay_sha256):
        ap.error('--relay-sha256 must be a lowercase SHA256 digest')
    spec = SOURCES[args.model]
    stage = args.install_from or download(args.model, args.network)
    if args.relay_sha256 and digest(stage) != args.relay_sha256:
        raise ValueError('relay sender/receiver hash mismatch')
    target = ROOT / f'perception_lab/checkpoints/{args.model}_2025' / spec['filename']
    receipt = install(stage, target, spec['size'], spec['sha256'], lambda p: cpu_parse(p,args.model))
    receipt.update(source=spec, route='relay' if args.install_from and args.relay_sha256 else args.network, relay_sha256=args.relay_sha256, author_hash_available=bool(spec['sha256']))
    save_json(REPORT / f'{args.model}_atomic_receipt.json', receipt)
    manifest = REPORT / 'download_manifest.json'
    state = json.loads(manifest.read_text()) if manifest.exists() else {}
    state[args.model] = receipt
    save_json(manifest, state)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
