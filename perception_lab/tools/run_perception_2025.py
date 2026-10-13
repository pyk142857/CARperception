"""Serial orchestration of the four 2025 candidates; never runs historical models."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
MODELS = ('deimv2','harpnext','mambafusion','grae3dmot')
STAGES = ('preflight', *MODELS, 'optimize', 'visualize', 'summarize')

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)

def commands(config, stage, config_path):
    env = config['environments']
    if stage == 'preflight':
        return [[env['preflight'],'tools/deployment_preflight.py','--out',config['report']+'/inputs'], [env['preflight'],'tools/perception_2025_preflight.py']]
    if stage in MODELS:
        if stage == 'grae3dmot':
            return [[env['grae3dmot'] if phase == 'run' else env.get('grae3dmot_eval','envs/sparse4d/bin/python'),
                     'tools/grae3dmot_2025.py','--phase',phase,'--config',str(config_path)]
                    for phase in ('run','evaluate','audit')]
        return [[env[stage], f'tools/{stage}_2025.py','--phase','run','--config',str(config_path)]]
    if stage == 'optimize':
        return [[env[m], f'tools/{m}_2025.py','--phase','optimize','--config',str(config_path)] for m in MODELS]
    if stage == 'visualize':
        return [[env.get('visualize', sys.executable),'tools/perception_2025_visualize.py','--config',str(config_path)]]
    if stage == 'summarize':
        return [[sys.executable,'tools/perception_2025_summarize.py','--config',str(config_path)]]
    raise ValueError(stage)

def hash_file(path):
    # Never cache by timestamps: some filesystems can preserve the same timestamp
    # for two writes, including same-length corruption of a prediction artifact.
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot(path):
    try:
        stat = path.stat()
        return {'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns,
                'ctime_ns': stat.st_ctime_ns, 'size': stat.st_size, 'sha256': hash_file(path)}
    except OSError:
        return None


def classify_run(returncode, result_paths, before, require_result=True):
    """Only this invocation's newly written evidence may classify its return code."""
    fresh = [p for p in result_paths if snapshot(p) is not None and snapshot(p) != before.get(p)]
    result = {'returncode': returncode, 'fresh_result': bool(fresh)}
    if not fresh:
        result.update(status='completed' if returncode == 0 and not require_result else 'failed',
                      reason='No fresh result from this invocation')
        return result
    path = fresh[0]
    try:
        module = json.loads(path.read_text())
        status = module['status']
    except (OSError, ValueError, KeyError, TypeError):
        result.update(status='failed', reason='Malformed fresh status JSON')
        return result
    result.update(result=str(path), result_sha256=hash_file(path), module_status=status)
    if returncode == 2 and status in ('blocked_resource', 'blocked_data'):
        result['status'] = status
    elif returncode == 0 and status in ('passed', 'partial', 'blocked_resource', 'blocked_data', 'failed'):
        result['status'] = status
    else:
        result.update(status='failed', reason='Exit code and fresh result do not report successful or blocked completion')
    return result


def output_proof(manifest_path, status_path):
    """Modules opt into recovery by publishing a complete explicit artifact list.

    Schema: {"complete": true, "artifacts": [{"path": <project-relative or absolute>,
    "sha256": <digest>}, ...]}. The status and all prediction/metric artifacts must
    be listed. Missing manifests are deliberately not synthesized from a directory.
    """
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('complete') is not True or not manifest.get('artifacts'):
            return None
        proof = {'manifest_sha256': hash_file(manifest_path), 'artifacts': {}}
        for record in manifest['artifacts']:
            path = Path(record['path'])
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            if not path.is_file() or hash_file(path) != record['sha256']:
                return None
            proof['artifacts'][str(path)] = record['sha256']
        if str(status_path.resolve()) not in proof['artifacts']:
            return None
        return proof
    except (OSError, ValueError, KeyError, TypeError):
        return None


def can_resume(previous, current_fingerprint, manifest_path, status_path):
    if previous.get('status') != 'passed' or previous.get('fingerprint') != current_fingerprint:
        return False
    proof = output_proof(manifest_path, status_path)
    return proof is not None and proof == previous.get('output_proof')


def environment_snapshot(executable):
    """Hash actual interpreter, installed Python/native code, and distribution metadata.

    No Torch import or GPU initialization. Content is read again on every check;
    metadata timestamps alone cannot certify installed native code is unchanged.
    """
    path = Path(executable)
    if not path.is_absolute():
        path = ROOT / path
    probe = "import sys,json,importlib.metadata as m; print(json.dumps({'version':sys.version,'prefix':sys.prefix,'paths':sys.path,'packages':sorted((d.metadata.get('Name',''),d.version) for d in m.distributions())}))"
    try:
        proc = subprocess.run([str(path), '-c', probe], cwd=ROOT, capture_output=True,
                              text=True, timeout=30)
        if proc.returncode:
            return {'executable': str(path), 'probe_error': proc.stderr[-2000:]}
        info = json.loads(proc.stdout)
        digest = hashlib.sha256(json.dumps(info, sort_keys=True).encode())
        files = {path, path.parent.parent/'pyvenv.cfg'}
        for directory in info['paths']:
            base = Path(directory)
            if base.name not in ('site-packages', 'dist-packages') or not base.is_dir():
                continue
            for item in base.rglob('*'):
                if item.is_file() and (item.suffix in ('.py', '.so', '.pth') or
                    item.name in ('METADATA', 'RECORD', 'WHEEL', 'direct_url.json', 'PKG-INFO')):
                    files.add(item)
        for item in sorted(files):
            digest.update(str(item).encode())
            digest.update(hash_file(item).encode() if item.is_file() else b'MISSING')
        return {'executable': str(path), 'code_and_metadata_sha256': digest.hexdigest(),
                'python_version': info['version'], 'hashed_files': len(files)}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        return {'executable': str(path), 'probe_error': str(exc)}


def source_paths(source):
    if not source.is_dir():
        return [source]
    # Both tracked content and local additions/patches affect the model executed.
    try:
        top = subprocess.run(['git', '-C', str(source), 'rev-parse', '--show-toplevel'],
                             capture_output=True, check=True, text=True).stdout.strip()
        if Path(top).resolve() != source.resolve():
            raise ValueError('Archive directory belongs to a parent repository')
        result = subprocess.run(['git', '-C', str(source), 'ls-files', '-z', '--cached',
                                 '--others', '--exclude-standard'], capture_output=True, check=True)
        files = [source / name.decode() for name in result.stdout.split(b'\0') if name]
    except (OSError, ValueError, subprocess.SubprocessError):
        ignored = {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'}
        files = [p for p in source.rglob('*') if p.is_file() and not ignored.intersection(p.parts)]
    return files


def data_paths(config):
    """Hash raw bytes referenced by inventories, not merely their recorded hashes."""
    paths = []
    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else ROOT/path
    for value in config.get('splits', {}).values():
        manifest = resolve(value)
        paths.append(manifest)
        if manifest.exists():
            for packet in json.loads(manifest.read_text()):
                for sensor in list(packet.get('sensors', {}).values()) + packet.get('lidar_sweeps', []):
                    if sensor.get('path'):
                        paths.append(resolve(sensor['path']))
    report = resolve(config.get('report', 'reports/perception_2025_20261013'))
    for name, keys in [('coco_files.json', ['file']), ('lidarseg_files.json', ['point_file', 'label_file'])]:
        manifest = report/'inputs'/name
        paths.append(manifest)
        if manifest.exists():
            for row in json.loads(manifest.read_text()):
                paths.extend(resolve(row[key]) for key in keys)
    if config.get('coco'):
        paths.append(resolve(config['coco'])/'annotations/instances_val2017.json')
    if config.get('data'):
        metadata = resolve(config['data'])/'v1.0-mini'
        paths.extend(metadata.glob('*.json'))
    return paths


def fingerprint(config, stage, cmds):
    h = hashlib.sha256(json.dumps({'config':config,'stage':stage,'commands':cmds},sort_keys=True).encode())
    paths = [Path(__file__)]
    paths.extend(ROOT/'tools'/p.name for p in (ROOT/'tools').glob('*.py'))
    paths.extend(ROOT/arg for cmd in cmds for arg in cmd if arg.endswith('.py'))
    paths.extend(data_paths(config))
    models = [stage] if stage in MODELS else list(config.get('models',{}))
    for name in models:
        model = config.get('models',{}).get(name,{})
        source = ROOT/model.get('source', 'third_party/NOT_CONFIGURED_'+name)
        paths.extend(source_paths(source))
        for key in ('checkpoint','config'):
            if model.get(key):
                path = ROOT/model[key]
                if not path.exists():
                    path = source/model[key]
                paths.append(path)
    for path in sorted(set(paths)):
        h.update(str(path).encode())
        h.update(hash_file(path).encode() if path.is_file() else b'MISSING')
    for executable in sorted(set(cmd[0] for cmd in cmds)):
        environment = environment_snapshot(executable)
        h.update(json.dumps(environment,sort_keys=True).encode())
        if environment.get('probe_error'):
            # Uninspectable environments are allowed to attempt execution, never resume.
            h.update(str(time.time_ns()).encode())
    return h.hexdigest()


def result_paths(out, stage, index):
    if stage == 'grae3dmot' and index:
        return [out/'grae3dmot'/('evaluate_status.json' if index == 1 else 'audit_status.json')]
    if stage == 'preflight':
        return [out/'inputs'/'summary.json' if index == 0 else out/'preflight.json']
    if stage == 'visualize':
        return [out/'visualization'/'status.json']
    if stage == 'optimize':
        return [out/MODELS[index]/'optimization.json', out/MODELS[index]/'status.json']
    return [out/stage/'status.json']


def aggregate(runs):
    statuses = [run['status'] for run in runs]
    for status in ('failed', 'blocked_resource', 'blocked_data', 'partial', 'completed'):
        if status in statuses:
            return status
    return 'passed' if statuses and all(s == 'passed' for s in statuses) else 'failed'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='configs/perception_2025.yaml')
    ap.add_argument('--phase',choices=(*STAGES,'all'),default='all')
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(); cp=Path(a.config)
    if not cp.is_absolute():cp=ROOT/cp
    config=json.loads(cp.read_text()); out=ROOT/config['report'];out.mkdir(parents=True,exist_ok=True)
    phases=STAGES if a.phase=='all' else (a.phase,)
    state_path=out/'runner_state.json';state={}
    # Shared lock prevents overlapping invocations of this orchestrator.
    lock_path=ROOT/'runs/perception_2025_gpu.lock';lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        for stage in phases:
            cmds=commands(config,stage,cp)
            if a.dry_run:print(json.dumps({'stage':stage,'commands':cmds}));continue
            fp=fingerprint(config,stage,cmds)
            old = state.get(stage, {})
            directory = out/('visualization' if stage == 'visualize' else stage)
            result = directory/'status.json'
            manifest = directory/'artifact_manifest.json'
            if a.resume and can_resume(old, fp, manifest, result):
                continue
            row = {'status':'running', 'fingerprint':fp, 'commands':cmds,
                   'start':time.time(), 'runs':[]}
            state[stage] = row
            write(state_path, state)
            for i, cmd in enumerate(cmds):
                log = out/'logs'/f'{stage}_{i}.log'
                log.parent.mkdir(exist_ok=True)
                expected = result_paths(out, stage, i)
                before = {path:snapshot(path) for path in expected}
                with log.open('w') as stream:
                    try:
                        rc = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT).returncode
                    except OSError as exc:
                        stream.write(str(exc)+'\n')
                        rc = 127
                run = classify_run(rc, expected, before, require_result=True)
                run['log'] = str(log.relative_to(out))
                row['runs'].append(run)
                if stage == 'grae3dmot' and run['status'] not in ('passed','partial'):
                    # Dependent evaluation must never consume an older inference artifact.
                    break
            row['status'] = aggregate(row['runs'])
            row['finish'] = time.time()
            # No manifest means no resume, even when a process returned zero.
            if row['status'] == 'passed':
                proof = output_proof(manifest, result)
                if proof is not None:
                    row['output_proof'] = proof
                else:
                    row['resume_disabled'] = 'No verified complete artifact manifest'
            write(state_path, state)
    return int(any(state.get(s,{}).get('status')=='failed' for s in phases))
if __name__=='__main__':sys.exit(main())
