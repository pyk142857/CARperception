"""Serial, subprocess-based execution with immutable runs and fail-closed evidence."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import fcntl
from evidence import hash_files, reusable

PHASES = {'baseline': ['smoke', 'replay', 'visualization'],
          'evaluate': ['evaluation'], 'deploy': ['benchmark', 'export', 'export_validation'],
          'experiment': ['train_A', 'train_B', 'evaluation'], 'report': ['report'],
          'preflight': ['preflight']}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--phase', choices=[*PHASES, 'all'])
    g.add_argument('--module')
    p.add_argument('--stage', default='all')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    plan_path = Path(args.plan).resolve()
    # JSON is a valid YAML 1.2 subset; also accept general YAML when installed.
    try:
        plan = json.loads(plan_path.read_text())
    except json.JSONDecodeError:
        import yaml
        plan = yaml.safe_load(plan_path.read_text())
    root = Path(plan.get('root', str(plan_path.parent.parent))).resolve()
    os.chdir(root)
    status_path = root / 'results/status.json'
    status_path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = (root / 'results/.pipeline.lock').open('a')
    try:
        fcntl.flock(process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another pipeline is active; refusing concurrent status updates.', file=sys.stderr)
        return 75
    status = json.loads(status_path.read_text()) if status_path.exists() else {'modules': {}}
    run_id = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    if args.module and args.module not in plan['modules']:
        p.error('Unknown module: ' + args.module)
    failed = False
    for mid, module in plan['modules'].items():
        if args.module and mid != args.module:
            continue
        if args.phase == 'experiment' and mid != 'M13':
            continue
        for stage, job in module.get('jobs', {}).items():
            if args.phase not in (None, 'all') and stage not in PHASES[args.phase] and job.get('phase') != args.phase:
                continue
            if args.module and args.stage != 'all' and args.stage != stage:
                continue
            records = status['modules'].setdefault(mid, {})
            out = root / 'runs' / run_id / mid / stage
            out.mkdir(parents=True)
            inputs = [plan_path, *Path(__file__).parent.glob('*.py')]
            missing = []
            for path in job.get('inputs', []):
                path = Path(path)
                if path.is_file():
                    inputs.append(path)
                else:
                    missing.append(str(path))
            for listing in job.get('input_lists', []):
                if not Path(listing).is_file():
                    missing.append(listing)
                    continue
                inputs.append(Path(listing))
                for item in Path(listing).read_text().splitlines():
                    if Path(item).is_file():
                        inputs.append(Path(item))
                    else:
                        missing.append(item)
            for listing in job.get('image_manifests', []):
                if not Path(listing).is_file():
                    missing.append(listing)
                    continue
                inputs.append(Path(listing))
                for image in json.loads(Path(listing).read_text())['images']:
                    if Path(image['path']).is_file():
                        inputs.append(Path(image['path']))
                    else:
                        missing.append(image['path'])
            for directory in job.get('code_roots', []):
                inputs.extend(Path(directory).rglob('*.py'))
            hashes = hash_files(inputs)
            fingerprint = hashlib.sha256(json.dumps({'job': job, 'hashes': hashes}, sort_keys=True).encode()).hexdigest()
            old = records.get(stage, {})
            if args.resume and reusable(old, fingerprint):
                print(mid, stage, 'verified cached evidence', flush=True)
                continue
            record = {'run_id': run_id, 'status': 'running', 'started_at': now(),
                      'command': job.get('command'), 'fingerprint': fingerprint,
                      'input_config_code_weight_hashes': hashes, 'artifacts': {}, 'exit_code': None}
            records[stage] = record
            reasons = []
            if missing:
                reasons.append('Missing input files: ' + ', '.join(missing))
            for dependency in job.get('depends', []):
                dm, ds = dependency.split(':')
                dep = status['modules'].get(dm, {}).get(ds, {})
                if not reusable(dep, dep.get('fingerprint')):
                    reasons.append('Dependency unavailable or corrupt: ' + dependency)
            if job.get('blocked_reason'):
                reasons.append(job['blocked_reason'])
            if not job.get('command'):
                reasons.append('No validated execution command configured')
            if job.get('not_applicable_reason'):
                record.update(status='not_applicable', reason=job['not_applicable_reason'])
            elif reasons:
                record.update(status='blocked', reason='; '.join(reasons))
            else:
                command = [s.replace('{out}', str(out)).replace('{root}', str(root)) for s in job['command']]
                record['command'] = command
                write_json(status_path, status)
                (root / 'TASK_STATE.md').write_text('# TASK_STATE — executing / partial\n\n'
                    + f'Run: {run_id}\n\nCurrent: {mid}/{stage}\n\nCommand: `{command!r}`\n\n'
                    + 'Evidence: results/status.json and runs/. No milestone claimed.\n')
                with (out / 'command.log').open('w') as log:
                    try:
                        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                                   YOLO_CONFIG_DIR=str(root / 'envs/yolo_settings'))
                        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                        record['exit_code'] = proc.returncode
                        artifacts = [Path(s.replace('{out}', str(out))) for s in job.get('artifacts', [])]
                        # Also bind native arrays, figures, and other outputs into the run evidence.
                        declared_complete = all(x.is_file() for x in artifacts)
                        all_outputs = [x for x in out.rglob('*') if x.is_file() and x.name not in ('command.log', 'status.json')]
                        if proc.returncode:
                            record.update(status='failed', reason=f'Command exit {proc.returncode}; see command.log')
                        elif not artifacts or not declared_complete:
                            record.update(status='failed', reason='Command produced no complete declared evidence')
                        else:
                            record.update(status='passed', artifacts=hash_files(set(artifacts + all_outputs)))
                    except OSError as error:
                        log.write(str(error))
                        record.update(status='failed', reason=str(error))
                record['log'] = str(out / 'command.log')
            record['finished_at'] = now()
            failed |= record['status'] not in ('passed', 'not_applicable')
            write_json(out / 'status.json', record)
            write_json(status_path, status)
            print(mid, stage, record['status'], record.get('reason', ''), flush=True)
    status['updated_at'] = now()
    write_json(status_path, status)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
