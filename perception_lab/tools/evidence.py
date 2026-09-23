"""Content-addressed execution evidence; never trust a passed label alone."""
import hashlib
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def hash_files(paths):
    return {str(Path(p).resolve()): sha256(p) for p in sorted(map(str, paths))}


def reusable(record, fingerprint):
    if record.get('status') != 'passed' or record.get('fingerprint') != fingerprint:
        return False
    artifacts = record.get('artifacts', {})
    if not artifacts:
        return False
    try:
        return all(sha256(p) == digest for p, digest in artifacts.items())
    except OSError:
        return False
