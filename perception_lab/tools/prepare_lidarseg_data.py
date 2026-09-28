"""Download and verify the official mini lidarseg expansion (404 keyframes)."""
import json,shutil,tarfile,urllib.request
from pathlib import Path
from evidence import sha256
ROOT=Path(__file__).resolve().parents[1]
URL='https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2'


def main():
    archive=ROOT/'data/downloads/nuScenes-lidarseg-mini-v1.0.tar.bz2'
    archive.parent.mkdir(parents=True,exist_ok=True)
    if not archive.exists():urllib.request.urlretrieve(URL,archive)
    root=ROOT/'data/nuscenes'
    with tarfile.open(archive) as stream:
        for member in stream.getmembers():
            if not member.isfile():continue
            name=member.name.removeprefix('./')
            if name.startswith('/') or '..' in Path(name).parts:raise ValueError('Unsafe archive path')
            dest=root/name;blob=stream.extractfile(member).read()
            if dest.exists() and dest.read_bytes()!=blob:
                if name!='v1.0-mini/category.json':raise ValueError('Unexpected differing existing file: '+name)
                old={x['token']:x for x in json.loads(dest.read_text())};new={x['token']:x for x in json.loads(blob)}
                if not all(k in new and new[k]['name']==v['name'] for k,v in old.items()):raise ValueError('Category token mismatch')
                backup=ROOT/'data/pre_lidarseg'/name;backup.parent.mkdir(parents=True,exist_ok=True)
                if not backup.exists():shutil.copy2(dest,backup)
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(blob)
    sd={x['token']:x for x in json.loads((root/'v1.0-mini/sample_data.json').read_text())}
    rows=json.loads((root/'v1.0-mini/lidarseg.json').read_text());total=0
    for row in rows:
        count=(root/row['filename']).stat().st_size
        if count!=(root/sd[row['token']]['filename']).stat().st_size//20:raise ValueError('Point-label count mismatch')
        total+=count
    out=ROOT/'outputs/lidarseg';out.mkdir(parents=True,exist_ok=True)
    result=dict(source_url=URL,archive_sha256=sha256(archive),keyframes=len(rows),total_points=total,all_point_counts_match=True)
    (out/'data_audit.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
