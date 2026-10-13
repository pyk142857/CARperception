"""Unmodified pycocotools COCOeval in a process that never imports DEIM's monkeypatch."""
import argparse,hashlib,json
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.cocoeval as implementation

def main():
 p=argparse.ArgumentParser();p.add_argument('--annotations',required=True);p.add_argument('--predictions',required=True);p.add_argument('--image-ids',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 assert COCOeval.__module__=='pycocotools.cocoeval',COCOeval.__module__
 ids=json.loads(Path(a.image_ids).read_text());assert len(ids)==len(set(ids))
 gt=COCO(a.annotations);ev=COCOeval(gt,gt.loadRes(a.predictions),'bbox');ev.params.imgIds=ids;ev.evaluate();ev.accumulate();ev.summarize()
 names=['AP','AP50','AP75','APs','APm','APl','AR1','AR10','AR100','ARs','ARm','ARl'];metrics=dict(zip(names,map(float,ev.stats)));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n');(out/'protocol.json').write_text(json.dumps(dict(evaluator=COCOeval.__module__,implementation_path=implementation.__file__,implementation_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),images=len(ids),prediction_sha256=hashlib.sha256(Path(a.predictions).read_bytes()).hexdigest(),max_dets=ev.params.maxDets,source='new_measured'),indent=2)+'\n')
if __name__=='__main__':main()
