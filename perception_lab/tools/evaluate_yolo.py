"""Official Ultralytics val followed by explicit official COCOeval evidence."""
import argparse
import json
import os
from pathlib import Path
import torch
from evidence import hash_files, sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');args=p.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    torch.set_num_threads(2)
    from ultralytics import YOLO
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    root=Path(__file__).resolve().parents[1];out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    image_paths=[Path(s) for s in (root/'data/coco/val2017.txt').read_text().splitlines()]
    if len(image_paths)!=5000 or not all(p.is_file() for p in image_paths):raise ValueError('Incomplete COCO val')
    (out/'input_hashes.json').write_text(json.dumps(hash_files(image_paths),indent=2))
    model=YOLO(root/'checkpoints/yolov8s.pt')
    model.val(data=str(root/'configs/coco_val_only.yaml'),split='val',batch=1,imgsz=640,conf=.001,iou=.7,
              max_det=300,device=args.device,workers=2,save_json=True,plots=False,verbose=False,
              project=str(out),name='native',exist_ok=True,half=False)
    pred=out/'native/predictions.json'
    if not pred.is_file():raise FileNotFoundError(pred)
    gt=COCO(str(root/'data/coco/annotations/instances_val2017.json'))
    evaluator=COCOeval(gt,gt.loadRes(str(pred)),'bbox')
    evaluator.params.imgIds=sorted(int(p.stem) for p in image_paths)
    evaluator.evaluate();evaluator.accumulate();evaluator.summarize()
    names=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
    results=[]
    for name,value in zip(names,evaluator.stats):
        results.append({'name':name,'value':float(value),'unit':'fraction','direction':'higher',
            'protocol':'official pycocotools bbox COCO 2017 val','split':'val2017','scope':'benchmark',
            'sample_count':5000,'run_id':out.parts[-3],'source':'measured','raw_evaluator_path':str(out/'command.log')})
    (out/'metrics.json').write_text(json.dumps(results,indent=2))
    precision=evaluator.eval['precision'];per_class={}
    for i,cat in enumerate(evaluator.params.catIds):
        values=precision[:,:,i,0,-1];values=values[values>-1]
        per_class[gt.cats[cat]['name']]=float(values.mean()) if len(values) else None
    (out/'per_class_AP.json').write_text(json.dumps(per_class,indent=2))
    (out/'provenance.json').write_text(json.dumps({'checkpoint_sha256':sha256(root/'checkpoints/yolov8s.pt'),
        'annotation_sha256':sha256(root/'data/coco/annotations/instances_val2017.json'),
        'device':args.device,'source':'measured','scope':'benchmark','sample_count':5000},indent=2))


if __name__=='__main__':main()
