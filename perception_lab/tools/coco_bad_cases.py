"""Threshold-defined COCO missed-box diagnostics from this run's predictions."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import cv2

def iou(a,b):
 ax,ay,aw,ah=a;bx,by,bw,bh=b
 inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
 return inter/(aw*ah+bw*bh-inter) if aw*ah+bw*bh>inter else 0

def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[1];out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 gt=json.loads((root/'data/coco/annotations/instances_val2017.json').read_text());pred=json.loads(Path(a.predictions).read_text())
 lookup=defaultdict(list)
 for x in pred:
  if x['score']>=.25:lookup[x['image_id']].append(x)
 images={x['id']:x for x in gt['images']};names={x['id']:x['name'] for x in gt['categories']};cases=[];seen=set()
 for ann in sorted(gt['annotations'],key=lambda x:(x['image_id'],x['id'])):
  if ann.get('iscrowd') or ann['area']<10000 or ann['image_id'] in seen:continue
  detections=lookup[ann['image_id']];same=[x for x in detections if x['category_id']==ann['category_id']]
  overlap=max([iou(ann['bbox'],x['bbox']) for x in same],default=0)
  if overlap>=.3:continue
  im=images[ann['image_id']];image=cv2.imread(str(root/'data/coco/images/val2017'/im['file_name']))
  for x in detections:
   bx,by,bw,bh=map(int,x['bbox']);cv2.rectangle(image,(bx,by),(bx+bw,by+bh),(0,0,255),1)
   cv2.putText(image,f"PRED {names[x['category_id']]} {x['score']:.2f}",(bx,max(12,by)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,255),1)
  x,y,w,h=map(int,ann['bbox'])
  cv2.rectangle(image,(x,y),(x+w,y+h),(255,255,0),3)
  cv2.putText(image,'GT: '+names[ann['category_id']],(x,max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,0),2)
  cv2.imwrite(str(out/f"{ann['image_id']:012d}.jpg"),image)
  cases.append({'image_id':ann['image_id'],'annotation_id':ann['id'],'class':names[ann['category_id']],
   'definition':'No same-class detection with score>=0.25 and IoU>=0.30 for this GT; diagnostic threshold, not a replacement for COCO AP',
   'max_same_class_iou':overlap,'GT_bbox_xywh':ann['bbox'],'source':'measured','prediction_source':str(Path(a.predictions).resolve()),
   'weather':'unknown','occlusion':'unknown','scope':'COCO2017_val_diagnostic','image':str(out/f"{ann['image_id']:012d}.jpg")})
  seen.add(ann['image_id'])
  if len(cases)==8:break
 if len(cases)!=8:raise RuntimeError('Fewer than 8 verified cases; do not fabricate')
 (out/'cases.json').write_text(json.dumps(cases,indent=2));print('Saved',len(cases),'real threshold-defined cases')
if __name__=='__main__':main()
