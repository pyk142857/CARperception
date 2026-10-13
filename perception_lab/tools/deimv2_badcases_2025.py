"""Independent COCO IoU=.5, score>=.25 failure diagnostic (not official AP)."""
import json,collections
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
def diagnostic_eval(gt, rows, image_ids):
 if not image_ids or len(image_ids)!=len(set(image_ids)) or not set(image_ids)<=set(gt.imgs):
  raise ValueError('Expected unique evaluated image IDs present in annotations')
 selected=set(image_ids)
 rows=[r for r in rows if r['image_id'] in selected and r['score']>=.25]
 if rows:dt=gt.loadRes(rows)
 else:
  dt=COCO();dt.dataset={'images':list(gt.imgs.values()),'categories':list(gt.cats.values()),'annotations':[]};dt.createIndex()
 ev=COCOeval(gt,dt,'bbox');ev.params.imgIds=list(image_ids);ev.params.iouThrs=np.array([.5]);ev.params.areaRng=[[0,1e10]];ev.params.areaRngLbl=['all'];ev.params.maxDets=[100];ev.evaluate()
 return dt,ev

def main():
 out=ROOT/'reports/perception_2025_20261013/deimv2/badcases';out.mkdir(exist_ok=True)
 gt=COCO(str(ROOT/'data/coco/annotations/instances_val2017.json'))
 rows=json.loads((ROOT/'runs/perception_2025/deimv2/coco_predictions.json').read_text())
 image_ids=json.loads((out.parent/'image_ids.json').read_text())
 dt,ev=diagnostic_eval(gt,rows,image_ids)
 images=collections.defaultdict(lambda:dict(fp=[],fn=[],tp=0))
 for x in ev.evalImgs:
  if x is None:continue
  r=images[int(x['image_id'])]
  r['fp'] += [int(i) for i,m,ignore in zip(x['dtIds'],x['dtMatches'][0],x['dtIgnore'][0]) if not m and not ignore]
  r['fn'] += [int(i) for i,m,ignore in zip(x['gtIds'],x['gtMatches'][0],x['gtIgnore']) if not m and not ignore]
  r['tp'] += sum(bool(m) and not bool(ig) for m,ig in zip(x['dtMatches'][0],x['dtIgnore'][0]))
 totals={k:sum(len(x[k]) if k!='tp' else x[k] for x in images.values()) for k in ['fp','fn','tp']}
 ranked=sorted(images,key=lambda i:len(images[i]['fp'])+len(images[i]['fn']),reverse=True)
 cases=[];paths={x['id']:x['file'] for x in json.loads((ROOT/'reports/perception_2025_20261013/inputs/coco_files.json').read_text())}
 for i in ranked[:3]:
  item=gt.imgs[i];im=Image.open(paths[i]).convert('RGB');draw=ImageDraw.Draw(im)
  for key,db,color in [('fp',dt.anns,'red'),('fn',gt.anns,'lime')]:
   for ident in images[i][key]:
    a=db[ident];x,y,w,h=a['bbox'];draw.rectangle((x,y,x+w,y+h),outline=color,width=2)
    draw.text((max(0,x),max(0,y)),key.upper()+' '+gt.cats[a['category_id']]['name'],fill=color,stroke_width=1,stroke_fill='black')
  name=f'coco_{i:012d}.jpg';im.save(out/name,quality=88);cases.append(dict(image_id=i,file=name,**images[i]))
 summary=dict(protocol='COCOeval IoU=.5 score>=.25 maxDets100 per category, all-area; crowd/ignore handled by COCOeval; diagnostic only, not official AP',source='new_measured native predictions',images=len(image_ids),totals=totals,cases=cases)
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (out/'per_image.json').write_text(json.dumps([dict(image_id=i,fp=len(x['fp']),fn=len(x['fn']),tp=x['tp']) for i,x in images.items()],indent=2)+'\n')
 print(totals)
if __name__=='__main__':main()
