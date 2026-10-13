"""Supplementary score>=.25 ONNX geometry diagnostic, no GPU or GT input."""
import json,gzip,collections
from pathlib import Path
from deimv2_trt_2025 import geometry
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'reports/perception_2025_20261013/deimv2/onnx';maps=[]
 for filename in ['coco_predictions.json','onnx_predictions.json']:
  d=collections.defaultdict(list)
  for x in json.loads((ROOT/'runs/perception_2025/deimv2'/filename).read_text()):
   if x['score']>=.25:d[x['image_id']].append(x)
  maps.append(d)
 ids=json.loads((out/'image_ids.json').read_text());rows=[dict(image_id=i,classes=geometry(maps[0][i],maps[1][i])) for i in ids]
 with gzip.open(out/'geometry_score025.json.gz','wt') as f:json.dump(rows,f)
 groups=[x for r in rows for x in r['classes']]
 summary=dict(images=len(ids),display_threshold=.25,matched=sum(x['matched'] for x in groups),unmatched=sum(x['unmatched'] for x in groups),max_box_component_px=max((x.get('max_box_component_px',0) for x in groups),default=0),max_score_difference=max((x.get('max_score_difference',0) for x in groups),default=0),caveat='Class-constrained nearest-box assignment is diagnostic, not proposal identity; large displacement cannot be attributed to the same object without gated matching.',evaluation='All300 predictions retained in official AP; threshold only this supplementary diagnostic.')
 (out/'geometry_score025_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
