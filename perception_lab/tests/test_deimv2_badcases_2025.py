import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from deimv2_badcases_2025 import COCO,diagnostic_eval


def test_empty_detections_count_only_evaluated_images_as_misses():
    gt=COCO()
    gt.dataset={'images':[{'id':1},{'id':2}], 'categories':[{'id':1,'name':'car'}],
                'annotations':[{'id':i,'image_id':i,'category_id':1,'bbox':[0,0,10,10],
                                'area':100,'iscrowd':0} for i in [1,2]]}
    gt.createIndex()
    _,evaluation=diagnostic_eval(gt,[],[2])
    records=[x for x in evaluation.evalImgs if x is not None]
    assert [x['image_id'] for x in records]==[2]
    assert records[0]['gtMatches'][0].tolist()==[0]
