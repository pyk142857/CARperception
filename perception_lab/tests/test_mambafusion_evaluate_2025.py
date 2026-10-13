import sys
from pathlib import Path
import copy
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from mambafusion_evaluate_2025 import validate_global


def fixture():
    return {'results':{'a':[dict(sample_token='a',translation=[1000,2000,2],size=[2,4,1],rotation=[1,0,0,0],velocity=[3,4],detection_name='car',detection_score=.001,attribute_name='vehicle.moving')]}}


def test_global_is_not_transformed_or_score_filtered():
    data=fixture(); before=copy.deepcopy(data)
    assert validate_global(data,['a'])['boxes']==1
    assert data==before


def test_wrong_tokens_rejected():
    with pytest.raises(ValueError,match='token'): validate_global(fixture(),['b'])


def test_missing_receipt_invalidates_prior_success(tmp_path):
    import json
    from mambafusion_evaluate_2025 import evaluate
    status=tmp_path/'status.json'
    status.write_text(json.dumps(dict(status='partial',inference='passed',frames_inferred=120,strict_load='passed',official_evaluation='passed',official_metrics={'old':True})))
    with pytest.raises(FileNotFoundError):evaluate(tmp_path,tmp_path)
    state=json.loads(status.read_text())
    assert state['official_evaluation']=='failed' and state['official_metrics'] is None


@pytest.mark.parametrize('key,value',[('translation',[float('nan'),0,0]),('rotation',[0,0,0,0]),('size',[0,1,1]),('detection_score',1.1),('attribute_name','invented')])
def test_invalid_global_box_rejected(key,value):
    data=fixture();data['results']['a'][0][key]=value
    with pytest.raises(ValueError):validate_global(data,['a'])
