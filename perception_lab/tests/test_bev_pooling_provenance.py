"""Do not compare fresh predictions against stale evaluation artifacts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from bevfusion_4090 import digest
from bev_pooling_full_model import verify_inference_and_evaluation


class ProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.run=Path(self.temp.name)/'run';self.run.mkdir()
        self.eval=Path(self.temp.name)/'eval';(self.eval/'metrics').mkdir(parents=True)
        (self.run/'predictions.json').write_text('[{"sample_token":"one","boxes3d":[]}]')
        (self.run/'performance.json').write_text('{"frames":1}')
        self.manifest=dict(status='passed',outputs_sha256={name:digest(self.run/name) for name in ['predictions.json','performance.json']})
        (self.run/'manifest.json').write_text(json.dumps(self.manifest))
        (self.eval/'official_predictions.json').write_text('{"results":{"one":[]}}')
        self.audit=dict(input_sha256=digest(self.run/'predictions.json'),output_sha256=digest(self.eval/'official_predictions.json'))
        self.write_audit()
        for name in ['metrics_summary.json','metrics_details.json']:(self.eval/'metrics'/name).write_text('{}')

    def write_audit(self):
        (self.eval/'export_audit.json').write_text(json.dumps(self.audit))

    def test_matching_artifacts_are_accepted(self):
        self.assertEqual(verify_inference_and_evaluation(self.run,self.manifest,self.eval)['predictions_sha256'],self.audit['input_sha256'])

    def test_changed_predictions_rejected_even_when_parseable(self):
        (self.run/'predictions.json').write_text('[{"sample_token":"two","boxes3d":[]}]')
        with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):
            verify_inference_and_evaluation(self.run,self.manifest,self.eval)

    def test_stale_evaluation_rejected(self):
        self.audit['input_sha256']='0'*64;self.write_audit()
        with self.assertRaisesRegex(ValueError,'different predictions'):
            verify_inference_and_evaluation(self.run,self.manifest,self.eval)

    def test_changed_performance_metadata_also_rejected(self):
        (self.run/'performance.json').write_text('{"frames":2}')
        with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):
            verify_inference_and_evaluation(self.run,self.manifest,self.eval)


if __name__=='__main__':unittest.main()
