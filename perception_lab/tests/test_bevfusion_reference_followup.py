import importlib.util
import pathlib
import unittest
import numpy as np

PATH = pathlib.Path(__file__).parents[1] / 'tools/bevfusion_reference_followup.py'

class FollowupTest(unittest.TestCase):
    def load(self):
        self.assertTrue(PATH.exists(), 'reference followup implementation missing')
        spec = importlib.util.spec_from_file_location('followup', PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_identity_alignment_survives_candidate_exchange(self):
        m = self.load()
        result = m.source_comparison(np.array([12, 3, 40]), np.array([3, 12, 90]),
            np.array([[1., 2.], [4., 5.], [7., 8.]]),
            np.array([[4.1, 5.], [1., 2.], [7., 8.]]))
        self.assertEqual(result['common_sources'], 2)
        self.assertEqual(result['removed_sources'], [40])
        self.assertEqual(result['added_sources'], [90])
        self.assertAlmostEqual(result['aligned_max_abs'], .1)

    def test_nonfinite_and_near_zero_are_not_false_passes(self):
        m = self.load()
        r = m.tensor_comparison(np.array([0., 2., np.nan]), np.array([.06, 2.09, np.nan]))
        self.assertEqual(r['nonfinite_elements'], 1)
        self.assertFalse(r['passed'])
        self.assertEqual(r['exceeded_elements'], 1)
        self.assertEqual(r['reference_abs_le_001_elements'], 1)

    def test_integer_identity_change_never_passes_float_tolerance(self):
        m = self.load()
        r = m.tensor_comparison(np.array([1000], dtype=np.int64), np.array([1001], dtype=np.int64))
        self.assertFalse(r['passed'])
        self.assertEqual(r['exceeded_elements'], 1)
        self.assertEqual(r['tolerance'], 'integer identity: exact equality')

    def test_pooling_skips_invalid_prefix_and_uses_exclusive_end(self):
        m = self.load()
        valid, destination = m.pooling_contributions(
            np.array([99, 98, 7, 8, 9]), np.array([[2, 4, 10], [4, 5, 11]]))
        np.testing.assert_array_equal(valid, [7, 8, 9])
        np.testing.assert_array_equal(destination, [10, 10, 11])

    def test_selection_keeps_first_priority_and_high_score_only(self):
        m = self.load()
        packets = [{'sample_token': x} for x in ['first', m.PRIORITY_TOKEN, 'high', 'low']]
        cases = [{'sample_token': 'high', 'reference_score': .6, 'candidate_score': .7},
                 {'sample_token': 'low', 'reference_score': .1, 'candidate_score': .8}]
        self.assertEqual([x['sample_token'] for x in m.select_packets(packets, cases)],
                         ['first', m.PRIORITY_TOKEN, 'high'])

if __name__ == '__main__':
    unittest.main()
