"""Tests prevent synthetic points, ignored GT and absent classes inflating scores."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from harpnext_2025 import map_labels, original_predictions, confusion, summarize, checkpoint_net


class OriginalPointEvaluation(unittest.TestCase):
    def test_optimization_publishes_fresh_runner_evidence(self):
        import tempfile
        import json
        from unittest.mock import patch
        import harpnext_2025 as module
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            checkpoint = folder/'checkpoint.pth'
            checkpoint.write_bytes(b'fixture; model execution mocked')
            audit = {'status': 'partial', 'checkpoint_loaded': False, 'adoption': 'not adopted'}
            with patch.object(sys, 'argv', ['harp', '--phase', 'optimize', '--out', str(folder), '--checkpoint', str(checkpoint)]), patch.object(module, 'preflight'), patch.object(module, 'audit_gpu', return_value=audit):
                self.assertEqual(module.main(), 0)
            self.assertEqual(json.loads((folder/'optimization.json').read_text()), audit)

    def test_checkpoint_rejects_missing_mixed_or_empty_state(self):
        for checkpoint in ({'model': {'x': 1}}, {'net': {}}, {'net': {'module.x': 1, 'y': 2}}):
            with self.assertRaises(ValueError):
                checkpoint_net(checkpoint)
        state, removed = checkpoint_net({'net': {'module.a': 1, 'module.b': 2}})
        self.assertEqual(state, {'a': 1, 'b': 2})
        self.assertTrue(removed)
        state, removed = checkpoint_net({'net': {'a': 1}})
        self.assertFalse(removed)

    def test_official_mapping_and_ignore(self):
        self.assertEqual(map_labels(np.array([0, 1, 9, 15, 16, 17, 30, 31])).tolist(),
                         [255, 255, 0, 2, 2, 3, 15, 255])
        with self.assertRaises(ValueError):
            map_labels(np.array([32]))

    def test_explicit_restore_and_synthetic_exclusion(self):
        # Network order: original 2, interpolated, original 0, original 1.
        self.assertEqual(original_predictions(np.array([5, 9, 3, 4]),
                                              np.array([2, -1, 0, 1]), 3).tolist(), [3, 4, 5])
        for indices in ([0, 0, 2], [0, -1, 2], [0, 1, 3]):
            with self.assertRaises(ValueError):
                original_predictions(np.array([1, 2, 3]), np.array(indices), 3)

    def test_metrics_do_not_reward_ignore_or_missing_classes(self):
        hist = confusion(np.array([0, 0, 1, 255]), np.array([0, 1, 1, 3]))
        result = summarize(hist)
        self.assertEqual(result['valid_points'], 3)
        self.assertAlmostEqual(result['accuracy'], 2 / 3)
        self.assertAlmostEqual(result['miou_union_gt_zero'], .5)
        self.assertIsNone(result['classes'][2]['iou'])
        self.assertEqual(result['classes'][0]['display_id'], 1)
        with self.assertRaises(ValueError):
            confusion(np.array([0, 1]), np.array([0]))
        with self.assertRaises(ValueError):
            confusion(np.array([0]), np.array([16]))

if __name__ == '__main__':
    unittest.main()
