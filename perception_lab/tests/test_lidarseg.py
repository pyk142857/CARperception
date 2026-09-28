import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
try:
    from lidarseg_utils import cylinder_features, remap_labels, confusion_matrix
except ImportError:
    cylinder_features=remap_labels=confusion_matrix=None

class LidarSegTests(unittest.TestCase):
    def test_cylinder_coordinates_and_original_order(self):
        self.assertIsNotNone(cylinder_features)
        p=np.array([[1,0,0,.5,2],[0,2,1,.8,3]],dtype=np.float32)
        grid,fea=cylinder_features(p,[3,3,3],[0,-np.pi,-1],[2,np.pi,1])
        np.testing.assert_array_equal(grid,[[1,1,1],[2,1,2]])
        np.testing.assert_allclose(fea[:,3:6],[[1,0,0],[2,np.pi/2,1]],atol=1e-6)
        np.testing.assert_allclose(fea[:,-1],[.5,.8])
        self.assertEqual(fea.shape,(2,9))
    def test_mapping_uses_semantic_classes_not_detection_ids(self):
        self.assertIsNotNone(remap_labels)
        np.testing.assert_equal(remap_labels(np.array([17,24,30,0]),{17:4,24:11,30:16,0:0}),[4,11,16,0])
        with self.assertRaises(ValueError):remap_labels(np.array([33]),{0:0})
    def test_confusion_ignores_gt_void_but_counts_pred_void_as_error(self):
        self.assertIsNotNone(confusion_matrix)
        h=confusion_matrix(np.array([0,4,4,11]),np.array([4,4,0,4]))
        self.assertEqual(h.sum(),3)
        self.assertEqual(h[4,0],1)
        self.assertEqual(h[11,4],1)

if __name__=='__main__':unittest.main()
