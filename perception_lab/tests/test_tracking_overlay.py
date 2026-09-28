import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
try:
    from tracking_overlay import box_corners, track_color, camera_box_segments, bev_outline
except ImportError:
    box_corners = None

class TrackingOverlayTests(unittest.TestCase):
    def test_box_dimensions_rotation_and_bev_axes(self):
        self.assertIsNotNone(box_corners)
        b = dict(center_xyz=[10, 2, 1], size_wlh=[2, 4, 2], rotation_wxyz=[1, 0, 0, 0])
        corners = box_corners(b)
        np.testing.assert_allclose(corners.min(0), [8, 1, 0])
        np.testing.assert_allclose(corners.max(0), [12, 3, 2])
        outline = bev_outline(b)
        np.testing.assert_allclose(outline[0], outline[-1])
        np.testing.assert_allclose(outline[:4].mean(0), [-2, -10])
        b['rotation_wxyz'] = [2**-.5, 0, 0, 2**-.5]
        corners = box_corners(b)
        np.testing.assert_allclose(corners.max(0)-corners.min(0), [2, 4, 2])

    def test_camera_box_rejects_behind_and_clips_edges(self):
        self.assertIsNotNone(box_corners)
        b = dict(center_xyz=[0, 0, 10], size_wlh=[2, 4, 2], rotation_wxyz=[1, 0, 0, 0])
        k = [[100, 0, 50], [0, 100, 50], [0, 0, 1]]
        segments = camera_box_segments(b, np.eye(4), k, 100, 100)
        self.assertEqual(len(segments), 12)
        self.assertTrue((np.asarray(segments) >= 0).all())
        self.assertTrue((np.asarray(segments) <= 99).all())
        b['center_xyz'] = [0, 0, -10]
        self.assertEqual(camera_box_segments(b, np.eye(4), k, 100, 100), [])

    def test_id_color_does_not_depend_on_frame_or_order(self):
        self.assertIsNotNone(box_corners)
        colors = {str(i): track_color(str(i)) for i in range(100)}
        for i in reversed(range(100)):
            self.assertEqual(track_color(str(i)), colors[str(i)])
        self.assertGreater(len(set(colors.values())), 90)

if __name__ == '__main__':
    unittest.main()
