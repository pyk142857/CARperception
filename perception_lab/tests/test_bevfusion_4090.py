"""Asynchronous sensor geometry and native convention regression tests."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from bevfusion_4090 import packet_input, CAMERAS


def transform(x, angle=0):
    c, s = np.cos(angle), np.sin(angle)
    out = np.eye(4)
    out[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    out[:3, 3] = x
    return out.tolist()


class PacketGeometryTest(unittest.TestCase):
    def test_asynchronous_camera_and_sweep_reference(self):
        sensor = dict(path='file', timestamp_us=1000000, K=np.eye(3).tolist(),
                      T_sensor_to_ego=transform([1, 0, 2], .3), T_ego_to_global=transform([10, 4, 0], .1))
        camera = dict(sensor, T_sensor_to_ego=transform([0, 1, 1], -.1), T_ego_to_global=transform([11, 4, 0], .2))
        packet = dict(sample_token='sample', sensors={name: camera for name in CAMERAS}, lidar_sweeps=[camera])
        packet['sensors']['LIDAR_TOP'] = sensor
        result = packet_input(packet)
        world_lidar = np.asarray(sensor['T_ego_to_global']) @ sensor['T_sensor_to_ego']
        world_camera = np.asarray(camera['T_ego_to_global']) @ camera['T_sensor_to_ego']
        np.testing.assert_allclose(world_lidar @ result['camera2lidar'][0], world_camera, atol=1e-6)
        np.testing.assert_allclose(result['lidar2camera'][0] @ result['camera2lidar'][0], np.eye(4), atol=1e-6)
        sweep = result['sweeps'][0]
        np.testing.assert_allclose(sweep['sensor2lidar_rotation'], result['camera2lidar'][0][:3, :3], atol=1e-6)
        np.testing.assert_allclose(sweep['sensor2lidar_translation'], result['camera2lidar'][0][:3, 3], atol=1e-6)
        self.assertNotIn('ann_info', result)
        self.assertEqual(len(result['image_paths']), 6)

    def test_sweep_contract_uses_nine_most_recent(self):
        sensor = dict(path='file', timestamp_us=1000000, K=np.eye(3).tolist(),
                      T_sensor_to_ego=transform([0, 0, 0]), T_ego_to_global=transform([0, 0, 0]))
        packet = dict(sample_token='sample', sensors={name: sensor for name in CAMERAS + ['LIDAR_TOP']},
                      lidar_sweeps=[dict(sensor, timestamp_us=1000000-i) for i in range(12)])
        result = packet_input(packet)
        self.assertEqual([s['timestamp'] for s in result['sweeps']], list(range(1000000, 999991, -1)))


if __name__ == '__main__':
    unittest.main()
