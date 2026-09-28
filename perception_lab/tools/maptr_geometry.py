"""Geometry and validation shared by MapTR inference and Rerun export."""
import numpy as np

CLASSES = ('divider', 'ped_crossing', 'boundary')
# Camera embeddings are trained in the official converter's order.
CAMERAS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
           'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def lidar_to_image(lidar, camera):
    lidar_world = np.asarray(lidar['T_ego_to_global']) @ np.asarray(lidar['T_sensor_to_ego'])
    camera_world = np.asarray(camera['T_ego_to_global']) @ np.asarray(camera['T_sensor_to_ego'])
    intrinsic = np.eye(4)
    intrinsic[:3, :3] = camera['K']
    return intrinsic @ np.linalg.inv(camera_world) @ lidar_world


def vectors_to_ego(points_xy, lidar_to_ego, ground_z=0.):
    """MapTR predicts XY only. Apply calibrated XY, explicitly assign display Z.

    No metric height is predicted; the constant ego ground plane is approximate.
    """
    points = np.asarray(points_xy, dtype=float)
    if points.ndim != 3 or points.shape[-1] != 2 or not np.isfinite(points).all():
        raise ValueError('Expected finite [vectors, points, 2] predictions')
    xyz = np.concatenate([points, np.zeros((*points.shape[:-1], 1))], axis=-1)
    matrix = np.asarray(lidar_to_ego)
    xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
    xyz[..., 2] = ground_z
    return xyz


def validate_frames(frames, packets):
    if [f['sample_token'] for f in frames] != [p['sample_token'] for p in packets]:
        raise ValueError('MapTR frame token/order mismatch')
    for frame in frames:
        if frame.get('source') != 'measured' or frame.get('status') != 'passed':
            raise ValueError('MapTR requires measured predictions')
        for vector in frame['vectors']:
            points = np.asarray(vector['points_xyz'])
            score = vector['score']
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 2
                    or not np.isfinite(points).all() or not np.isfinite(score)
                    or not 0 <= score <= 1 or vector['class_name'] not in CLASSES):
                raise ValueError('Invalid MapTR vector')
