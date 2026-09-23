"""Asynchronous sensor transforms. Poses are column-vector rigid transforms."""
import numpy as np


def sensor_to_sensor(source_ego_to_global, source_sensor_to_ego,
                     target_ego_to_global, target_sensor_to_ego):
    return (np.linalg.inv(target_sensor_to_ego) @ np.linalg.inv(target_ego_to_global)
            @ source_ego_to_global @ source_sensor_to_ego)


def transform_points(matrix, xyz):
    return (matrix @ np.vstack((xyz, np.ones((1, xyz.shape[1])))))[:3]


def project_depth(xyz_camera, intrinsic, height, width):
    z = xyz_camera[2]
    valid = np.isfinite(xyz_camera).all(axis=0) & (z > .1) & (z <= 80)
    xyz = xyz_camera[:, valid]
    pixels = intrinsic @ xyz
    pixels = pixels[:2] / pixels[2:3]
    good = (pixels[0] >= 0) & (pixels[0] < width) & (pixels[1] >= 0) & (pixels[1] < height)
    pixels, z = pixels[:, good], xyz[2, good]
    depth = np.full((height, width), np.inf, dtype=np.float32)
    uv = np.floor(pixels).astype(np.int64)
    np.minimum.at(depth, (uv[1], uv[0]), z)
    mask = np.isfinite(depth)
    depth[~mask] = 0
    return depth, mask, pixels, z
