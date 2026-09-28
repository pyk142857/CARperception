"""Shared geometry and deterministic identity colors for tracking views."""
import colorsys
import hashlib
import numpy as np
from pyquaternion import Quaternion
from camera_overlay import project_segments

EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def track_color(identity):
    hue = int.from_bytes(hashlib.sha256(str(identity).encode()).digest()[:4], 'big') / 2**32
    return tuple(round(x*255) for x in colorsys.hsv_to_rgb(hue, .72, 1.))


def box_corners(box):
    width, length, height = box['size_wlh']
    signs = np.array([[1,1,-1],[1,-1,-1],[-1,-1,-1],[-1,1,-1],
                      [1,1,1],[1,-1,1],[-1,-1,1],[-1,1,1]])
    local = signs * np.array([length,width,height]) / 2
    return local @ Quaternion(box['rotation_wxyz']).rotation_matrix.T + box['center_xyz']


def bev_outline(box):
    return -box_corners(box)[[0,1,2,3,0]][:,[1,0]]


def camera_box_segments(box, camera_to_ego, intrinsic, width, height):
    corners = box_corners(box)
    return [segment for edge in EDGES for segment in
            project_segments(corners[list(edge)], camera_to_ego, intrinsic, width, height)]
