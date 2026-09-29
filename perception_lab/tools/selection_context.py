"""Camera placement for selected targets; never crops or transforms pointcloud data."""
import numpy as np

def camera_eye(item):
    target=np.asarray(item['center'],dtype=float)
    size=np.asarray(item['size'],dtype=float)
    if target.shape!=(3,) or size.shape!=(3,) or not np.isfinite(target).all() or not np.isfinite(size).all() or (size<=0).any():
        raise ValueError('Invalid target geometry')
    direction=np.array([-.7,-1.,.65]);direction/=np.linalg.norm(direction)
    distance=max(6.,2.5*float(np.linalg.norm(size)))
    return (target+distance*direction).tolist(),target.tolist()
