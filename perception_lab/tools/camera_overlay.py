"""Project reference-ego polylines onto calibrated camera images with clipping."""
import numpy as np


def project_segments(points, camera_to_ego, intrinsic, width, height, near=.1):
    """Return visible pixel segments; never connect across invisible intervals.

    camera_to_ego must include camera capture-time motion compensation. This is
    geometric projection only, without object occlusion or road-height inference.
    """
    points=np.asarray(points,dtype=float)
    inverse=np.linalg.inv(camera_to_ego)
    camera=points @ inverse[:3,:3].T + inverse[:3,3]
    result=[]
    for a,b in zip(camera[:-1],camera[1:]):
        a,b=a.copy(),b.copy()
        if not np.isfinite([a,b]).all() or max(a[2],b[2])<near:
            continue
        if a[2]<near:
            a=a+(b-a)*((near-a[2])/(b[2]-a[2]))
        if b[2]<near:
            b=b+(a-b)*((near-b[2])/(a[2]-b[2]))
        pixel=np.asarray([a,b]) @ np.asarray(intrinsic).T
        pixel=pixel[:,:2]/pixel[:,2:3]
        start,delta=pixel[0],pixel[1]-pixel[0]
        low,high=0.,1.
        for axis,maximum in enumerate([width-1,height-1]):
            if abs(delta[axis])<1e-12:
                if not 0<=start[axis]<=maximum:
                    high=-1.;break
            else:
                limits=sorted([(0-start[axis])/delta[axis],(maximum-start[axis])/delta[axis]])
                low=max(low,limits[0]);high=min(high,limits[1])
        if low<high:
            result.append(np.clip([start+low*delta,start+high*delta],[0,0],[width-1,height-1]).tolist())
    return result
