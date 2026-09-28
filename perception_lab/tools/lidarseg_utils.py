"""Point-order-preserving Cylinder3D preprocessing and nuScenes label helpers."""
import numpy as np

NAMES = ['ignore','barrier','bicycle','bus','car','construction_vehicle','motorcycle',
         'pedestrian','traffic_cone','trailer','truck','driveable_surface','other_flat',
         'sidewalk','terrain','manmade','vegetation']
# Consistent project palette; indexed by the nuScenes 16-class training ID (+0).
PALETTE = np.array([[70,70,70],[112,128,144],[220,20,60],[255,127,80],[255,158,0],
 [233,150,70],[255,61,99],[0,0,230],[47,79,79],[255,140,0],[255,99,71],
 [0,207,191],[175,0,75],[75,0,75],[112,180,60],[255,240,150],[0,175,0]],dtype=np.uint8)


def cylinder_features(points, grid_size, lower, upper):
    xyz=np.asarray(points)[:,:3]
    polar=np.stack([np.sqrt(xyz[:,0]**2+xyz[:,1]**2),np.arctan2(xyz[:,1],xyz[:,0]),xyz[:,2]],axis=1)
    lower,upper=np.asarray(lower),np.asarray(upper)
    interval=(upper-lower)/(np.asarray(grid_size)-1)
    grid=np.floor((np.clip(polar,lower,upper)-lower)/interval).astype(np.int64)
    centers=(grid.astype(np.float32)+.5)*interval+lower
    features=np.concatenate([polar-centers,polar,xyz[:,:2],points[:,3:4]],axis=1)
    return grid,features.astype(np.float32)


def remap_labels(raw, mapping):
    raw=np.asarray(raw)
    if any(int(x) not in mapping for x in np.unique(raw)):
        raise ValueError('Unknown raw lidarseg category')
    lut=np.zeros(max(mapping)+1,dtype=np.uint8)
    for key,value in mapping.items():lut[key]=value
    return lut[raw]


def confusion_matrix(gt, pred):
    gt,pred=np.asarray(gt),np.asarray(pred)
    if gt.shape!=pred.shape or ((gt<0)|(gt>16)|(pred<0)|(pred>16)).any():
        raise ValueError('Invalid point labels or alignment')
    keep=gt!=0
    return np.bincount(17*gt[keep].astype(np.int64)+pred[keep],minlength=17*17).reshape(17,17)
