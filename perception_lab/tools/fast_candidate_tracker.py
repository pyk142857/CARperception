"""CPU-optimized S4_greedy_low; preserves reference outputs and diagnostics.

This specializes the measured candidate, not the Kalman/Hungarian experiments.
"""
import copy
import numpy as np
from experimental_tracker import NAMES,GATES

LABELS={name:i for i,name in enumerate(NAMES)}
FIELDS={'translation','size','rotation','velocity','detection_name','detection_score'}
VECTORS={'translation','size','rotation','velocity'}

def clone_detection(d):
    # Fast path for the measured JSON schema. Unknown/nested extensions retain
    # deepcopy semantics rather than sharing caller-owned mutable objects.
    if set(d)!=FIELDS:return copy.deepcopy(d)
    out=d.copy()
    for key in VECTORS:
        value=d[key]
        if type(value) is list and all(type(x) in (int,float) for x in value):out[key]=value.copy()
        elif isinstance(value,np.ndarray) and value.dtype.kind in 'fiu' and value.ndim==1:out[key]=value.copy()
        else:return copy.deepcopy(d)
    return out

class FastCandidateTracker:
    def __init__(self,max_age=3):
        self.max_age=max_age;self.reset()
    def reset(self):
        self.tracks=[];self.states={};self.id_count=0;self.diagnostics=[]
    def step_centertrack(self,results,time_lag):
        results=[clone_detection(d) for d in results if d['detection_name'] in LABELS]
        if not results:
            # Match reference backprojection behavior, including last diagnostics.
            self.tracks=[];self.states={};return []
        N,M=len(results),len(self.tracks)
        centers=np.array([d['translation'][:2] for d in results])
        offsets=-np.array([d['velocity'][:2] for d in results])*time_lag
        labels=np.array([LABELS[d['detection_name']] for d in results])
        for i,d in enumerate(results):
            d['ct']=centers[i].copy();d['tracking']=offsets[i].copy();d['label_preds']=int(labels[i])
        ds=(centers+offsets.astype(np.float32)).astype(np.float32)
        ts=np.array([t['ct'] for t in self.tracks],np.float32).reshape(-1,2)
        dx=ds[:,None,0]-ts[None,:,0];dy=ds[:,None,1]-ts[None,:,1]
        dist=np.sqrt(dx*dx+dy*dy)
        gates=np.array([GATES[d['detection_name']] for d in results],np.float32)
        track_labels=np.array([t['label_preds'] for t in self.tracks])
        same=labels[:,None]==track_labels[None,:]
        valid=same & (dist<=gates[:,None]);cost=dist+(~valid)*1e18
        scores=np.array([d['detection_score'] for d in results])
        # Keep original row order within each stage and original column tie order.
        # Consume columns in one matrix, avoiding stage submatrices and copies.
        order=np.r_[np.flatnonzero(scores>=.5),np.flatnonzero(scores<.5)]
        matches=[]
        if M:
            for i in order:
                j=int(cost[i].argmin())
                if cost[i,j]<1e16:matches.append((int(i),j));cost[:,j]=1e18
        used_d={i for i,j in matches};used_t={j for i,j in matches};ret=[];log=[]
        has_same=same.any(axis=1);has_valid=valid.any(axis=1)
        for i,j in matches:
            d=results[i];old=self.tracks[j];d.update(tracking_id=old['tracking_id'],age=1,active=old['active']+1);ret.append(d)
            log.append(dict(detection_index=i,tracking_id=d['tracking_id'],action='matched',distance_m=float(dist[i,j]),score=d['detection_score']))
        for i,d in enumerate(results):
            if i in used_d:continue
            reason='competition' if has_valid[i] else 'gate_rejected' if has_same[i] else 'no_same_class_track'
            nearest=float(dist[i,same[i]].min()) if has_same[i] else None
            suppressed=d['detection_score']<.25
            if not suppressed:
                self.id_count+=1;d.update(tracking_id=self.id_count,age=1,active=1);ret.append(d)
            log.append(dict(detection_index=i,tracking_id=None if suppressed else d['tracking_id'],action='suppressed_birth' if suppressed else 'new',reason=reason,nearest_same_class_m=nearest,score=d['detection_score']))
        for j,t in enumerate(self.tracks):
            if j not in used_t and t['age']<self.max_age:
                t['age']+=1;t['active']=0;t['ct']=t['ct']-t['tracking'];ret.append(t)
        self.tracks=ret;self.diagnostics=log
        return ret
