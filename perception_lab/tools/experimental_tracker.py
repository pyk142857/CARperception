"""Experimental fixed-detection tracker. No GT input; baseline is author-compatible."""
import copy
import numpy as np
from scipy.optimize import linear_sum_assignment

NAMES=['bicycle','bus','car','motorcycle','pedestrian','trailer','truck']
GATES=dict(car=4,truck=4,bus=5.5,trailer=3,pedestrian=1,motorcycle=13,bicycle=3)

def assign(cost,hungarian=True):
    if not cost.shape[0] or not cost.shape[1]:return np.empty((0,2),dtype=int)
    if hungarian:
        valid=cost<1e16
        # Finite penalty guarantees maximum valid cardinality without 1e18 precision loss.
        penalty=(min(cost.shape)+1)*(float(cost[valid].max())+1) if valid.any() else 1.
        a,b=linear_sum_assignment(np.where(valid,cost,penalty))
        return np.array([(i,j) for i,j in zip(a,b) if cost[i,j]<1e16],dtype=int).reshape(-1,2)
    cost=cost.copy();pairs=[]
    for i in range(len(cost)):
        j=cost[i].argmin()
        if cost[i,j]<1e16:pairs.append((i,j));cost[:,j]=1e18
    return np.array(pairs,dtype=int).reshape(-1,2)

class ExperimentalTracker:
    def __init__(self,hungarian=True,max_age=3,motion='backproject',rich=False,two_stage=False,birth_score=.5):
        self.hungarian=hungarian;self.max_age=max_age;self.motion=motion
        self.rich=rich;self.two_stage=two_stage;self.birth_score=birth_score;self.reset()
    def reset(self):
        self.tracks=[];self.states={};self.id_count=0;self.diagnostics=[]
    def predict(self,dt):
        F=np.eye(4);F[0,2]=F[1,3]=dt
        G=np.array([[dt*dt/2,0],[0,dt*dt/2],[dt,0],[0,dt]])
        for ident,(x,P) in list(self.states.items()):self.states[ident]=(F@x,F@P@F.T+4*G@G.T)
    def update(self,track,new=False):
        ident=track['tracking_id'];z=np.array(track['translation'][:2])
        if new:self.states[ident]=(np.r_[z,track['velocity'][:2]],np.diag([.25,.25,9,9]));return
        x,P=self.states[ident];H=np.eye(2,4);R=np.eye(2)*.25
        gain=np.linalg.solve(H@P@H.T+R,(P@H.T).T).T
        x=x+gain@(z-H@x);A=np.eye(4)-gain@H
        self.states[ident]=(x,A@P@A.T+gain@R@gain.T)
    def step_centertrack(self,results,time_lag):
        results=[copy.deepcopy(d) for d in results if d['detection_name'] in NAMES]
        if self.motion=='backproject' and not results:
            self.tracks=[];self.states={};return [] # mirror author baseline on empty input
        for d in results:
            d['ct']=np.array(d['translation'][:2]);d['tracking']=-np.array(d['velocity'][:2])*time_lag
            d['label_preds']=NAMES.index(d['detection_name'])
        if self.motion=='kalman':self.predict(time_lag)
        N,M=len(results),len(self.tracks)
        ds=np.array([d['ct']+d['tracking'].astype(np.float32) if self.motion=='backproject' else d['ct'] for d in results],np.float32).reshape(-1,2)
        ts=np.array([t['ct'] if self.motion=='backproject' else self.states[t['tracking_id']][0][:2] for t in self.tracks],np.float32).reshape(-1,2)
        dist=np.sqrt(((ds[:,None,:]-ts[None,:,:])**2).sum(axis=2))
        gates=np.array([GATES[d['detection_name']] for d in results],np.float32)
        same=np.array([[d['detection_name']==t['detection_name'] for t in self.tracks] for d in results],bool).reshape(N,M)
        valid=same & (dist<=gates[:,None]);cost=dist+(~valid)*1e18
        if self.rich and N and M:
            for i,d in enumerate(results):
                for j,t in enumerate(self.tracks):
                    if not valid[i,j]:continue
                    size=np.mean(np.abs(np.log(np.maximum(d['size'],1e-3)/np.maximum(t['size'],1e-3))))
                    v=np.array(d['velocity'][:2]);w=np.array(t['velocity'][:2]) if self.motion=='backproject' else self.states[t['tracking_id']][0][2:]
                    direction=0.
                    if min(np.linalg.norm(v),np.linalg.norm(w))>=.5:direction=1-np.clip(v@w/(np.linalg.norm(v)*np.linalg.norm(w)),-1,1)
                    if valid[i,j]:cost[i,j]=dist[i,j]/gates[i]+.25*size+.15*direction
        if self.two_stage:
            pairs=[];remaining=list(range(M))
            for indices in [[i for i,d in enumerate(results) if d['detection_score']>=.5],[i for i,d in enumerate(results) if d['detection_score']<.5]]:
                local=assign(cost[np.ix_(indices,remaining)],self.hungarian)
                pairs.extend((indices[i],remaining[j]) for i,j in local)
                used={remaining[j] for i,j in local};remaining=[j for j in remaining if j not in used]
            matches=np.array(pairs,dtype=int).reshape(-1,2)
        else:matches=assign(cost,self.hungarian)
        used_d={i for i,j in matches};used_t={j for i,j in matches};ret=[];log=[]
        for i,j in matches:
            d=results[i];old=self.tracks[j];d.update(tracking_id=old['tracking_id'],age=1,active=old['active']+1);ret.append(d)
            if self.motion=='kalman':self.update(d)
            log.append(dict(detection_index=int(i),tracking_id=d['tracking_id'],action='matched',distance_m=float(dist[i,j]),score=d['detection_score']))
        for i,d in enumerate(results):
            if i in used_d:continue
            eligible=[j for j in range(M) if valid[i,j]]
            reason='competition' if eligible else 'gate_rejected' if same[i].any() else 'no_same_class_track'
            nearest=float(dist[i,same[i]].min()) if same[i].any() else None
            suppressed=self.two_stage and d['detection_score']<self.birth_score
            if not suppressed:
                self.id_count+=1;d.update(tracking_id=self.id_count,age=1,active=1);ret.append(d)
                if self.motion=='kalman':self.update(d,new=True)
            log.append(dict(detection_index=i,tracking_id=None if suppressed else d['tracking_id'],action='suppressed_birth' if suppressed else 'new',reason=reason,nearest_same_class_m=nearest,score=d['detection_score']))
        for j,t in enumerate(self.tracks):
            if j not in used_t and t['age']<self.max_age:
                t['age']+=1;t['active']=0
                t['ct']=t['ct']-t['tracking'] if self.motion=='backproject' else self.states[t['tracking_id']][0][:2].copy()
                ret.append(t)
        self.tracks=ret;alive={t['tracking_id'] for t in ret};self.states={i:s for i,s in self.states.items() if i in alive};self.diagnostics=log
        return ret
