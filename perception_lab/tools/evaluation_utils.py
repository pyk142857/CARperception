"""Explicit fixed-threshold diagnostic matching (not official benchmark metrics)."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def match_boxes(gt, pred, gate=2., tracking=False, previous=None):
    if not gt or not pred:
        return [], list(range(len(gt))), list(range(len(pred)))
    d=np.linalg.norm(np.array([g['translation'][:2] for g in gt])[:,None,:]-
                     np.array([p['translation'][:2] for p in pred])[None,:,:],axis=2)
    valid=np.array([[g['class_name']==p['class_name'] for p in pred] for g in gt]) & (d<gate)
    matches=[]; used_g=set(); used_p=set()
    def add(g,p):
        matches.append((g,p,float(d[g,p])));used_g.add(g);used_p.add(p)
    if tracking:
        ids={p['tracking_id']:i for i,p in enumerate(pred)}
        for g,box in enumerate(gt):
            p=ids.get((previous or {}).get(box['instance_token']))
            if p is not None and p not in used_p and valid[g,p]:add(g,p)
        gs=[g for g in range(len(gt)) if g not in used_g]
        ps=[p for p in range(len(pred)) if p not in used_p]
        if gs and ps:
            # Large penalty makes maximum valid cardinality precede distance minimization.
            cost=np.where(valid[np.ix_(gs,ps)],d[np.ix_(gs,ps)],(len(gt)+len(pred)+1)*max(gate,1.))
            rows,cols=linear_sum_assignment(cost)
            for a,b in zip(rows,cols):
                g,p=gs[a],ps[b]
                if valid[g,p]:add(g,p)
    else:
        for p in sorted(range(len(pred)),key=lambda i:(-pred[i]['score'],i)):
            choices=[g for g in range(len(gt)) if g not in used_g and valid[g,p]]
            if choices:add(min(choices,key=lambda g:d[g,p]),p)
    return matches,[g for g in range(len(gt)) if g not in used_g],[p for p in range(len(pred)) if p not in used_p]


def identity_events(history, scene, frame, pairs):
    events=[]
    for gt_id,pred_id in pairs:
        key=(scene,gt_id)
        if key in history:
            old_frame,old_id=history[key]
            gap=frame-old_frame-1
            if old_id!=pred_id or gap:
                kind=('gap_id_change' if gap else 'id_switch') if old_id!=pred_id else 'gap_recovery'
                events.append(dict(kind=kind,instance_token=gt_id,previous_tracking_id=old_id,
                                   tracking_id=pred_id,previous_frame=old_frame,missed_frames=gap))
        history[key]=(frame,pred_id)
    return events
