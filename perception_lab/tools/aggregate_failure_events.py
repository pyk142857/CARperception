"""Deterministic, conservative event aggregation for the mini diagnostic."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
KINDS = {'false_negative','false_positive','center_error_over_1m','id_switch','gap_id_change'}
PARAMS = dict(max_gap_seconds=.75, fp_residual_gate_m=2., max_size_ratio=2.,
              context_seconds=2., continuity='adjacent frames only', version=1)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def filehash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def adjacent(a,b):
    dt=(b['timestamp_us']-a['timestamp_us'])/1e6
    return b['frame']==a['frame']+1 and 0<dt<=PARAMS['max_gap_seconds']

def fp_cost(a,b):
    if not adjacent(a,b):return None
    if a['module']=='tracking' and (not a.get('tracking_id') or a['tracking_id']!=b.get('tracking_id')):return None
    sa=np.asarray(a['size_wlh']);sb=np.asarray(b['size_wlh'])
    if np.max(np.maximum(sa/sb,sb/sa))>PARAMS['max_size_ratio']:return None
    dt=(b['timestamp_us']-a['timestamp_us'])/1e6
    predicted=np.asarray(a['center_global'])+np.asarray(a['velocity_global'])*dt
    residual=float(np.linalg.norm((predicted-np.asarray(b['center_global']))[:2]))
    return residual if residual<=PARAMS['fp_residual_gate_m'] else None

def aggregate(rows):
    buckets=defaultdict(list);seen=set()
    for r in sorted(rows,key=lambda x:(x['frame'],x['case_id'])):
        if r['case_id'] in seen:raise ValueError('Duplicate case_id')
        seen.add(r['case_id'])
        if r['kind'] not in KINDS:continue
        key=(r['scene_token'],r['module'],r['kind'],r['class_name'])
        if r['kind']!='false_positive':
            if not r.get('instance_token'):raise ValueError('Missing GT instance')
            key+=(r['instance_token'],)
        buckets[key].append(r)
    sequences=[]
    for key,items in sorted(buckets.items()):
        if key[2]!='false_positive':
            active=None
            for r in items:
                if active and r['frame']==active[-1]['frame']:raise ValueError('Duplicate GT observation')
                if active and adjacent(active[-1],r):active.append(r)
                else:
                    active=[r];sequences.append(active)
        else:
            frames=defaultdict(list)
            for r in items:frames[r['frame']].append(r)
            active=[]
            for frame,current in sorted(frames.items()):
                costs=np.full((len(active),len(current)),1e9)
                for i,seq in enumerate(active):
                    for j,r in enumerate(current):
                        cost=fp_cost(seq[-1],r)
                        if cost is not None:costs[i,j]=cost
                matched={};new=[]
                if costs.size:
                    ii,jj=linear_sum_assignment(costs)
                    for i,j in zip(ii,jj):
                        if costs[i,j]<1e9:matched[j]=active[i]
                for j,r in enumerate(current):
                    if j in matched:
                        seq=matched[j];seq.append(r)
                    else:seq=[r];sequences.append(seq)
                    new.append(seq)
                active=new
    events=[]
    for seq in sequences:
        first,last=seq[0],seq[-1]
        representative=max(seq,key=lambda r:(r.get('center_error_m') or 0,r.get('score') or 0,-r['frame']))
        ids=[r['case_id'] for r in seq]
        event_id='event_'+digest([PARAMS,first['scene_token'],ids])[:16]
        transitions=[dict(frame=r['frame'],previous_id=r.get('previous_tracking_id'),new_id=r.get('tracking_id'),
                          previous_frame=r.get('previous_frame'),missed_frames=r.get('missed_frames',0))
                     for r in seq if r['kind'] in {'id_switch','gap_id_change'}]
        events.append(dict(event_id=event_id,scene_token=first['scene_token'],
            module=first['module'],kind=first['kind'],class_name=first['class_name'],
            instance_token=first.get('instance_token'),tracking_id=first.get('tracking_id'),
            start_frame=first['frame'],end_frame=last['frame'],
            start_seconds=first['elapsed_seconds'],end_seconds=last['elapsed_seconds'],
            span_seconds=(last['timestamp_us']-first['timestamp_us'])/1e6,
            failure_frame_count=len(seq),case_ids=ids,representative_case_id=representative['case_id'],
            representative_frame=representative['frame'],
            median_distance_m=float(np.median([r['distance_m'] for r in seq])),
            transition_count=len(transitions),transitions=transitions,
            association_method='GT instance' if first['kind']!='false_positive' else
                ('prediction ID + gated motion' if first['module']=='tracking' else 'gated motion Hungarian'),
            association_quality='identity from GT' if first['kind']!='false_positive' else 'heuristic; review required'))
    return sorted(events,key=lambda e:(e['start_frame'],e['module'],e['kind'],e['event_id']))

def enrich(rows,packets,predictions):
    enriched=[]
    for original in rows:
        r=dict(original);packet=packets[r['frame']]
        if packet['sample_token']!=r['sample_token'] or packet['timestamp_us']!=r['timestamp_us']:
            raise ValueError('Case/packet mismatch')
        r['scene_token']=packet['scene_token']
        pose=np.asarray(packet['sensors']['LIDAR_TOP']['T_ego_to_global'])
        r['center_global']=(pose@np.r_[r['center_ego'],1])[:3].tolist()
        if r['kind']=='false_positive':
            pred=predictions[r['module']][r['frame']]
            if pred['sample_token']!=r['sample_token']:raise ValueError('Prediction sample mismatch')
            box=pred['boxes3d' if r['module']=='detection' else 'tracks3d'][r['prediction_index']]
            if box['class_name']!=r['class_name'] or not np.allclose(box['center_xyz'],r['center_ego'],atol=1e-5):
                raise ValueError('Prediction box mismatch')
            r['size_wlh']=box['size_wlh']
            r['velocity_global']=(pose[:3,:3]@np.r_[box.get('velocity_xy',[0,0]),0]).tolist()
            if not np.all(np.isfinite(r['velocity_global'])) or min(r['size_wlh'])<=0:
                raise ValueError('Invalid FP geometry')
        enriched.append(r)
    return enriched

def build_groups(events):
    groups={}
    for e in events:
        distance=e['median_distance_m']
        bucket='0–20m' if distance<20 else ('20–40m' if distance<40 else '40m+')
        key=(e['module'],e['kind'],e['class_name'],bucket)
        gid='group_'+digest(key)[:16];e['group_id']=gid
        g=groups.setdefault(gid,dict(group_id=gid,module=key[0],kind=key[1],class_name=key[2],
            distance_bucket=bucket,event_ids=[],case_count=0,max_span_seconds=0.,scene_tokens=[]))
        g['event_ids'].append(e['event_id']);g['case_count']+=len(e['case_ids'])
        g['max_span_seconds']=max(g['max_span_seconds'],e['span_seconds'])
        if e['scene_token'] not in g['scene_tokens']:g['scene_tokens'].append(e['scene_token'])
    for g in groups.values():g['event_count']=len(g['event_ids'])
    return sorted(groups.values(),key=lambda g:(-g['event_count'],g['group_id']))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=ROOT/'reports/failure_events')
    args=parser.parse_args()
    report=ROOT/'reports/mini_evaluation'
    summary=json.loads((report/'summary.json').read_text())
    # Verify every original evaluation input before attaching extra geometry.
    for path,expected in summary['source_sha256'].items():
        if filehash(path)!=expected:raise ValueError('Changed evaluation input: '+path)
    rows=json.loads((report/'cases.json').read_text())
    packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
    paths={module:next(p for p in summary['source_sha256'] if '/'+tag+'/' in p)
           for module,tag in [('detection','M05'),('tracking','M11')]}
    predictions={m:json.loads(Path(p).read_text()) for m,p in paths.items()}
    events=aggregate(enrich(rows,packets,predictions));groups=build_groups(events)
    case_ids=[cid for e in events for cid in e['case_ids']]
    assert sorted(case_ids)==sorted(r['case_id'] for r in rows if r['kind'] in KINDS)
    frames=[dict(frame=i,scene_token=p['scene_token'],elapsed_seconds=(p['timestamp_us']-packets[0]['timestamp_us'])/1e6)
            for i,p in enumerate(packets)]
    data=dict(schema_version=1,parameters=PARAMS,cases_sha256=filehash(report/'cases.json'),
              frames=frames,events=events,groups=groups)
    data['dataset_id']=digest(data)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'events.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    stats=dict(scope='scene-0061 mini teaching diagnostic',dataset_id=data['dataset_id'],
        original_case_count=len(rows),failure_case_count=len(case_ids),excluded_gap_recovery=len(rows)-len(case_ids),
        event_count=len(events),group_count=len(groups),by_kind=dict(Counter(e['kind'] for e in events)),
        parameters=PARAMS,case_membership='each failure case belongs to exactly one event',
        source_sha256={str(p):filehash(p) for p in [report/'cases.json',report/'summary.json',
           ROOT/'manifests/frame_packets.json',Path(__file__),*map(Path,paths.values())]},
        limitations=['FP association is heuristic, not GT identity','groups describe symptoms, not root causes',
                     'no exposure denominator: counts are not failure rates','single-scene teaching scope'])
    (args.out/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:stats[k] for k in ['failure_case_count','event_count','group_count','by_kind']}))

if __name__=='__main__':main()
