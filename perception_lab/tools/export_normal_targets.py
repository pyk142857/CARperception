"""Reproduce fixed-threshold matching to identify normal overlay targets."""
import json
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def normal_indices(pred,matched,cases):
    bad={r['matched_prediction_index'] for r in cases if r['kind']=='center_error_over_1m'}
    bad_ids={r['tracking_id'] for r in cases if r['kind'] in {'id_switch','gap_id_change'}}
    return [pred[p]['prediction_index'] for _,p,_ in matched
            if pred[p]['prediction_index'] not in bad and pred[p].get('tracking_id') not in bad_ids]

def main():
    from evaluate_mini import load_frames,evaluate,measured_results
    from failure_overlay import load_cases
    from evidence import sha256
    packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
    status=json.loads((ROOT/'results/status.json').read_text())
    det,dp=measured_results(status,'M05','mini_scene',packets)
    tracks,tp=measured_results(status,'M11','mini_tracking_M05',packets)
    by_frame,summary=load_cases(ROOT/'reports/mini_evaluation',packets,.25)
    frames,_,_=load_frames(packets,det,tracks)
    totals,_,_,cases,details=evaluate(frames,packets,summary['score'],summary['distance_gate_m'])
    for i,c in enumerate(cases):c['case_id']=f'case_{i:05d}'
    original=json.loads((ROOT/'reports/mini_evaluation/cases.json').read_text())
    assert cases==original,'Recomputed diagnostic cases differ'
    assert totals==summary['totals'],'Recomputed totals differ'
    rows=[];counts=defaultdict(int)
    for i in range(len(packets)):
        row={}
        for module in ['detection','tracking']:
            _,pred,matched,_,_=details[(module,i)]
            indices=normal_indices(pred,matched,[c for c in by_frame[i] if c['module']==module])
            source=det[i]['boxes3d'] if module=='detection' else tracks[i]['tracks3d']
            row[module]=[dict(source[n],prediction_index=n) for n in indices]
            counts[module]+=len(indices)
        rows.append(row)
    out=ROOT/'outputs/rerun/normal_targets.json';out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(cases_sha256=sha256(ROOT/'reports/mini_evaluation/cases.json'),
        source_sha256={str(p):sha256(p) for p in [dp,tp]},counts=dict(counts),frames=rows)))
    print(dict(counts))
if __name__=='__main__':main()
