"""Export version-bound scores for every case without changing recording case hashes."""
import json
from analyze_failure_confidence import ROOT,case_confidence
from evaluate_mini import measured_results
from failure_overlay import load_cases
from evidence import sha256

def main():
    packets=json.loads((ROOT/'manifests/frame_packets.json').read_text())
    status=json.loads((ROOT/'results/status.json').read_text())
    load_cases(ROOT/'reports/mini_evaluation',packets,.25)
    det,dp=measured_results(status,'M05','mini_scene',packets)
    tracks,tp=measured_results(status,'M11','mini_tracking_M05',packets)
    source=ROOT/'reports/mini_evaluation/cases.json'
    cases=json.loads(source.read_text())
    scores={c['case_id']:case_confidence(c,det[c['frame']]['boxes3d'] if c['module']=='detection'
                                      else tracks[c['frame']]['tracks3d']) for c in cases}
    out=ROOT/'reports/confidence_filter';out.mkdir(exist_ok=True)
    (out/'scores.json').write_text(json.dumps(dict(cases_sha256=sha256(source),scores=scores,
        prediction_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [dp,tp]},
        protocol='FN=null; FP=original prediction; localization=matched prediction; identity=current ID prediction'),indent=2)+'\n')
    print(f'Exported {len(scores)} scores; null={sum(s is None for s in scores.values())}')

if __name__=='__main__':main()
