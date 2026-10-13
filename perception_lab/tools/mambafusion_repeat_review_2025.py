"""Summarize saved native repeat noise; never alter detector predictions."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/perception_2025_20261013/mambafusion'
BASELINE=REPORT/'repeat_diagnostic_baseline'
def main():
    rows=[]
    for split,count in [('scene0061',39),('mini_val',81)]:
        gates=json.loads((BASELINE/(split+'_first_middle_last.json')).read_text())
        if len(gates)!=3 or not all(g.get('repeat_input_identical') for g in gates):
            raise ValueError('Six real frame gates with identical repeated inputs required')
        for index in [0,count//2,count-1]:
            for repeat in range(2):
                path=BASELINE/f'{split}_gate_{index}_repeat_{repeat}.npz';d=np.load(path)
                a,b=d['base_boxes'],d['repeat_boxes'];an,bn=d['base_names'],d['repeat_names']
                dist=np.linalg.norm(a[:,None,:3]-b[None,:,:3],axis=2)
                cost=dist+(an[:,None]!=bn[None,:])*1e5
                i,j=linear_sum_assignment(cost);matched=(an[i]==bn[j])&(dist[i,j]<2)
                high=(d['base_scores'][i]>=.25)|(d['repeat_scores'][j]>=.25)
                chosen=matched&high
                rows.append(dict(split=split,index=index,repeat=repeat,artifact=str(path.relative_to(REPORT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),base_count=len(a),repeat_count=len(b),unmatched_base=int(len(a)-matched.sum()),unmatched_repeat=int(len(b)-matched.sum()),matched_high_count=int(chosen.sum()),matched_high_center_max_m=float(dist[i[chosen],j[chosen]].max()),matched_high_score_max_abs=float(np.abs(d['base_scores'][i[chosen]]-d['repeat_scores'][j[chosen]]).max())))
    result=dict(status='reviewed_native_nondeterminism',scope='6 frames, 2 repeats each, identical numeric input hashes; matching diagnostic only',matching='class-aware Hungarian 3D center, accept distance<2m; high means either score>=0.25; unmatched candidates retained in counts',original_aligned_repeat_budget_passed=False,decision='Proceed with original author native model for mini diagnostic evaluation. Do not claim deterministic predictions or optimization equivalence. No prediction filtering, reordered input, operator substitution or budget relaxation.',possible_sources=['DynamicVoxelVFE scatter_mean GPU atomic accumulation','LION scatter_add/mean and diffusion topk','TransFusion zero-masked heatmap argsort/top200 proposal selection'],causal_localization='Source-supported possibilities only; no operator-level causal proof',rows=rows)
    (REPORT/'repeat_noise_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
