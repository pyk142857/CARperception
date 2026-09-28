# Mini detection and tracking diagnostics

Goal: evaluate the existing 39-frame CenterPoint / PubTracker outputs against official annotations and produce traceable failure cases without model reruns.

Scope: fixed-score teaching diagnostics, not official mAP/NDS/AMOTA. Use official category mapping, distance ranges, zero-point and bike-rack filters. Match in global horizontal coordinates. Detection uses score-ordered greedy matching; tracking preserves valid previous associations then uses gated Hungarian assignment. Separate consecutive-frame switches from identity changes after gaps.

- [x] Add synthetic tests: duplicate predictions, wrong class, distance gate, assignment cardinality, switches and gaps.
- [x] Implement reusable matching and identity event helpers.
- [x] Load verified measured outputs, official GT and calibration; validate frame order, save source hashes.
- [x] Compute per-class/frame totals, center errors, score sensitivity, FN/FP/identity cases and exclusions.
- [x] Produce JSON/CSV, Markdown report, representative BEV failure images and a local HTML gallery.
- [x] Run tests, inspect figures, reconcile counts and document reproduction and limits in README.

Files: tools/evaluation_utils.py, tools/evaluate_mini.py, tests/test_evaluation.py, reports/mini_evaluation/ generated artifacts, README.md and perception_lab/README.md.
