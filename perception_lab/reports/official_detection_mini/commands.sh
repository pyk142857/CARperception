#!/usr/bin/env bash
# Run from perception_lab; requires the verified measured mini run and local nuScenes mini.
set -euo pipefail
python_env=envs/nuscenes_eval/bin/python
predictions=runs/20260923T043110_d6ae8066/M05/mini_scene/predictions.json
result=runs/20260923T043110_d6ae8066/M05/mini_scene/nuscenes_detection.json
report=reports/official_detection_mini
"$python_env" tools/export_nuscenes_detection.py --predictions "$predictions" --dataroot data/nuscenes --version v1.0-mini --eval-set mini_scene_0061 --scene scene-0061 --out "$result" --audit "$report/export_audit.json" > "$report/export.log" 2>&1
"$python_env" -m nuscenes.eval.detection.evaluate "$result" --eval_set mini_scene_0061 --version v1.0-mini --dataroot data/nuscenes --output_dir "$report/metrics" --plot_examples 5 --render_curves 1 > "$report/evaluation.log" 2>&1
