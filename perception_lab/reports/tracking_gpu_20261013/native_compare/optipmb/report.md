# OptiPMB native online tracking on frozen BEVFusion detections

This is a nuScenes mini teaching diagnostic, not official full-val performance. Both frozen files were consumed without modifications; all 39 / 81 sample tokens have freshly generated online predictions. Official TrackingEval is run separately by the shared evaluator; consult this folder's `metrics.json` and each split's `official_eval/` for actual accuracy, rather than inferring accuracy from these execution checks.

The author CenterPoint default configuration is applied **unchanged** to fixed BEVFusion output. OptiPMB commit `ed0e0467efb08fd8a96b03647b09169564a74acd` is unmodified. No learned weights are used. Every source/config/library hash, complete per-class configuration, Python/package versions and package reuse paths are in [source_manifest.json](source_manifest.json).

| Split | Raw detections | Tracking allowlist | After author score | After author NMS | Output boxes | Full input-to-tracks time |
|---|---:|---:|---:|---:|---:|---:|
| scene0061, 39 frames | 7,789 | 4,580 | 1,434 | 1,408 | 1,437 | 3.346898 s / 85.82 ms per frame |
| mini_val, 81 frames | 16,199 | 15,007 | 4,390 | 4,288 | 4,034 | 10.741198 s / 132.61 ms per frame |

Timing uses four CPU threads, no CUDA, and includes frozen JSON/packet reads, actual LiDAR file reads, sensor-to-global transforms, measurement formatting, native prediction/update/pruning/extraction and online box construction. Import/config initialization, diagnostic evidence hashes, disk export and evaluation are excluded. These are single-pass descriptive timings, not a paired speedup claim. [timing.json](scene0061/timing.json) and [mini timing](mini_val/timing.json) preserve individual frames. Instrumented replays are separate and are never substituted for production timings.

## Input and execution contracts

The reader accesses only packet sample/scene/sensor metadata plus frozen predicted boxes. It does not open nuScenes GT annotation tables, instance IDs or author GT-aggregated files. Each corresponding keyframe LIDAR_TOP file supplies real XYZ and remission; no historical sweeps are added. Coordinates use `T_ego_to_global @ T_sensor_to_ego`. Source detections are already global and are not transformed again. Size `[w,l,h]` maps to author `[h,w,l]`; quaternions retain `[w,x,y,z]`, while author scipy conversion accepts `[x,y,z,w]`. Exactly as `preprocess_data.py`, the raw author bbox array contains `[h,w,l,x,y,z,roll,pitch,yaw]`; the filter consumes its last element as yaw. Global velocity and original input score are preserved.

`format_measurements` is loaded verbatim from the pinned author's AST to avoid importing unrelated evaluation/HOTA modules. The unchanged native filter executes `predict` (which calls `preprocess_measurments`) → `update` → `prune` → `extract_states`. The state returned by update is pruned in place, matching the author runner. Output uses the current estimate's last mean column, auxiliary score and stored `hwla` quaternion, with first-seen integer label assignment and class-prefixed IDs as in `Run_OptiPMB.py:225–302`. It never exports offline results, backward-smoothed historical means or display-score filtering. A new filter, track dictionary and label sequence are created at every scene boundary. `dt` is the actual LIDAR_TOP timestamp difference in seconds (zero on the first scene frame).

Author score filtering uses strict `> keep_score_thld`, followed by native per-class `NMS_CV2`, and all native existence, birth, prune, score-scaling and point-dependent detection-probability parameters remain unchanged. The complete per-class values are archived in the source manifest; no mini GT parameter tuning occurred.

## Evidence

[Validation summary](validation_summary.json), per-frame [39-frame diagnostics](scene0061/diagnostics.json), [81-frame diagnostics](mini_val/diagnostics.json), and online validations record all execution evidence:

- 120 calls each to native preprocessing, prediction, update, prune and extraction, including each scene's first/middle/last frame.
- Native 3D point counting executed for 1,408 / 4,288 post-NMS boxes. The author removed points with sensor distance ≤3m: 1,354,112→1,014,970 points for scene0061; 2,812,480→2,092,978 for mini_val.
- Track Pd-scaling branches executed 2,805 / 7,963 times; PPP Pd-scaling branches executed 74 / 980 times. Point counts are real and vary by box.
- Real FastMurty `mhtda` executed 38 / 79 times, once on each frame after scene initialization. The author-bundled Linux binary loads directly, so rebuilding was unnecessary. [Known-matrix verification](fastmurty_verification.json) confirms diagonal assignment and optimal cost −17 for `[[-9,-1],[-2,-8]]`, with actual library path and hash. The upstream build commit of that bundled binary is not supplied by the author; provenance is the pinned OptiPMB distribution and exact binary hash, not a claimed independently reproduced build.
- Per-frame LiDAR file hashes, point counts, remission/global-coordinate hashes, transform hashes, input/post-NMS hashes and coordinate round trips are saved. Maximum round-trip errors are 4.55e−13m / 9.09e−13m.
- Separately instrumented complete replays produce identical canonical online output hashes to the timed runs.
- For all three scenes, an isolated three-frame prefix exactly matches the same prefix from full inference, proving these emitted records do not change with future frames. Isolated scene restarts reproduce their full-run prefixes. Empty first-frame and empty-after-live-frame paths execute successfully. The untouched author code emits a NumPy FutureWarning for string comparison on its empty class array; no output failure occurs under the pinned environment.
- Four focused tests cover bbox/velocity/quaternion mapping, class allowlist, exact 3m removal, strict score threshold, genuine point counts, online fields/IDs and the native assignment library. [Test output](focused_tests.txt).

OptiPMB consumes point-count information that GRAE does not. This comparison therefore concerns native tracking systems on identical frozen detections with different auxiliary inputs. Any accuracy difference cannot be attributed exclusively to the PMB association module. CPU timing includes that extra input work. The measured mini scope does not establish full-validation generalization.

Reproduction: [run_commands.md](run_commands.md).
