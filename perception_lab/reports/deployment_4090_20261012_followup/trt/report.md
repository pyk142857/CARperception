# TensorRT FP16 follow-up: four fixed tokens

Production inference and isolated diagnostics completed. Full-pipeline same-input output determinism fails; this does not overturn the prior official mAP/NDS budget result. No precision fix, engine replacement, or new speedup is claimed. Reference cross-implementation module replay has also completed; [reference replay](../reference/replay/summary.json) records the 40 shared-input cases.

Data: nuScenes mini, first scene0061 frame plus the three prior >=0.5 anomalous tokens. Each token is preprocessed once, followed by 3 warmups and 5 repeats per production/diagnostic library. Image, point and four matrix FP32 input bytes match reference exactly for all tokens ([identity](input_identity_comparison.json)). Production library/plans remain unchanged; actual loaded paths and hashes are in [production](production_loaded_libraries.json) and [diagnostic](diagnostic_loaded_libraries.json). The independent capture build reuses production objects except the opt-in TensorRT-binding capture wrapper.

## Repeat baseline

| Token prefix | Max changed sparse-depth pixels | Max LiDAR BEV elements outside tolerance | Geometry contribution-count differing cells | Production AA >=0.5 max center m | Production AA >=0.5 max yaw rad |
|---|---:|---:|---:|---:|---:|
| ca9a282c | 1 | 141 | 126 | 0.03768 | 0.01272 |
| e0845f53 | 4180 | 284 | 136 | 0.05322 | 0.00379 |
| b26e7915 | 4473 | 113 | 156 | 0.03773 | 0.05273 |
| 747aa46b | 3334 | 93 | 104 | 0.01922 | 0.01126 |

Full-candidate output statistics and explanatory >=0.25/>=0.5 groups are retained in [output groups](output_repeat_score_groups.json). Scores were not filtered for acceptance. Row-order comparisons are separate from the original class-constrained 2m Hungarian diagnostic; atomic output ordering is not treated as proposal identity.

Camera sparse-depth pixels vary at overlapping projected points; the CPU collision reconstruction is labelled approximate because CUDA FMA/boundary rounding can differ. LiDAR BEV also varies, including on the first frame where camera sparse depth barely changes. This identifies more than one variable upstream boundary, not a proof of a specific sparse-convolution kernel cause. Capturing buffers synchronizes the stream, so capture perturbation is separately compared with the uninstrumented production baseline.

## Same-upstream isolation

All camera-backbone, BEV-pool, camera downsample, fuser+decoder and head replays run five times on identical saved FP16-representable inputs for each of the two upstream sources, across all four tokens. All native output tensors of these isolated production modules are bitwise stable in these runs. The pool invokes the existing production `car_bev_pool_forward`, default upstream mode. Its source and geometry inputs are explicit; the reference replay receives exactly these inputs. This distinguishes local module stability from upstream variation, The completed reference replay passes the fixed tensor tolerance for all camera, pool, downsample and fuser+decoder cases. Head rows require actual source alignment; the marked-graph limitation still applies.

The fuser engine boundary includes fusion, decoder backbone and decoder neck; a raw fusion-only output is not exposed by production. LiDAR voxelization/SCN internals are not captured. See [full module statistics](module_replay.json) and [capture gaps](capture_gaps.json).

Actual geometry total contributions match on each frame, but their per-cell assignments differ (126/136/156/104 cells; max differences 2/1/1/2). This is separate from the previously fixed exclusive final interval endpoint, which remains correct. Assignment differences alone do not establish their arithmetic cause. Same-mapping pool replay does not measure the effect of changing mappings; the additional reference dual-mapping experiment in the reference replay report evaluates that effect separately.

## Candidate identity and instrumentation limits

The existing LayerNorm-plugin head is retained. The separate marked-output ONNX exposes actual `TopK_211` input `724`, values `728`, and selected source indices `729`. Its six original native outputs differ from the unmodified head at the bit level, although all satisfy the fixed tensor tolerance for the observed inputs. Therefore source identity is attributed to the diagnostic graph, not asserted as exact production-engine identity. Its own repeated same-input TopK indices are checked with exact integer equality.

[Candidate source cases](candidate_source_cases.json) retain the full native same-source comparisons, Hungarian pairs annotated with source indices, and nearest-current-reference comparisons to historical cases:

- `747aa46...`: the diagnostic Hungarian center difference remains 1.6280m, pairing reference source 278404 with candidate source 278222. The candidate with the actual same source 278404 is only 0.01872m away in standardized coordinates (native bottom-center difference 0.02023m). This supports candidate exchange in the diagnostic graph.
- `e0845f53...`: 0.94450m diagnostic Hungarian center difference pairs source 181015 with 180836; source 181015 masked score changes 0.653751→0, while neighboring 180836 changes 0→0.673828. These are scores after heatmap local max-pool peak selection and before TopK. This is not merely a global cutoff event and is not post-decoding box NMS. Candidate raw pre-mask logits remain a capture gap.
- `b26e7915...`: both sides retain source 19731; native yaw differs 0.33761rad and native center 0.02813m (standardized center 0.03272m). Candidate exchange does not explain this diagnostic same-source yaw difference.

The [high-score same-input head control](high_score_same_input_head_control.json) further isolates upstream effects. For b26/source 19731, identical-middle FP32-reference versus marked-FP16-head yaw differences are only 0.007411 and 0.000470rad for the two upstream sources. The FP32 reference head itself changes yaw from -1.279433 to -0.941354rad when given the other upstream middle: approximately the observed full-path 0.338rad shift. For e084/source 181015, both heads select it on reference middle and both omit it on TRT middle. These observations support propagation through changed upstream features for these cases; they do not identify a specific sensitive layer.

P2F: no new FP32 layer/plugin variant is justified by the current localization. No precision modification was made. The existing LayerNorm plugin already uses FP32 reduction in its half path; it is not a new fix. Production weights, configuration, libraries, plans and decode thresholds are unchanged, so the existing 39/81-frame evaluation results are retained and no new full evaluation is claimed.

Historical outputs did not store source identities. These are newly measured correspondences near the historical cases, not proof of the historical engine's internal query selection. Pre-TopK scores, selected indices and cutoff/sort margins remain in local replay NPZ files.

## Error convention and reproduction

Float tensor acceptance is `abs(candidate-reference) <= 0.05 + 0.02*abs(reference)` with nonfinite values failing. Reports include shapes/dtypes, nonfinite counts, error quantiles, exceeded counts, and near-zero absolute errors; relative errors are restricted to `abs(reference)>0.01`. Integer source/mapping identity uses exact equality. No threshold was relaxed.

See [captured node shapes/dtypes/layouts](capture_nodes.json), [metrics](metrics.csv), [commands](run_commands.md), [status](status.json). These four-token results are teaching diagnostics, not a new official 39/81-frame evaluation. Raw captures, marked plan and module inputs/outputs remain under `perception_lab/runs/deployment_4090_20261012_followup/trt/`; large binary captures and engines are excluded from Git.

Runner source identities are in [version provenance](runner_version_provenance.json). Launch-time Python hashes were not originally captured: earlier versions were reconstructed from this session’s edit log and labelled accordingly; this limitation is distinct from the runtime-verified loaded library/engine hashes. The four subsequent same-input replay comparisons retain summary statistics, not separate raw tensors for every replay; independent offline byte comparison of all five runs is therefore unavailable.
