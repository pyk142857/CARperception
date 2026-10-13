# GRAE environment

Inference uses an isolated Python 3.10 venv with read-only existing Torch 1.13.1+cu117 packages; the original author stack is Torch 1.9/cu111, so this is a validated compatibility environment. Existing TensorRT/SpConv environments were not modified. Official TrackingEval runs with existing `envs/sparse4d/bin/python` and nuScenes devkit 1.2.0.

```json
{
  "python": "3.10.13",
  "torch": "1.13.1+cu117",
  "cuda_runtime": "11.7",
  "dependencies": {
    "lap": "0.5.13",
    "fvcore": "0.1.5.post20221221",
    "numpy": "1.23.5",
    "scipy": "1.10.1",
    "nuscenes-devkit": "1.1.11"
  },
  "environment_scope": "isolated envs/grae3dmot_2025 read-only overlay of existing bevfusion/maptr torch; old environments unchanged"
}
```

SimpleTrack pinned dependency is NMS-only. Source and weight identities are in resource_manifest.json; actual strict-load adapter details are in checkpoint_key_mapping.json. No graph export or TensorRT tracker is claimed. Native float32 is retained.
