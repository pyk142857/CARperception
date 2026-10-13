# MambaFusion NumPy compatibility: exact source delta

Native loading reached `strict=True` with1,403 state entries, then exposed two existing-author-code incompatibilities with the installed NumPy. The failure logs are preserved as `mamba_numpy_alias_failure.log` and `mamba_numpy_ragged_failure.log`. These are local compatibility fixes, not changes to model weights, layer mathematics or preprocessing geometry.

1. `pcdet/models/backbones_2d/base_bev_backbone.py`: exactly three `astype(np.int)` expressions become `astype(int)`. NumPy's removed `np.int` alias was the Python builtin `int`; the reciprocal-stride rounding/casting calculation is retained.
2. `pcdet/datasets/dataset.py`: a dedicated `img_process_infos` collation branch explicitly builds an object array before stacking. Each camera's resize/crop/flip/rotation tuple is preserved exactly. Existing NumPy rejects this ragged metadata through implicit array construction. The tensor transfer utility already skips `img_process_infos`; other collation branches are unchanged.

The shareable [patch](mambafusion_numpy_compat.patch) contains only these two edits. The helper accepts only each file's exact upstream or exact patched SHA256, validates both files before writing either, and does nothing when already applied. Unknown intervening edits are rejected rather than overwritten.

From the repository root:

```bash
perception_lab/envs/grae3dmot_2025/bin/python perception_lab/tools/mambafusion_numpy_compat_2025.py \
  --source perception_lab/third_party/MambaFusion2025
perception_lab/envs/grae3dmot_2025/bin/python -m pytest \
  perception_lab/tests/test_mambafusion_numpy_compat_2025.py -q
```

The three focused CPU tests cover exact patch bytes/hashes and a second no-op application; unknown-file rejection before partial modification; and the extracted real author collation method preserving six-camera metadata, point batching and unrelated frame metadata. All three passed. Running the helper against the live checkout returned `already_applied` for both files; this packaging task did not alter that running checkout or use the GPU.

## Historical source proof versus the executed tree

The original `mambafusion/source_manifest.json` and its archived copy are intentionally unchanged. They prove the acquired author archive contained550 matching files at acquisition; this does not describe the tree after compatibility fixes and extension building.

The new [current-source audit](mambafusion_source_after_compat.json) hashes all550 historical archive paths:547 currently match; three differ. Two are the explicit compatibility edits above. The third is generated `pcdet/version.py`, changing version suffix `60cd009` to pinned checkout suffix `d85c2fe` during the existing build. The compatibility helper does not modify that generated version file. Additional build products outside the original archive path set are not asserted to be author-source files.

Upstream commit remains `d85c2feeb19207b20e42d50421afb11c095f3efa`. The archive SHA256, preserved historical-manifest SHA256, original/current per-file hashes, and patch SHA256 are recorded in the audit. Native inference acceptance and repeat-output stability are separate checks owned by the running experiment; this compatibility result makes no claim that those have finished.
