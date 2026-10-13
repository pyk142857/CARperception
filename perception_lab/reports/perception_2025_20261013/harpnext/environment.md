# HARP-NeXt environment

Python 3.10.13, Torch 2.1.2+cu118, torch-scatter 2.1.2+pt21cu118, NumPy 1.26.2, CUDA runtime 11.8. Author recommends Torch2.5.0/cu118; this run reused Torch2.1.2/cu118 in an isolated22MiB venv without changing old environments. Source clone40MiB. No large dependency installs.

GPU scatter_mean([1,3,8],[0,0,1]) returned [2,8]. Native GPU constructor initially failed because Torch2.1 lacks torch.uint32. The recorded patch changes only two empty label reset buffers to int64; set_label overwrites them with input-derived tensors before projection. No model math/weight layout was changed. Following checkpoint recovery, strict loading and all404 native network forwards passed in this same environment. See strict_load.json, module_hooks.json and report.md. No environment upgrade was needed.
