# Remaining work after published full-chain measurements

The 39/81-frame native comparison, 120-frame TileLang equivalence, official tracking evaluation, 30 full-chain AB/BA pairs, and synchronized browser playback have completed. This task is not fully accepted yet.

The single RTX 4090 is occupied by unrelated waterwaydetection training (initial PID 718139, subsequently separate evaluation processes). That process has not been stopped or modified. The earlier microbenchmark was rejected because the process appeared during measurement; its raw evidence is retained. Full-chain measurements finished before this process began and remain valid.

Resume only when the GPU is free:

1. Rerun `tools/tracking_gpu_benchmark_20261013.py --phase micro` in `envs/grae3dmot_2025`; require a clean monitor and 270 accepted pairs.
2. Additional PyTorch axis-aligned reference has passed (maximum absolute error 0). The profiler reported CUPTI initialization failure; its trace is incomplete and is retained as a diagnostic only, not performance evidence.
3. Run the tracking acceptance audit, regenerate reports, and publish the completed B–E measurements.
4. Run the prepared BEV pooling benchmark with the TensorRT environment. Reject measurements if another compute process is present. If stable local benefit passes, execute the isolated full-model plan and new 39/81-frame detection evaluations; otherwise record the measured non-adoption outcome.
5. Run all-stage acceptance, update status/README and publish the final report.

The isolated pooling patches have been prepared but have not been built or GPU-tested. Browser verification during unrelated training is functional evidence only, never a latency measurement. No background GPU job has been scheduled by this checkpoint.
