# MapTR mini → Rerun implementation plan

Goal: run the existing pretrained MapTR on scene-0061 and replay measured vector predictions alongside existing detections.
Architecture: isolated legacy inference environment; six-camera inputs and calibration only; per-frame vectors in reference ego coordinates; Rerun loads token-validated predictions and clears the layer every frame. No map GT is fed to inference. Model has no height output: display on an explicitly approximate ground plane.

- [x] Inspect upstream config, checkpoint, coordinate conventions and dependencies; create isolated environment without modifying working detectors.
- [x] Add geometry tests for camera projection, LiDAR-to-ego vector conversion and invalid/token-mismatched results; observe failure before implementation.
- [x] Implement tools/maptr_mini.py and reusable geometry; strictly load all checkpoint parameters; run one frame, then all 39; preserve model/config hashes and raw scores.
- [x] Extend tools/rerun_mini.py with per-class line layers, a BEV view, and score threshold; atomic recording replacement after validation.
- [x] Verify all frame tokens, finite vectors, recording contents, HTTP loading and actual browser rendering; update README/RERUN.md/TASK_STATE with commands and limitations.
