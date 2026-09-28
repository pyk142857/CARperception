# Local Rerun deployment

Implements the Rerun-first direction approved in the README discussion. Reuse the existing 39-frame mini scene and measured detections/tracks without rerunning GPU inference. Install a pinned SDK/viewer in `perception_lab/envs/rerun` and save recordings under ignored `outputs/rerun/`.

Log six cameras with calibrated pinholes and sensor-time ego compensation into a reference-ego frame. Read raw lidar in its sensor frame and transform it into that same reference frame. Convert global GT to reference ego. Existing standardized CenterPoint and PubTracker outputs are already in reference ego; their `native_coordinate_system` describes the pre-adapter format, not the standardized boxes. Reorder nuScenes width/length/height to Rerun x/y/z length/width/height, and quaternion wxyz to xyzw.

Group records on a sample/frame timeline for joint inspection, preserving physical sensor timestamps as inspectable metadata. This is a keyframe visualization, not resampling sensors to an identical acquisition time. Track history is accumulated in world coordinates and transformed back into each current reference-ego frame. Only active tracks are rendered per frame; clear dynamic layers to prevent stale boxes.

Serve only on loopback. Provide a launcher, documentation and a saved recording; verify frame coverage and coordinate conversions, then inspect the viewer. Other visualization products and new inference are out of scope.
