# Claude Code Context for jf-track

This file provides context for AI agents working on this jellyfish tracking codebase.

## Project Overview

jf-track is a video tracking system designed to track jellyfish in microscopy videos. It uses background subtraction and two-pass detection to identify:
1. **Mouth** - A larger anatomical feature (35-160 pixels)
2. **Bulbs** - Smaller features around the mouth (5-35 pixels)

The system computes a direction vector from the bulb center-of-mass to the mouth position.

## Key Architecture Decisions

### Rolling Background Subtraction
Background is computed as the **median of the last N frames** (default 10), not a static image sampled across the video. This handles gradual illumination changes and is implemented in:
- `RollingBackgroundManager` in `src/adaptive_background.py`
- Used by both tracking (`run_two_pass_tracking`) and visualization (`save_two_pass_labeled_video`)

### Adaptive Background for Rotation
When `adaptive_background=True`, the system handles videos where the background rotates:
- Uses ORB feature matching + RANSAC to estimate frame-to-frame rotation
- State machine: STATIC → ROTATING → TRANSITION → STATIC
- During rotation: aligns buffered frames to current orientation before computing median
- Automatically estimates rotation center from larger angular displacements

### Circular ROI Mask
Tracking is constrained to a circle centered at `(width/2, height/2)` with radius `min(width, height)/2`. This is because the jellyfish is expected to remain in this central region.

### Track Merging
The mouth may be temporarily lost (occlusion) and reacquired. `merge_mouth_tracks()` links non-overlapping track segments into one continuous track. For overlapping frames, the detection with larger area is used.

## File Structure

```
src/
├── tracking.py           # Main tracking pipeline, ROI mask
├── tracker.py            # RobustTracker class, TrackingData dataclass
├── adaptive_background.py # Rolling and rotation-compensated background
├── direction_analysis.py  # Bulb CoM → mouth direction computation
├── visualizations.py      # Video output with annotations
└── save_results.py        # Zarr storage
```

## Key Functions

### `run_two_pass_tracking()` in `src/tracking.py`
Main entry point. Returns `(mouth_tracking, bulb_tracking, fps)`.

Important parameters:
- `background_buffer_size`: Frames for rolling median (default 10)
- `adaptive_background`: Enable rotation compensation
- `rotation_start_threshold_deg`: Degrees/frame to trigger rotation detection
- `mouth_min_area`/`mouth_max_area`: Pixel area bounds for mouth detection
- `bulb_min_area`/`bulb_max_area`: Pixel area bounds for bulb detection

### `merge_mouth_tracks()` in `src/tracking.py`
Links non-overlapping mouth track segments. Call after `run_two_pass_tracking()`.

### `compute_direction_analysis()` in `src/direction_analysis.py`
Computes per-frame direction from bulb center-of-mass to mouth. Uses temporal smoothing.

### `save_two_pass_labeled_video()` in `src/visualizations.py`
Creates annotated video. `background_mode` options:
- `"original"`: Show original video
- `"diff"`: Show background subtraction (uses same rolling background as tracking)
- `"mask"`: Show binary mask with ROI applied

### `visualize_adaptive_background()` in `src/adaptive_background.py`
Diagnostic visualization showing state machine, rotation angles, and background alignment.

## Data Structures

### `TrackingData` (dataclass in `src/tracker.py`)
Arrays of shape `(n_tracks, n_frames)` with NaN for missing data:
- `x`, `y`: Centroid positions
- `area`: Detection area in pixels
- `bbox_*`: Bounding box coordinates

### `DirectionAnalysis` (dataclass in `src/direction_analysis.py`)
Arrays of shape `(n_frames,)`:
- `mouth_x`, `mouth_y`: Smoothed mouth position
- `bulb_com_x`, `bulb_com_y`: Bulb center of mass
- `direction_x`, `direction_y`: Direction vector
- `direction_angle_deg`: Angle in degrees

## Common Gotchas

1. **Background mode in visualization**: Must use `RollingBackgroundManager` to match what tracking sees, not a static sampled background.

2. **Frame indexing**: Tracking uses 0-based indices internally. The `frame` array in `TrackingData` stores frame numbers.

3. **ROI mask**: Applied to binary mask after thresholding, before object detection. Anything outside the circle is ignored.

4. **Rotation center estimation**: Only reliable when cumulative rotation exceeds `_min_angle_for_center` (default 3 degrees). Uses weighted average of estimates.

5. **Track merging**: `select_best_mouth_track` is an alias for `merge_mouth_tracks` for backwards compatibility.

## Testing Changes

After modifying tracking logic:
1. Run on a short clip first (`max_frames=500`)
2. Check `visualize_adaptive_background()` output to verify background subtraction
3. Use `background_mode="diff"` or `"mask"` in labeled video to see what tracking sees
4. Verify ROI circle is visible in visualizations
