# Claude Code Context for jf-track

This file provides context for AI agents working on this jellyfish tracking codebase.

## Project Overview

jf-track is a video tracking system designed to track jellyfish in microscopy videos. It uses background subtraction and two-pass detection to identify:
1. **Mouth** - A larger anatomical feature (35-160 pixels)
2. **Bulbs** - Smaller features around the mouth (5-35 pixels)

The system computes a direction vector from the bulb center-of-mass to the mouth position.

## Key Architecture Decisions

### Background Subtraction Strategy
- **Non-rotating videos**: build a single median-intensity projection using every available frame (or `max_frames` when set). The global background is thresholded once, optionally via Otsu, and reused for both tracking and visualization.
- **Rotating videos**: rely on the adaptive background state machine from `AdaptiveBackgroundManager`. Static episodes fill a buffer that produces a median background; ROTATING/TRANSITION frames are skipped until the state returns to STATIC. `BackgroundProcessor` exposes rotation-aware helpers so search points can be rotated between episodes.

### Adaptive Background for Rotation
When `adaptive_background=True`, the system handles videos where the background rotates:
- Uses ORB feature matching + RANSAC to estimate frame-to-frame rotation
- State machine: STATIC → ROTATING → TRANSITION → STATIC
- During rotation: aligns buffered frames to current orientation before computing median and rotates stored search points so they remain aligned with the animal
- Automatically estimates rotation center from larger angular displacements

### ROI Handling
- Users can define circular, polygonal, or bounding-box ROIs via the GUI prompts or the feature-sampling workflow. These ROI settings are saved inside `TrackingParameters` and reused when visualizing results.
- Rotating videos default to a centered circle; non-rotating videos default to a full-frame mask unless the user specifies otherwise.

### Pinned Mouth Mode
- Some recordings pin or occlude the mouth, making it impossible to track directly. The prompt workflow now asks whether the mouth is pinned. When enabled, the user clicks a single reference point on the median projection instead of tracking the mouth feature.
- `TrackingParameters.mouth_pinned=True` disables mouth detection, stores the fixed `pinned_mouth_point`, and uses that reference for all gonad/bulb search radii in both tracking and visualization.

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
Main entry point for the legacy mouth/bulb pipeline. Returns `(mouth_tracking, bulb_tracking, fps)`.

Important parameters:
- `background_buffer_size`: Number of frames persisted in the adaptive background buffer. For non-rotating videos the processor ignores this and uses the full-video median background.
- `adaptive_background`: Enable rotation compensation
- `rotation_start_threshold_deg`: Degrees/frame to trigger rotation detection
- `mouth_min_area`/`mouth_max_area`: Pixel area bounds for mouth detection
- `mouth_search_radius`: Max distance (pixels) from last known position to search for mouth and bulbs when mouth is lost (None = no limit). When `adaptive_background=True` the search center is rotated automatically when the state machine detects motion.
- `bulb_min_area`/`bulb_max_area`: Pixel area bounds for bulb detection
- `bulb_search_radius`: Max distance (pixels) from mouth to consider bulbs when mouth is tracked (None = no limit)

### `merge_mouth_tracks()` in `src/tracking.py`
Links non-overlapping mouth track segments. Call after `run_two_pass_tracking()`.

### `compute_direction_analysis()` in `src/direction_analysis.py`
Computes per-frame direction from bulb center-of-mass to mouth. Uses temporal smoothing.

### `save_two_pass_labeled_video()` in `src/visualizations.py`
Creates annotated video. `background_mode` options:
- `"original"`: Show original video
- `"diff"`: Show background subtraction (uses same rolling background as tracking)
- `"mask"`: Show binary mask with ROI applied

Optional search radius visualization:
- `bulb_search_radius`: Draws cyan circle around mouth showing bulb search area
- `mouth_search_radius`: Draws orange circle when mouth is lost showing search area
- `adaptive_background`: When enabled, the search area rotates with the background (matches tracking behavior)

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

1. **Background mode in visualization**: `BackgroundProcessor` is shared between tracking and both visualization paths. Always drive labeled videos through the processor rather than sampling your own background so that thresholds/ROIs stay aligned.

2. **Frame indexing**: Tracking uses 0-based indices internally. The `frame` array in `TrackingData` stores frame numbers.

3. **ROI mask**: Applied to binary mask after thresholding, before object detection. Rotating videos default to a centered circle; non-rotating videos default to the full frame. GUI-selectable ROIs override both behaviors.

4. **Rotation center estimation**: Only reliable when cumulative rotation exceeds `_min_angle_for_center` (default 3 degrees). Uses weighted average of estimates.

5. **Track merging**: `select_best_mouth_track` is an alias for `merge_mouth_tracks` for backwards compatibility.

## Testing Changes

After modifying tracking logic:
1. Run on a short clip first (`max_frames=500`).
2. For rotating footage, check `visualize_adaptive_background()` to review state transitions and rotation center estimation.
3. For non-rotating footage, confirm the full-video median background looks reasonable (no residual jellyfish).
4. Use `background_mode="diff"` or `"mask"` in labeled video to see exactly what the tracker processed.
5. Verify the expected ROI outline appears in visualizations (circle/polygon/bounding box), especially when loading saved `.zarr` results.
6. For pinned-mouth runs, confirm the fixed reference marker appears in the labeled/composite videos and that gonad/bulb detections stay within the configured search radii.
