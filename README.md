# jf-track

jf-track is a prompt-driven jellyfish tracking toolkit with a configurable multi-object pipeline (mouth, gonads, tentacle bulbs) and a legacy two-pass mode for backwards compatibility.

NOTE: This app was entirely developed using agent-based development tools (Claude Code, Codex, OpenCode).

## Features

- **Prompted multi-object workflow** – `python main.py` launches a Tkinter wizard that captures video metadata, rotation/pinned-mouth choices, ROI selection, and class counts before running the modern multi-pass tracker.
- **Class-specific heuristics** – Mouth/gonad/bulb passes reuse component statistics but apply independent area/shape filters plus `search_radius`, `track_search_radius`, `ownership_radius`, `exclude_objects`, and score margins to keep labels stable.
- **Adaptive & static backgrounds** – Non-rotating clips use a 60-sample median projection (with optional Otsu threshold). Rotating clips rely on the adaptive ORB+RANSAC state machine (STATIC → ROTATING → TRANSITION), which pauses detections during motion and rotates stored search points.
- **Interactive tools** – ROI selector (circle/polygon/bounding box), feature sampler, and pinned-mouth workflows feed their configuration straight into `TrackingParameters`, so downstream visualizations reproduce the same ROI, search radii, and thresholds.
- **Visualization & analysis** – Every run emits `multi_object_tracking.zarr`, `multi_object_labeled.mp4`, and the composite diff/background video. `analyze.py` produces pulse-distance plots/CSVs, while `debug_background.py` previews thresholds.
- **Pulse QA composite** – Optional `--pulse-video` flag renders a second MP4 that pairs the labeled frame with a live pulse-distance plot cursor so you can spot-check the timing of swim cycles.
- **Legacy compatibility** – `run_two_pass_tracking()` and the original `save_two_pass_*` helpers remain available when only mouth + bulb tracking is required.

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/ahiser1117/jf-track.git
cd jf-track

# Install dependencies (creates .venv automatically)
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/ahiser1117/jf-track.git
cd jf-track

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package and its dependencies
pip install .
```

### Dependencies

- Python 3.12+
- Core tracking dependencies: OpenCV (`opencv-python`), NumPy, SciPy, scikit-image, zarr
- See `pyproject.toml` for the full dependency list (analysis notebooks and optional tooling)

## Quick Start

### Prompt-driven workflow (recommended)

```
python main.py
```

Running the entrypoint launches a Tkinter GUI that walks through the configuration:

1. Choose the video file.
2. Answer whether the video is rotating (enables adaptive background automatically).
3. Indicate if the mouth is **pinned/occluded**. When pinned, you’ll click once on the median projection to set a fixed reference point instead of tracking the mouth.
4. Decide if you want to draw a custom ROI (circle, polygon, or bounding box). Skipping this step keeps the defaults (centered circle for rotating videos, entire frame for non-rotating videos).
5. Decide if you want to run interactive feature sampling for automatic parameter tuning (this workflow now sets the background threshold directly).
6. Specify how many mouths, gonads, and tentacle bulbs are visible (bulbs can be left blank for auto-detect). When the mouth is pinned the count is forced to zero automatically.
7. Enter an optional frame limit (leave blank to process the full clip).

After the prompts, the tracker runs with the captured parameters, writes everything to `<video_dir>/<video_name>_results/`, and renders:

- `multi_object_tracking.zarr`: per-object tracking arrays plus serialized `TrackingParameters`.
- `multi_object_labeled.mp4`: standard overlay.
- `multi_object_labeled_composite.mp4`: labeled frame + background + diff panes, all backed by the same `BackgroundProcessor` instance used for tracking.
- `multi_object_labeled_pulse.mp4` (when `--pulse-video` is provided): labeled frame on the left, synced bulb-distance plot with a moving cursor on the right.

When the wizard asks for a frame limit, enter `1000` (for example) to run only the first thousand frames; leave it blank to process the entire video.

- `analyze.py` (optional): once the run finishes, analyze tentacle-bulb pulse distances directly from the video or the results directory:

```bash
python analyze.py /path/to/video.mp4         # or /path/to/video_results/
```

The script automatically discovers `multi_object_tracking.zarr`, computes the per-frame center-of-mass of the tentacle bulbs, measures each bulb’s distance to that center, and saves both `pulse_distance.png` and `pulse_distance.csv` in the results folder. Pass `--show` to open the plot interactively, or `--no-plot` / `--no-csv` to suppress files.

- `debug_background.py` helps visualize raw/background/diff images and multiple thresholds before running the full tracker:

```
python debug_background.py --video path/to/video.mp4 --frame 250 --show
python debug_background.py --video path/to/video.mp4 --frame 250 --auto-threshold --thresholds 5 10 20 --save-prefix bg_debug
```

If you skip feature sampling, the tracker falls back to the automatic per-video threshold derived from the sampled background. `debug_background.py` remains a handy way to preview that threshold before you run the full pipeline.

Need a synced QA artifact? Launch the prompt workflow with `--pulse-video` to generate the pulse composite automatically (use `--pulse-object-type gonad` to visualize other classes). The extra MP4 lives alongside the standard outputs.

### Programmatic use

For scripted experiments you can call either the modern multi-object API or the legacy two-pass helper.

```python
from src.tracking import run_multi_object_tracking
from src.tracker import TrackingParameters
from src.save_results import save_multi_object_tracking_to_zarr
from src.visualizations import save_multi_object_labeled_video

params = TrackingParameters()
params.video_type = "rotating"
params.num_gonads = 2
params.num_tentacle_bulbs = 8
params.update_object_counts()

results, fps = run_multi_object_tracking(
    "path/to/video.mp4",
    params,
    max_frames=500,
)
save_multi_object_tracking_to_zarr(results, "output.zarr", fps, params)
save_multi_object_labeled_video(
    video_path="path/to/video.mp4",
    zarr_path="output.zarr",
    output_path="labeled.mp4",
    composite_output_path="labeled_composite.mp4",
    show_search_radii=True,
)
```

Legacy two-pass usage:

```python
from src.tracking import run_two_pass_tracking, merge_mouth_tracks
from src.direction_analysis import compute_direction_analysis
from src.save_results import save_two_pass_tracking_to_zarr
from src.visualizations import save_two_pass_labeled_video

mouth_tracking_raw, bulb_tracking, fps, params = run_two_pass_tracking(
    "path/to/video.avi",
    max_frames=1000,
    background_buffer_size=10,
    adaptive_background=True,
)
mouth_tracking = merge_mouth_tracks(mouth_tracking_raw)
direction = compute_direction_analysis(mouth_tracking, bulb_tracking)
save_two_pass_tracking_to_zarr(mouth_tracking, bulb_tracking, direction, "output.zarr", fps, params)
save_two_pass_labeled_video("path/to/video.avi", "output.zarr", "labeled.mp4")
```

## How It Works

### Background Subtraction

- **Non-rotating videos** – `BackgroundProcessor` samples up to 60 evenly spaced frames (respecting `max_frames`), computes a median background, and optionally derives an Otsu threshold from the residuals. The same background feeds both tracking and labeled videos.
- **Rotating videos** – `AdaptiveBackgroundManager` estimates rotation with ORB + RANSAC, tracks STATIC/ROTATING/TRANSITION states, and aligns buffered frames to the current orientation before recomputing the median. Search points (mouth history, ROI vertices, pinned references) are rotated while motion occurs. Tracking resumes only when the state returns to STATIC.

### Multi-Object Detection

- The background-subtracted binary mask is reused for each enabled object type. Components are filtered by class-specific area/shape ranges and then rescored using the configured heuristics.
- `search_radius` + `track_search_radius` keep detections local to either the mouth reference or each tracker’s smoothed centroid history.
- `ownership_radius`, `exclude_objects`, and configurable `score_margin` comparisons ensure components are only claimed when the class clearly wins, preventing gonad/bulb swaps when blobs overlap.
- Accepted components are removed from contention so downstream passes cannot reuse the same blob. Pinned-mouth mode bypasses detection entirely but still provides a reference point for gonads/bulbs.

### ROI Masking

- Rotating videos default to a centered circle; non-rotating videos default to the full frame. Interactive selectors or feature sampling can persist a circle, polygon, or bounding-box ROI that downstream visualizations reuse.
- Masks are applied immediately after thresholding and before measuring connected components.

### Legacy Two-Pass & Track Merging

- The legacy two-pass helper still runs mouth+bulb detection with area-only constraints. `merge_mouth_tracks()` can combine fragmented mouth segments into a continuous track for that mode.

### Direction & Pulse Analysis

- `compute_direction_analysis()` calculates bulb center-of-mass → mouth vectors (with smoothing) and stores direction arrays for the two-pass workflow. `analyze.py` reads the multi-object `.zarr` to compute pulse-distance series and optionally emit `pulse_distance.png/.csv`.

## Output Format

- `multi_object_tracking.zarr` contains one group per enabled object type (`mouth/`, `gonad/`, `tentacle_bulb/`, etc.). Each group mirrors the `TrackingData` arrays (`track`, `x`, `y`, `area`, `bbox_*`, `frame`). Global attributes store `fps`, `object_types`, and the serialized `TrackingParameters`.
- Legacy runs that call `save_two_pass_tracking_to_zarr()` produce `mouth/`, `bulb/`, and `direction/` groups plus the same parameter metadata.

## Visualization

- `save_multi_object_labeled_video()` renders both the standard labeled video and an optional composite view that stacks the labeled frame, current background, and diff/mask output. Because it reuses the same `BackgroundProcessor`, overlays stay aligned with what the tracker processed.
- `save_two_pass_labeled_video()` remains available for legacy runs and supports the `"original"`, `"diff"`, and `"mask"` background modes.

## Configuration

Key parameters for `run_two_pass_tracking()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `background_buffer_size` | 10 | Frames sampled per static episode for median background |
| `threshold` | 10 | Binary threshold for background subtraction |
| `mouth_min_area` | 35 | Minimum mouth area (pixels) |
| `mouth_max_area` | 160 | Maximum mouth area (pixels) |
| `bulb_min_area` | 5 | Minimum bulb area (pixels) |
| `bulb_max_area` | 35 | Maximum bulb area (pixels) |
| `adaptive_background` | False | Enable rotation compensation |
| `rotation_start_threshold_deg` | 0.01 | Rotation detection threshold (deg/frame) |
| `rotation_stop_threshold_deg` | 0.005 | Rotation stop threshold (deg/frame) |
| `mouth_search_radius` | None | Max distance from last mouth position for re-detection |
| `bulb_search_radius` | None | Max distance from mouth for bulb filtering |

## License

MIT License
