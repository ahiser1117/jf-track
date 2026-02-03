# jf-track

A video tracking system for jellyfish microscopy, featuring two-pass detection of anatomical features and adaptive background subtraction for rotating backgrounds.

NOTE: This app was entirely developed using Agent-based development tools (Claude Code, Codex, OpenCode).

## Features

- **Two-pass tracking**: Detects larger features (mouth) and smaller features (bulbs) separately with different size thresholds
- **Full-video median (non-rotating)**: Builds a single median-intensity background from the entire non-rotating clip (or the configured `max_frames`) and reuses it for all detections
- **Adaptive background for rotation**: Handles rotating samples with ORB + RANSAC and a STATIC/ROTATING/TRANSITION state machine that skips detections until a fresh static episode is ready
- **Rotation-compensated background**: Handles rotating samples with ORB + RANSAC and a STATIC/ROTATING/TRANSITION state machine
- **Direction analysis**: Computes orientation vector from bulb center-of-mass to mouth position
- **Circular ROI masking**: Constrains tracking to central region where jellyfish is expected
- **Search-radius filtering**: Optional spatial filters for mouth re-acquisition and bulb detection

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

Running the entrypoint launches a small Tkinter GUI that walks through the entire configuration:

1. Choose the video file.
2. Answer whether the video is rotating (enables adaptive background automatically).
3. Decide if you want to draw a custom ROI (circle, polygon, or bounding box). Skipping this step keeps the defaults (centered circle for rotating videos, entire frame for non-rotating videos).
4. Decide if you want to run interactive feature sampling for automatic parameter tuning.
5. Specify how many mouths, gonads, and tentacle bulbs are visible (bulbs can be left blank for auto-detect).
6. Select how the background threshold should be determined. By default the app computes a per-video Otsu threshold (recommended), but you can enter a manual value if needed.

After the prompts, the tracker runs with the selected parameters, saves results to `tracking_results/multi_object_tracking.zarr`, and renders two visualizations automatically:

- `multi_object_labeled.mp4`: standard annotated video.
- `multi_object_labeled_composite.mp4`: labeled frame + background + diff side by side, using the same adaptive/rolling background logic as the tracker.

When the wizard asks for a frame limit, enter `1000` (for example) to run only the first thousand frames; leave it blank to process the entire video.

- `debug_background.py` helps visualize raw/background/diff images and multiple thresholds before running the full tracker:

```
python debug_background.py --video path/to/video.mp4 --frame 250 --show
python debug_background.py --video path/to/video.mp4 --frame 250 --auto-threshold --thresholds 5 10 20 --save-prefix bg_debug
```

This makes it easy to confirm that the histogram-based threshold highlights the jellyfish before running the full pipeline.

### Programmatic use

For scripted experiments you can still call the underlying APIs directly:

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

- **Non-rotating videos**: `BackgroundProcessor` loads the clip once, computes a median-intensity projection across every available frame (or `max_frames` if specified), and optionally derives an Otsu threshold from the residuals. This produces a single global background that matches what the tracker and labeled videos use.

- **Rotating videos**: `AdaptiveBackgroundManager` monitors frame-to-frame rotation with ORB + RANSAC, maintains a STATIC/ROTATING/TRANSITION state machine, and only produces masks during STATIC episodes. Buffered frames are rotated to align with the current orientation before computing the median, and search centers are rotated while the sample spins so they reappear in the correct coordinates when the episode stabilizes.

Tracking is **skipped during ROTATING/TRANSITION** states; detections only occur during STATIC episodes.

### Two-Pass Detection

Objects are detected using connected component analysis on the background-subtracted binary mask:

1. **Mouth detection**: Finds objects with area between `mouth_min_area` and `mouth_max_area` (default 35-160 pixels)
2. **Bulb detection**: Finds smaller objects between `bulb_min_area` and `bulb_max_area` (default 5-35 pixels)

Both passes use the same background-subtracted mask but with different size filters.

### ROI Masking

- **Rotating videos** default to a centered circle, but you can draw custom circles/polygons/bounding boxes via the GUI prompts or the interactive feature-sampling workflow. The selected ROI is stored in `TrackingParameters` and reused for downstream visualization.
- **Non-rotating videos** default to full-frame ROI unless a custom shape is provided.
- ROI masks are applied to binary masks after thresholding, before any connected-component analysis.

### Track Merging

The mouth may be temporarily lost due to occlusion. The `merge_mouth_tracks()` function links non-overlapping track segments into a single continuous track. When tracks overlap (rare), the track closest to the last known position is preferred; if no prior position exists, the larger-area detection is used.

### Direction Analysis

For each frame, the system computes:
- Bulb center-of-mass (average position of all detected bulbs)
- Direction vector from bulb CoM to mouth
- Direction angle and magnitude (0° = right, 90° = up in image coordinates)

Positions are temporally smoothed to reduce noise.

## Output Format

Results are stored in zarr format with the following structure:

```
output.zarr/
├── mouth/
│   ├── track     # Track IDs
│   ├── x, y      # Positions (n_tracks, n_frames)
│   ├── area      # Detection area
│   └── ...
├── bulb/
│   └── ...       # Same structure as mouth
└── direction/
    ├── mouth_x, mouth_y
    ├── bulb_com_x, bulb_com_y
    ├── bulb_count
    ├── direction_x, direction_y
    ├── direction_magnitude
    └── direction_angle_deg
```

Tracking parameters are stored in zarr attributes (`parameters`) when provided to `save_two_pass_tracking_to_zarr()`.

## Visualization

- `save_multi_object_labeled_video()` (used by `main.py`) renders both the standard labeled video and an optional composite view that stacks the labeled frame, the current background image, and the diff/mask output. This composite uses the same rolling/adaptive background processor as the tracker, so what you see exactly matches what the detection step saw.
- `save_two_pass_labeled_video()` remains available for legacy two-pass runs and supports the `"original"`, `"diff"`, and `"mask"` background modes described above. Both visualization helpers auto-load parameters from the `.zarr` output to keep overlays and search radii consistent with the tracking configuration.

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
