# jf-track

A video tracking system for jellyfish microscopy, featuring two-pass detection of anatomical features and adaptive background subtraction for rotating backgrounds.

## Features

- **Two-pass tracking**: Detects larger features (mouth) and smaller features (bulbs) separately with different size thresholds
- **Rolling background subtraction**: Uses median of recent frames rather than a static background
- **Rotation-compensated background**: Handles videos where the sample rotates, automatically detecting rotation and aligning frames
- **Direction analysis**: Computes orientation vector from bulb center-of-mass to mouth position
- **Circular ROI masking**: Constrains tracking to central region where jellyfish is expected

## Installation

### Using uv (recommended)

```bash
# Clone the repository
git clone https://github.com/ahiser1117/jf-track.git
cd jf-track

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/ahiser1117/jf-track.git
cd jf-track

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.10+
- OpenCV (`opencv-python`)
- NumPy
- SciPy
- scikit-image
- zarr

## Quick Start

```python
from src.tracking import run_two_pass_tracking, merge_mouth_tracks
from src.direction_analysis import compute_direction_analysis
from src.save_results import save_two_pass_tracking_to_zarr
from src.visualizations import save_two_pass_labeled_video

# Run tracking
mouth_tracking_raw, bulb_tracking, fps = run_two_pass_tracking(
    "path/to/video.avi",
    max_frames=1000,
    background_buffer_size=10,
    adaptive_background=True,  # Enable if background rotates
)

# Merge mouth track segments (handles occlusion gaps)
mouth_tracking = merge_mouth_tracks(mouth_tracking_raw)

# Compute direction analysis
direction = compute_direction_analysis(mouth_tracking, bulb_tracking)

# Save results
save_two_pass_tracking_to_zarr(mouth_tracking, bulb_tracking, direction, "output.zarr", fps)

# Create visualization
save_two_pass_labeled_video("path/to/video.avi", "output.zarr", "labeled.mp4")
```

## How It Works

### Background Subtraction

The system uses a **rolling median background** computed from the last N frames (default 10). This adapts to gradual changes in illumination while maintaining sensitivity to moving objects.

For videos where the sample rotates:
1. Frame-to-frame rotation is estimated using ORB feature matching
2. When rotation exceeds a threshold, the system enters "rotating" mode
3. Buffered frames are rotated to align with the current orientation before computing the median
4. After rotation stops, the system transitions back to static mode with a fresh background

### Two-Pass Detection

Objects are detected using connected component analysis on the background-subtracted binary mask:

1. **Mouth detection**: Finds objects with area between `mouth_min_area` and `mouth_max_area` (default 35-160 pixels)
2. **Bulb detection**: Finds smaller objects between `bulb_min_area` and `bulb_max_area` (default 5-35 pixels)

Both passes use the same background-subtracted mask but with different size filters.

### ROI Masking

A circular region-of-interest mask constrains detection to the center of the frame:
- Center: `(width/2, height/2)`
- Radius: `min(width, height)/2`

Objects outside this region are ignored.

### Track Merging

The mouth may be temporarily lost due to occlusion. The `merge_mouth_tracks()` function links non-overlapping track segments into a single continuous track. When tracks overlap (rare), the detection with larger area is used.

### Direction Analysis

For each frame, the system computes:
- Bulb center-of-mass (average position of all detected bulbs)
- Direction vector from bulb CoM to mouth
- Direction angle and magnitude

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
    ├── direction_x, direction_y
    └── direction_angle_deg
```

## Visualization

The `save_two_pass_labeled_video()` function creates annotated videos with:
- Orange circles for mouth position
- Cyan circles for bulb positions
- Yellow crosshair for bulb center-of-mass
- Green arrow showing direction vector
- Gray circle showing ROI boundary

Background modes:
- `"original"`: Original video frames
- `"diff"`: Background subtraction difference (shows what tracking sees)
- `"mask"`: Binary mask after thresholding

## Configuration

Key parameters for `run_two_pass_tracking()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `background_buffer_size` | 10 | Frames for rolling median background |
| `threshold` | 10 | Binary threshold for background subtraction |
| `mouth_min_area` | 35 | Minimum mouth area (pixels) |
| `mouth_max_area` | 160 | Maximum mouth area (pixels) |
| `bulb_min_area` | 5 | Minimum bulb area (pixels) |
| `bulb_max_area` | 35 | Maximum bulb area (pixels) |
| `adaptive_background` | False | Enable rotation compensation |
| `rotation_start_threshold_deg` | 0.01 | Rotation detection threshold (deg/frame) |

## License

MIT License
