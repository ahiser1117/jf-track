# jf-track Multi-Object Tracking System

A comprehensive jellyfish tracking system with configurable object detection, interactive feature sampling, and flexible ROI selection.

## 🎯 New Features

### Multi-Object Detection
- **Mouth**: Large anatomical feature (35-160 pixels) 
- **Gonads**: Oblong structures (0-4, aspect ratio ≥ 1.5, 20-80 pixels)
- **Tentacle Bulbs**: Small round features (user-specified count, 5-35 pixels)

### Interactive Configuration
- **Feature Sampling**: Click on objects to automatically set size/shape parameters
- **ROI Selection**: Draw custom tracking region (circle or polygon)
- **Object Configuration**: Interactive prompts for object counts and parameters

### Enhanced Detection
- **Shape Filtering**: Aspect ratio, eccentricity, and solidity constraints
- **Smart Assignment**: Uses expected counts to improve tracking accuracy
- **Multi-Pass Pipeline**: Mouth → Gonads → Tentacle Bulbs

### Object Permanence
- Mouth detections (or the pinned reference) create a smoothed reference point that downstream classes can follow even when the current frame lacks a mouth detection.
- Each object type defines both `search_radius` (relative to a reference object such as the mouth) and `track_search_radius` (relative to its own smoothed history), keeping reassociation local. `ownership_radius` further dampens identity swaps when multiple detections cluster together.
- `exclude_objects` and per-class `score_margin` rules penalize components that drift too close to other object types, ensuring gonads and bulbs only claim blobs when they clearly win the scoring comparison.
- Once a component is claimed, it is removed from the candidate pool so later passes cannot reuse the same blob in the same frame.

## 🚀 Quick Start

### Prompt Workflow (Recommended)

```
python main.py
```

Launching the app opens a Tkinter wizard that asks:

1. **Video selection** – choose the clip to analyze.
2. **Rotation** – answer “Is this video rotating?” (turns adaptive background on/off).
3. **Pinned mouth** – indicate whether the mouth is pinned/occluded. When pinned you’ll click a single reference point on the median projection, and the tracker will skip mouth detection while still giving the gonads/bulbs a fixed reference.
4. **Custom ROI** – optionally launch the ROI selector (circle, polygon, or bounding box) for both rotating and non-rotating clips. Skipping this step keeps the defaults (centered circle for rotating videos, full-frame mask for non-rotating videos).
5. **Parameter optimization** – decide whether to run interactive feature sampling (the slider chosen there now sets the tracking threshold directly).
6. **Object counts** – specify mouths, gonads, and tentacle bulbs (leave bulbs blank for auto-detect). When the mouth is pinned the wizard automatically disables mouth tracking.
7. **Frame limit** – enter how many frames to process (leave blank to run the entire video). This lets you test the pipeline on the first few hundred frames before committing to a full run.

After tracking completes, a folder named `<video_dir>/<video_name>_results/` is created next to the input clip. It contains:

- `multi_object_tracking.zarr` – the full set of tracks and run parameters.
- `multi_object_labeled.mp4` – standard overlay.
- `multi_object_labeled_composite.mp4` – labeled/original, background, and diff frames side by side so you can compare what the tracker saw.
- `multi_object_labeled_pulse.mp4` – optional QA composite (created when you pass `--pulse-video`) pairing the labeled frame with a live pulse-distance plot cursor.

Run `python main.py --pulse-video` (or `--pulse-video --pulse-object-type gonad`) to generate the pulse composite automatically at the end of the workflow.

### Pulse analysis (new)

Use the top-level helper to visualize tentacle-bulb pulse distances straight from the video or results directory:

```bash
python analyze.py /path/to/video.mp4          # or /path/to/video_results/
```

This computes the bulb center-of-mass per frame, averages the distances from each bulb to that center, and saves both `pulse_distance.png` and `pulse_distance.csv` inside the results folder. Add `--show` to open the plot interactively or `--no-plot`/`--no-csv` to limit the outputs.

> ❗ **Batch mode**: the legacy `batch_process.py` still targets the old flag-based CLI and is currently considered unsupported until a non-interactive configuration path is reintroduced.

## 🌀 Video Types & ROI Expectations

- **Rotating videos** – choose “Yes” when prompted. The adaptive background state machine (STATIC/ROTATING/TRANSITION) activates automatically, the search centers are rotated during motion, and the default ROI is a centered circle unless you draw a custom region.
- Small jitters are ignored: rotation only triggers after several confident consecutive measurements, and any episode with <5° total motion is discarded so tracking stays active on non-rotating clips.

- **Non-rotating videos** – choose “No.” A full-video median background is computed once and reused. If you skip feature sampling, the tracker falls back to the automatic per-video threshold; otherwise the sampler’s slider defines the threshold explicitly.

- **Visualizations** – both the standard and composite MP4s are produced automatically inside the per-video results folder. The composite view reuses the same background/diff data as the rotating pipeline, making it easy to debug illumination changes.

### Testing
```bash
# Test interactive workflow
python test_multi_object_tracking.py interactive video.mp4

# Test batch mode
python test_multi_object_tracking.py batch video.mp4

# Test visualization rendering
python test_multi_object_tracking.py visualization video.mp4 --output labeled.mp4

# Test feature sampling helper
python test_multi_object_tracking.py sampling video.mp4

# Test ROI selector
python test_multi_object_tracking.py roi video.mp4

# Test synced pulse composite
python test_multi_object_tracking.py pulse video.mp4 --output pulse.mp4
```

## 📋 Configuration Options

### Object Type Parameters

#### Mouth Detection
- `mouth_min_area`: Minimum area in pixels (default: 35)
- `mouth_max_area`: Maximum area in pixels (default: 160)  
- `mouth_max_disappeared`: Max frames before track loss (default: 15)
- `mouth_max_distance`: Max tracking distance in pixels (default: 50)
- `mouth_search_radius`: Search radius when mouth lost

#### Gonad Detection  
- `num_gonads`: Number of gonads to detect (0-4)
- `gonad_min_area`: Minimum area in pixels (default: 20)
- `gonad_max_area`: Maximum area in pixels (default: 80)
- `aspect_ratio_min`: Minimum aspect ratio ≥ 1.5 for oblong shapes
- `aspect_ratio_max`: Maximum aspect ratio (default: 3.0)
- `eccentricity_min`: Minimum eccentricity for elongated shapes (default: 0.7)

#### Tentacle Bulb Detection
- `num_tentacle_bulbs`: When omitted, all bulbs found are tracked (auto-detect)
- `bulb_min_area`: Minimum area in pixels (default: 5)
- `bulb_max_area`: Maximum area in pixels (default: 35)
- `aspect_ratio_max`: Maximum aspect ratio for round shapes (default: 1.5)
- `eccentricity_max`: Maximum eccentricity for circular shapes (default: 0.7)

### ROI Configuration
- **Non-rotating videos**: Default ROI is the entire frame. Select a circle, polygon, or bounding box if you want to clip processing to a smaller region.
- **Rotating videos**: ROI defaults to a centered circle, but you can override it with circle/polygon/bounding-box selections. The stored ROI is reused when generating labeled videos.
- Inside the selector: `a` = auto circle, `c` = circle mode, `p` = polygon mode, `r` = reset, `q` = finish. Circle mode still uses three boundary clicks to define the ROI.

## 🎮 Interactive Controls

### Feature Sampling
- `m`: Switch to mouth sampling mode (disabled when the mouth is pinned)
- `g`: Switch to gonad sampling mode  
- `t`: Switch to tentacle bulb sampling mode
- `u`: Undo last sample
- `c`: Clear all samples
- `b`: Cycle the background subtraction threshold and write it back to the tracking parameters
- `n/p`: Next/previous frame
- `q`: Finish sampling (you’ll be asked whether to apply ROI suggestions that were derived from your annotations)

### ROI Selection
- `a`: Auto ROI mode
- `c`: Circle selection mode
- `p`: Polygon selection mode
- `f`: Finish polygon
- `r`: Reset selection
- `q`: Finish ROI selection

## 📊 Output Format

### Multi-Object Zarr Structure
```
results.zarr/
├── mouth/                 # Mouth tracking data
│   ├── track              # Track IDs
│   ├── x, y              # Positions
│   ├── area               # Detection areas
│   ├── major_axis_length_mm
│   └── bbox_*
├── gonad/                 # Gonad tracking data
│   └── [same structure as mouth]
├── tentacle_bulb/         # Bulb tracking data
│   └── [same structure as mouth]
└── .attrs                 # Metadata
    ├── fps               # Frame rate
    ├── object_types       # ['mouth', 'gonad', 'tentacle_bulb']
    └── parameters        # Full tracking configuration
```

### Visualization
- **Mouth**: Orange circles (radius 6)
- **Gonads**: Magenta ellipses (8×4, oblong shape)
- **Tentacle Bulbs**: Cyan circles (radius 3)
- **Track Trails**: Colored lines showing last 20 frames
- **ROI Boundary**: Gray circle/polygon outline
- **Search Radii**: Optional colored circles around reference positions
- **Automatic Outputs**: Every prompt-driven run now saves the `.zarr` store and both visualization videos inside `<video_dir>/<video_name>_results/`. The composite MP4 shows the labeled frame, the current background, and the diff image side by side using the same adaptive background logic as rotating runs.

## 🔄 Backward Compatibility

The system maintains full backward compatibility with existing jf-track workflows:

### Legacy Two-Pass API
```python
from src.tracking import run_two_pass_tracking

# Existing function unchanged
mouth_tracking, bulb_tracking, fps, params = run_two_pass_tracking(
    video_path="video.mp4",
    mouth_min_area=35,
    mouth_max_area=160,
    bulb_min_area=5,
    bulb_max_area=35
)
```

### Automatic System Selection
- **Legacy parameters only** → Uses original two-pass tracking
- **Gonads enabled** → Uses multi-object system
- **Extended configuration** → Uses multi-object system

## 🧪 Testing and Validation

### Unit Tests
```bash
# Test individual components
python -c "from src.interactive_sampling import run_interactive_feature_sampling"
python -c "from src.roi_selector import run_interactive_roi_selection"
```

### Integration Tests
```bash
# Test full pipeline with a sample video (limiting frames for speed)
python test_multi_object_tracking.py interactive sample_video.mp4 --max-frames 200

# Validate results
python -c "
from src.save_results import load_multi_object_tracking_from_zarr
results = load_multi_object_tracking_from_zarr('sample_video_results/multi_object_tracking.zarr')
print(f'Loaded {len(results)} object types')
"
```

## 🔧 Troubleshooting

### Common Issues

1. **No objects detected**
   - Try interactive feature sampling to set correct parameters
   - Adjust the background subtraction threshold inside `TrackingParameters` (or rerun prompts and enable feature sampling)
   - Verify ROI covers the tracking area

2. **Gonads not detected**
   - Ensure aspect_ratio_min ≥ 1.5 for oblong shapes
   - Gonads may be hidden (set `num_gonads=0` if not visible)

3. **Too many false detections**
   - Increase shape filtering constraints
   - Use smaller search radii
   - Adjust area ranges to be more specific

4. **Track fragmentation**
   - Increase `max_disappeared` values
   - Decrease `max_distance` for tighter assignment
   - Check ROI stability (objects leaving ROI)

### Debug Mode
```bash
# Launch the prompt-driven CLI (same as python main.py)
python -m src.cli

# Check intermediate results programmatically
python -c "
from src.tracking import run_multi_object_tracking
from src.tracker import TrackingParameters
params = TrackingParameters()
params.num_gonads = 2
params.update_object_counts()
results = run_multi_object_tracking('video.mp4', params, max_frames=50)
"
```

## 📚 API Reference

### Core Functions

#### `run_multi_object_tracking(video_path, params, max_frames=None)`
Multi-object tracking with configurable object types.

#### `detect_objects_with_shape_filtering(mask, config, pixel_size_mm=0.01)`  
Enhanced detection with aspect ratio, eccentricity, and solidity filtering.

#### `run_interactive_feature_sampling(video_path, params=None)`
Interactive clicking interface for automatic parameter tuning. Pass an existing `TrackingParameters` instance to reuse the current ROI and threshold configuration.

#### `run_interactive_roi_selection(video_path)`
Interactive drawing interface for custom ROI selection.

### Data Classes

#### `TrackingParameters`
Extended parameter class with multi-object configuration:
```python
params = TrackingParameters()
params.num_gonads = 2
params.num_tentacle_bulbs = 8
params.roi_mode = "circle"
params.update_object_counts()
```

#### `SampledFeature`
Feature sampling results with shape properties:
```python
@dataclass
class SampledFeature:
    object_type: str           # 'mouth', 'gonad', 'tentacle_bulb'
    centroid: Tuple[float, float]
    aspect_ratio: float
    eccentricity: float
    # ... other properties
```

## 🎯 Best Practices

### Parameter Tuning
1. **Start with feature sampling** to get good initial parameters
2. **Validate with short clips** (`max_frames=100`) before full processing
3. **Adjust shape filters** based on object morphology
4. **Use appropriate search radii** to constrain detection areas

### Performance Optimization
1. **Limit ROI area** to reduce processing time
2. **Use reasonable frame limits** for testing
3. **Optimize background parameters** for your video conditions
4. **Batch process multiple videos** with same parameters

### Quality Assurance  
1. **Check tracking statistics** (number of tracks, gaps)
2. **Verify visualization** makes sense for your data
3. **Validate object counts** match biological expectations
4. **Test parameter sensitivity** with small variations

## 📈 Development Roadmap

- [ ] Enhanced direction analysis with multi-object vectors
- [ ] Track quality metrics and confidence scoring  
- [ ] Automatic parameter optimization using machine learning
- [ ] Real-time tracking with live video input
- [ ] Web-based configuration interface
- [ ] Export to additional formats (HDF5, CSV)
- [ ] Integration with popular analysis pipelines (TrackMate, Fiji)

## 🤝 Contributing

1. **Test on diverse video datasets** 
2. **Report parameter ranges** for different jellyfish species
3. **Conduct performance benchmarks**
4. **Submit improvement suggestions** and bug reports

## 📄 License

This project maintains the same license as the original jf-track system.

## 📦 Batch Processing Helper

The existing `batch_process.py` script still targets the old flag-based CLI and is awaiting an update. For now, script automation by importing `run_multi_object_tracking()` directly or by writing your own wrapper that supplies parameters without the GUI prompts.
