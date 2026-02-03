import cv2
import numpy as np
from collections import deque
from collections.abc import Iterable as ABCIterable
from typing import Optional, Tuple, List, Any
from skimage.measure import label, regionprops
from src.tracker import RobustTracker, TrackingData, TrackingParameters
from src.adaptive_background import BackgroundProcessor, RotationState
from src.mask_utils import clean_binary_mask


def rotate_point(
    point: tuple[float, float], angle_deg: float, center: tuple[float, float]
) -> tuple[float, float]:
    """
    Rotate a point around a center by a given angle.

    Args:
        point: (x, y) coordinates to rotate
        angle_deg: Rotation angle in degrees (positive = clockwise)
        center: (cx, cy) center of rotation

    Returns:
        Rotated (x, y) coordinates
    """
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Translate point to origin
    px, py = point[0] - center[0], point[1] - center[1]

    # Rotate (clockwise rotation in screen coordinates where y points down)
    rx = px * cos_a - py * sin_a
    ry = px * sin_a + py * cos_a

    # Translate back
    return (rx + center[0], ry + center[1])


def create_circular_roi_mask(height: int, width: int) -> np.ndarray:
    """
    Create a circular region-of-interest mask.

    The circle is centered at the middle of the frame with radius min(width, height)/2.
    This constrains tracking to the central circular region where the jellyfish is expected.

    Args:
        height: Frame height in pixels
        width: Frame width in pixels

    Returns:
        Binary mask (uint8) where 255 = inside ROI, 0 = outside
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    cv2.circle(mask, center, radius, (255,), -1)
    return mask


def get_roi_params(height: int, width: int) -> tuple[tuple[int, int], int]:
    """
    Get ROI circle parameters for visualization.

    Args:
        height: Frame height in pixels
        width: Frame width in pixels

    Returns:
        (center, radius) where center is (cx, cy)
    """
    center = (width // 2, height // 2)
    radius = min(width, height) // 2
    return center, radius


# --- Main Pipeline ---


def compute_background(
    video_path: str,
    num_samples: int = 10,
    max_frames: int | None = None,
    max_samples: int | None = 20,
) -> np.ndarray:
    """
    Compute an average-frame background for a video by sampling frames.

    Args:
        video_path: Path to video file.
        num_samples: Number of frames to sample evenly across the video (defaults to 10).
        max_frames: Optional cap on frames to consider when spacing samples.

    Returns:
        background image as uint8 array (H, W, 3).
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if max_samples is not None:
        num_samples = min(num_samples, max_samples)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video has no frames")

    frames_to_use = min(total_frames, max_frames) if max_frames else total_frames
    # Evenly spaced indices across the span we're considering
    indices = np.linspace(0, frames_to_use - 1, num_samples).astype(int)

    samples = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            samples.append(frame.astype(np.float32))

    cap.release()

    if not samples:
        raise ValueError("No frames available to compute background")

    background = np.median(samples, axis=0).astype(np.uint8)
    return background


def compute_component_stats(
    binary_mask: np.ndarray,
    pixel_size_mm: float = 0.01,
) -> list[dict]:
    """Compute region properties for all connected components in a mask."""

    label_img = label(binary_mask)
    props = regionprops(label_img)
    components: list[dict] = []

    for idx, prop in enumerate(props):
        area = prop.area
        y, x = prop.centroid

        major_axis_length_px = getattr(
            prop, "axis_major_length", getattr(prop, "major_axis_length", 0.0)
        )
        minor_axis_length_px = getattr(
            prop, "axis_minor_length", getattr(prop, "minor_axis_length", 0.0)
        )

        minor_axis_length_mm = minor_axis_length_px * pixel_size_mm
        major_axis_length_mm = major_axis_length_px * pixel_size_mm

        aspect_ratio = None
        if minor_axis_length_px > 0:
            aspect_ratio = major_axis_length_px / minor_axis_length_px

        orientation_rad = getattr(prop, "orientation", 0.0)
        orientation_deg = float(np.degrees(orientation_rad)) if orientation_rad else 0.0

        components.append(
            {
                "component_id": idx,
                "centroid": (x, y),
                "area": area,
                "major_axis_length_px": major_axis_length_px,
                "minor_axis_length_px": minor_axis_length_px,
                "major_axis_length_mm": major_axis_length_mm,
                "minor_axis_length_mm": minor_axis_length_mm,
                "aspect_ratio": aspect_ratio,
                "eccentricity": getattr(prop, "eccentricity", 0.0),
                "solidity": getattr(prop, "solidity", 0.0),
                "orientation_deg": orientation_deg,
                "bounding_box": prop.bbox,
            }
        )

    return components


def _filter_components_by_config(
    components: list[dict],
    object_config: dict,
    *,
    include_shape_fields: bool,
) -> list[dict]:
    """Filter precomputed components using an object configuration."""

    min_area = object_config.get("min_area", 0)
    max_area = object_config.get("max_area", float("inf"))
    aspect_ratio_min = object_config.get("aspect_ratio_min")
    aspect_ratio_max = object_config.get("aspect_ratio_max")
    eccentricity_min = object_config.get("eccentricity_min")
    eccentricity_max = object_config.get("eccentricity_max")
    solidity_min = object_config.get("solidity_min")
    solidity_max = object_config.get("solidity_max")

    detections: list[dict] = []

    for component in components:
        area = component["area"]
        if not (min_area < area < max_area):
            continue

        aspect_ratio = component["aspect_ratio"]
        if aspect_ratio_min is not None:
            if aspect_ratio is None or aspect_ratio < aspect_ratio_min:
                continue
        if aspect_ratio_max is not None:
            if aspect_ratio is None or aspect_ratio > aspect_ratio_max:
                continue

        if eccentricity_min is not None:
            if component["eccentricity"] < eccentricity_min:
                continue
        if eccentricity_max is not None:
            if component["eccentricity"] > eccentricity_max:
                continue

        if solidity_min is not None:
            if component["solidity"] < solidity_min:
                continue
        if solidity_max is not None:
            if component["solidity"] > solidity_max:
                continue

        detection = {
            "centroid": component["centroid"],
            "area": area,
            "major_axis_length_mm": component["major_axis_length_mm"],
            "bounding_box": component["bounding_box"],
            "component_id": component.get("component_id"),
        }

        if include_shape_fields:
            detection.update(
                {
                    "minor_axis_length_mm": component["minor_axis_length_mm"],
                    "aspect_ratio": component["aspect_ratio"]
                    if component["aspect_ratio"] is not None
                    else float("inf"),
                    "eccentricity": component["eccentricity"],
                    "solidity": component["solidity"],
                    "orientation_deg": component["orientation_deg"],
                }
            )

        detections.append(detection)

    max_count = object_config.get("count")
    if max_count is not None and max_count > 0:
        detections = detections[:max_count]

    return detections


def detect_objects_with_stats(
    binary_mask: np.ndarray | None,
    pixel_size_mm: float = 0.01,
    min_area: int = 35,
    max_area: int = 160,
    components: list[dict] | None = None,
) -> list[dict]:
    """Detect objects using simple area constraints."""

    if components is None:
        if binary_mask is None:
            raise ValueError("binary_mask is required when components are not provided")
        components = compute_component_stats(binary_mask, pixel_size_mm=pixel_size_mm)

    config = {"min_area": min_area, "max_area": max_area}
    return _filter_components_by_config(
        components, config, include_shape_fields=False
    )


def detect_objects_with_shape_filtering(
    binary_mask: np.ndarray | None,
    object_config: dict,
    pixel_size_mm: float = 0.01,
    components: list[dict] | None = None,
) -> list[dict]:
    """
    Enhanced object detection with shape-based filtering.

    Accepts either a binary mask or precomputed component statistics for reuse
    across multiple object types.
    """

    if components is None:
        if binary_mask is None:
            raise ValueError("binary_mask is required when components are not provided")
        components = compute_component_stats(binary_mask, pixel_size_mm=pixel_size_mm)

    config = {
        "min_area": object_config.get("min_area", 5),
        "max_area": object_config.get("max_area", 160),
        "aspect_ratio_min": object_config.get("aspect_ratio_min"),
        "aspect_ratio_max": object_config.get("aspect_ratio_max"),
        "eccentricity_min": object_config.get("eccentricity_min"),
        "eccentricity_max": object_config.get("eccentricity_max"),
        "solidity_min": object_config.get("solidity_min"),
        "solidity_max": object_config.get("solidity_max"),
        "count": object_config.get("count"),
    }

    return _filter_components_by_config(
        components, config, include_shape_fields=True
    )


def run_tracking(
    video_path: str,
    max_frames: int | None = None,
    background_samples: int = 5,
    min_area: int = 35,
    max_area: int = 160,
    threshold: int = 10,
    max_disappeared: int = 15,
    max_distance: int = 50,
) -> tuple[TrackingData, float]:
    """
    Run tracking on a video and return data-oriented tracking results.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (None for all)
        background_samples: Number of frames to sample for background computation
        min_area: Minimum object area in pixels
        max_area: Maximum object area in pixels
        threshold: Binary threshold for background subtraction
        max_disappeared: Maximum frames a track can disappear before being removed
        max_distance: Maximum distance for track association

    Returns:
        tracking_data: TrackingData object with arrays of shape (n_tracks, n_frames)
        fps: Video frame rate
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = RobustTracker(max_disappeared=max_disappeared, max_distance=max_distance)

    # Calculate background (median of sampled frames)
    print("Calculating background...")
    background = compute_background(
        video_path, num_samples=background_samples, max_frames=max_frames
    )
    gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        frame_idx += 1
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Segmentation
        diff = cv2.absdiff(gray, gray_bg)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        current_stats = detect_objects_with_stats(
            mask, min_area=min_area, max_area=max_area
        )

        # Update Tracker
        tracker.update(current_stats)

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    # Get data-oriented tracking results
    tracking_data = tracker.get_tracking_data()
    print(
        f"Tracking complete: {tracking_data.n_tracks} tracks, {tracking_data.n_frames} frames"
    )

    return tracking_data, fps


def run_two_pass_tracking(
    video_path: str,
    max_frames: int | None = None,
    background_buffer_size: int = 10,
    threshold: int = 10,
    # Mouth (large object) parameters
    mouth_min_area: int = 35,
    mouth_max_area: int = 160,
    mouth_max_disappeared: int = 15,
    mouth_max_distance: int = 50,
    mouth_search_radius: int | None = None,
    # Bulb (small object) parameters
    bulb_min_area: int = 5,
    bulb_max_area: int = 35,
    bulb_max_disappeared: int = 10,
    bulb_max_distance: int = 30,
    bulb_search_radius: int | None = None,
    # Adaptive background parameters (for rotating backgrounds)
    adaptive_background: bool = False,
    rotation_start_threshold_deg: float = 0.01,
    rotation_stop_threshold_deg: float = 0.005,
    rotation_center: tuple[float, float] | None = None,
) -> tuple[TrackingData, TrackingData, float, TrackingParameters]:
    """
    Run two-pass tracking: first for the mouth (larger object), then for bulbs (smaller objects).

    Uses a per-episode background built from evenly spaced frames. When using
    adaptive background, tracking is performed only during STATIC episodes.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (None for all)
        background_buffer_size: Number of frames sampled per episode for background (default 10)
        threshold: Binary threshold for background subtraction
        mouth_min_area: Minimum area for mouth detection
        mouth_max_area: Maximum area for mouth detection
        mouth_max_disappeared: Max frames mouth can disappear
        mouth_max_distance: Max distance for mouth track association
        mouth_search_radius: Max distance from last known position to search for mouth (None = no limit)
        bulb_min_area: Minimum area for bulb detection
        bulb_max_area: Maximum area for bulb detection
        bulb_max_disappeared: Max frames bulb can disappear
        bulb_max_distance: Max distance for bulb track association
        bulb_search_radius: Max distance from mouth to consider bulbs when mouth is tracked (None = no limit).
            When mouth is lost, mouth_search_radius is used for bulb filtering instead.
        adaptive_background: Enable rotation-compensated background subtraction
        rotation_start_threshold_deg: Degrees/frame to trigger rotation detection
        rotation_stop_threshold_deg: Degrees/frame to consider rotation stopped
        rotation_center: Fixed rotation center (cx, cy), or None for auto-detection

    Returns:
        mouth_tracking: TrackingData for the mouth
        bulb_tracking: TrackingData for the bulbs
        fps: Video frame rate
        parameters: TrackingParameters used for this run
    """
    # Build tracking parameters
    params = TrackingParameters(
        background_buffer_size=background_buffer_size,
        threshold=threshold,
        mouth_min_area=mouth_min_area,
        mouth_max_area=mouth_max_area,
        mouth_max_disappeared=mouth_max_disappeared,
        mouth_max_distance=mouth_max_distance,
        mouth_search_radius=mouth_search_radius,
        bulb_min_area=bulb_min_area,
        bulb_max_area=bulb_max_area,
        bulb_max_disappeared=bulb_max_disappeared,
        bulb_max_distance=bulb_max_distance,
        bulb_search_radius=bulb_search_radius,
        adaptive_background=adaptive_background,
        rotation_start_threshold_deg=rotation_start_threshold_deg,
        rotation_stop_threshold_deg=rotation_stop_threshold_deg,
        rotation_center=rotation_center,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    mouth_tracker = RobustTracker(
        max_disappeared=mouth_max_disappeared, max_distance=mouth_max_distance
    )
    bulb_tracker = RobustTracker(
        max_disappeared=bulb_max_disappeared, max_distance=bulb_max_distance
    )

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize background processor (centralizes background subtraction logic)
    bg_mode = "adaptive" if adaptive_background else "rolling"
    print(
        f"Initializing {bg_mode} background processor (buffer_size={background_buffer_size})..."
    )
    bg_processor = BackgroundProcessor(
        video_path=video_path,
        width=frame_width,
        height=frame_height,
        background_buffer_size=background_buffer_size,
        threshold=threshold,
        adaptive_background=adaptive_background,
        rotation_start_threshold_deg=rotation_start_threshold_deg,
        rotation_stop_threshold_deg=rotation_stop_threshold_deg,
        rotation_center=rotation_center,
        max_frames=max_frames,
        roi_mask=None,
        use_auto_threshold=False,
    )

    roi_center, roi_radius = bg_processor.get_roi_params()
    print(
        f"ROI mask: circle at ({roi_center[0]}, {roi_center[1]}) with radius {roi_radius}"
    )

    frame_idx = 0
    last_mouth_position = None  # Track last known mouth position for search radius
    prev_state = bg_processor.get_state()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process frame through background subtraction
        diff, mask, is_ready = bg_processor.process_frame(frame_idx, gray)
        state = bg_processor.get_state()

        if state != prev_state:
            if state in (RotationState.ROTATING, RotationState.TRANSITION):
                mouth_tracker.objects.clear()
                mouth_tracker.disappeared.clear()
                bulb_tracker.objects.clear()
                bulb_tracker.disappeared.clear()
                last_mouth_position = None
            elif state == RotationState.STATIC:
                last_mouth_position = None
            prev_state = state

        if not is_ready:
            # Not enough frames yet, skip processing
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Filling background buffer...")
            if max_frames is not None and frame_idx >= max_frames:
                break
            continue

        if state != RotationState.STATIC:
            # Skip tracking during rotation/transition episodes
            mouth_tracker.update([])
            bulb_tracker.update([])
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"Processed {frame_idx} frames")

            if max_frames is not None and frame_idx >= max_frames:
                break

            continue

        component_stats = compute_component_stats(mask)

        # Detect mouth (larger objects)
        mouth_stats = detect_objects_with_stats(
            None,
            min_area=mouth_min_area,
            max_area=mouth_max_area,
            components=component_stats,
        )

        # Filter mouth detections by distance from last known position if enabled
        if mouth_search_radius is not None and last_mouth_position is not None:
            mouth_stats = [
                m
                for m in mouth_stats
                if np.linalg.norm(
                    np.array(m["centroid"]) - np.array(last_mouth_position)
                )
                <= mouth_search_radius
            ]

        mouth_tracker.update(mouth_stats)

        # Get current mouth position for spatial filtering of bulbs and update last known position
        mouth_position = None
        if mouth_tracker.objects:
            # Use the first (and typically only) mouth object
            first_mouth = next(iter(mouth_tracker.objects.values()))
            mouth_position = first_mouth["centroid"]
            last_mouth_position = mouth_position  # Update last known position

        # Detect bulbs (smaller objects)
        bulb_stats = detect_objects_with_stats(
            None,
            min_area=bulb_min_area,
            max_area=bulb_max_area,
            components=component_stats,
        )

        # Filter bulbs by distance from mouth position
        # - If mouth is tracked: use bulb_search_radius from current position
        # - If mouth is lost: use mouth_search_radius from last known position
        if mouth_position is not None and bulb_search_radius is not None:
            bulb_stats = [
                b
                for b in bulb_stats
                if np.linalg.norm(np.array(b["centroid"]) - np.array(mouth_position))
                <= bulb_search_radius
            ]
        elif last_mouth_position is not None and mouth_search_radius is not None:
            bulb_stats = [
                b
                for b in bulb_stats
                if np.linalg.norm(
                    np.array(b["centroid"]) - np.array(last_mouth_position)
                )
                <= mouth_search_radius
            ]

        bulb_tracker.update(bulb_stats)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    # Report rotation episodes if adaptive background was used
    rotation_method = getattr(bg_processor, "get_rotation_episodes", None)
    if callable(rotation_method):
        raw_episodes = rotation_method()
        if isinstance(raw_episodes, ABCIterable):
            episodes = list(raw_episodes)
        elif raw_episodes:
            episodes = [raw_episodes]
        else:
            episodes = []
        if episodes:
            print(f"\nDetected {len(episodes)} rotation episode(s):")
            for i, ep in enumerate(episodes):
                start = getattr(ep, "start_frame", "?")
                end = getattr(ep, "end_frame", "?")
                rotation_value = getattr(ep, "total_rotation_deg", None)
                if isinstance(rotation_value, (int, float)):
                    rotation_text = f"{rotation_value:.1f}"
                else:
                    rotation_text = str(rotation_value)
                print(
                    f"  Episode {i + 1}: frames {start}-{end}, "
                    f"total rotation: {rotation_text} deg"
                )

    # Get data-oriented tracking results
    mouth_tracking = mouth_tracker.get_tracking_data()
    bulb_tracking = bulb_tracker.get_tracking_data()

    print(
        f"Mouth tracking complete: {mouth_tracking.n_tracks} tracks, {mouth_tracking.n_frames} frames"
    )
    print(
        f"Bulb tracking complete: {bulb_tracking.n_tracks} tracks, {bulb_tracking.n_frames} frames"
    )

    return mouth_tracking, bulb_tracking, fps, params


def merge_mouth_tracks(mouth_tracking: TrackingData) -> TrackingData:
    """
    Merge multiple mouth track segments into a single continuous track.

    The mouth may be temporarily lost (e.g., due to occlusion) and reacquired,
    creating multiple non-overlapping track segments. This function links them
    into one unified track.

    For frames where multiple tracks have data (overlap), the track nearest to
    the last known mouth position is preferred. If there is no prior position,
    the track with the larger area is used.

    Args:
        mouth_tracking: TrackingData potentially containing multiple mouth track segments

    Returns:
        TrackingData with a single merged mouth track
    """
    if mouth_tracking.n_tracks == 0:
        print("Warning: No mouth tracks found")
        return mouth_tracking

    if mouth_tracking.n_tracks == 1:
        print("Single mouth track found, no merging needed")
        return mouth_tracking

    n_frames = mouth_tracking.n_frames
    n_tracks = mouth_tracking.n_tracks

    # Initialize merged arrays with NaN
    merged_x = np.full(n_frames, np.nan)
    merged_y = np.full(n_frames, np.nan)
    merged_area = np.full(n_frames, np.nan)
    merged_major_axis = np.full(n_frames, np.nan)
    merged_bbox_min_row = np.full(n_frames, np.nan)
    merged_bbox_min_col = np.full(n_frames, np.nan)
    merged_bbox_max_row = np.full(n_frames, np.nan)
    merged_bbox_max_col = np.full(n_frames, np.nan)
    merged_frame = np.full(n_frames, np.nan)

    # Track statistics for reporting
    tracks_used = set()
    overlap_frames = 0
    last_mouth_position = None

    # For each frame, pick the best track data
    for frame_idx in range(n_frames):
        # Find all tracks with valid data at this frame
        valid_tracks = []
        for track_idx in range(n_tracks):
            if not np.isnan(mouth_tracking.x[track_idx, frame_idx]):
                valid_tracks.append(track_idx)

        if len(valid_tracks) == 0:
            # No track has data for this frame
            continue
        elif len(valid_tracks) == 1:
            # Exactly one track has data - use it
            best_idx = valid_tracks[0]
        else:
            # Multiple tracks have data (overlap)
            overlap_frames += 1
            if last_mouth_position is None:
                areas = [mouth_tracking.area[t, frame_idx] for t in valid_tracks]
                best_idx = valid_tracks[np.argmax(areas)]
            else:
                distances = [
                    np.linalg.norm(
                        np.array(
                            (
                                mouth_tracking.x[t, frame_idx],
                                mouth_tracking.y[t, frame_idx],
                            )
                        )
                        - np.array(last_mouth_position)
                    )
                    for t in valid_tracks
                ]
                best_idx = valid_tracks[np.argmin(distances)]

        # Copy data from best track
        tracks_used.add(best_idx)
        merged_x[frame_idx] = mouth_tracking.x[best_idx, frame_idx]
        merged_y[frame_idx] = mouth_tracking.y[best_idx, frame_idx]
        merged_area[frame_idx] = mouth_tracking.area[best_idx, frame_idx]
        merged_major_axis[frame_idx] = mouth_tracking.major_axis_length_mm[
            best_idx, frame_idx
        ]
        merged_bbox_min_row[frame_idx] = mouth_tracking.bbox_min_row[
            best_idx, frame_idx
        ]
        merged_bbox_min_col[frame_idx] = mouth_tracking.bbox_min_col[
            best_idx, frame_idx
        ]
        merged_bbox_max_row[frame_idx] = mouth_tracking.bbox_max_row[
            best_idx, frame_idx
        ]
        merged_bbox_max_col[frame_idx] = mouth_tracking.bbox_max_col[
            best_idx, frame_idx
        ]
        merged_frame[frame_idx] = mouth_tracking.frame[best_idx, frame_idx]
        last_mouth_position = (merged_x[frame_idx], merged_y[frame_idx])

    # Report what was merged
    valid_count = np.sum(~np.isnan(merged_x))
    track_ids_used = [mouth_tracking.track_ids[i] for i in sorted(tracks_used)]
    print(f"Merged {len(tracks_used)} mouth track segments: {track_ids_used}")
    print(f"  Total valid frames: {valid_count}, Overlap frames: {overlap_frames}")

    # Find gaps in the merged track
    valid_mask = ~np.isnan(merged_x)
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        gaps = []
        for i in range(len(valid_indices) - 1):
            gap_start = valid_indices[i]
            gap_end = valid_indices[i + 1]
            if gap_end - gap_start > 1:
                gaps.append((gap_start + 1, gap_end - 1, gap_end - gap_start - 1))
        if gaps:
            print(f"  Gaps in merged track: {len(gaps)}")
            for start, end, length in gaps[:5]:  # Show first 5 gaps
                print(f"    Frames {start}-{end} ({length} frames)")
            if len(gaps) > 5:
                print(f"    ... and {len(gaps) - 5} more gaps")

    # Create new TrackingData with merged track
    return TrackingData(
        n_tracks=1,
        n_frames=n_frames,
        track_ids=np.array([0]),  # New unified track ID
        x=merged_x.reshape(1, -1),
        y=merged_y.reshape(1, -1),
        area=merged_area.reshape(1, -1),
        major_axis_length_mm=merged_major_axis.reshape(1, -1),
        bbox_min_row=merged_bbox_min_row.reshape(1, -1),
        bbox_min_col=merged_bbox_min_col.reshape(1, -1),
        bbox_max_row=merged_bbox_max_row.reshape(1, -1),
        bbox_max_col=merged_bbox_max_col.reshape(1, -1),
        frame=merged_frame.reshape(1, -1),
    )


# Keep old name as alias for backwards compatibility
select_best_mouth_track = merge_mouth_tracks


def run_multi_object_tracking(
    video_path: str,
    params: TrackingParameters,
    max_frames: int | None = None,
    pixel_size_mm: float = 0.01,
) -> tuple[dict[str, TrackingData], float]:
    """
    Multi-object tracking pipeline with configurable object types.
    
    Supports mouth, gonads, and tentacle bulbs with individual detection parameters
    and shape-based filtering. Uses a rolling background for subtraction.
    
    Args:
        video_path: Path to video file
        params: TrackingParameters with multi-object configuration
        max_frames: Maximum number of frames to process (None for all)
        pixel_size_mm: For converting pixel lengths to mm
    
    Returns:
        Tuple of (tracking_results, fps) where tracking_results maps object
        types to TrackingData and fps is the video frame rate.
    """
    print(f"Starting multi-object tracking on: {video_path}")
    video_type = getattr(params, "video_type", "non_rotating") or "non_rotating"
    video_type = video_type.lower()
    
    # Get enabled object types
    enabled_types = params.get_enabled_object_types()
    if not enabled_types:
        raise ValueError("No object types enabled in tracking parameters")
    
    print(f"Enabled object types: {', '.join(enabled_types)}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get frame dimensions for ROI mask
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    roi_mask = get_roi_mask_for_video(params, height, width, video_type)

    # Initialize background processor
    params.adaptive_background = video_type == "rotating"
    bg_processor = BackgroundProcessor(
        video_path=video_path,
        width=width,
        height=height,
        background_buffer_size=params.background_buffer_size,
        threshold=params.threshold,
        adaptive_background=params.adaptive_background,
        rotation_start_threshold_deg=params.rotation_start_threshold_deg,
        rotation_stop_threshold_deg=params.rotation_stop_threshold_deg,
        rotation_center=params.rotation_center,
        max_frames=max_frames,
        roi_mask=roi_mask,
        use_auto_threshold=params.use_auto_threshold and not params.adaptive_background,
    )
    if not params.adaptive_background and params.use_auto_threshold:
        params.threshold = bg_processor.threshold
        print(
            f"Using auto threshold {params.threshold} for non-rotating background subtraction"
        )
    
    # Initialize trackers for each enabled object type
    trackers = {}
    for obj_type in enabled_types:
        config = params.get_object_config(obj_type)
        if config is None:
            raise ValueError(f"No configuration found for object type: {obj_type}")
        trackers[obj_type] = RobustTracker(
            max_disappeared=config["max_disappeared"],
            max_distance=config["max_distance"]
        )

    smoothing_window = max(1, params.temporal_smoothing_window or 1)
    position_history: dict[str, dict[int, deque]] = {
        obj_type: {} for obj_type in enabled_types
    }
    smoothed_positions_cache: dict[str, list[tuple[float, float]]] = {
        obj_type: [] for obj_type in enabled_types
    }
    position_history.setdefault("mouth", {})
    smoothed_positions_cache.setdefault("mouth", [])

    pinned_reference = params.pinned_mouth_point if params.mouth_pinned else None
    if pinned_reference is not None:
        smoothed_positions_cache.setdefault("mouth", []).append(pinned_reference)

    def _distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        return float(np.linalg.norm(np.array(point_a) - np.array(point_b)))

    def _range_score(value, minimum, maximum):
        if value is None:
            return 1.0
        if minimum is None and maximum is None:
            return 1.0
        low = minimum if minimum is not None else value
        high = maximum if maximum is not None else value
        if high == low:
            return 1.0 if value == low else 0.0
        if value < low:
            return max(0.0, 1.0 - (low - value) / (high - low))
        if value > high:
            return max(0.0, 1.0 - (value - high) / (high - low))
        center = (low + high) / 2.0
        half_span = (high - low) / 2.0 or 1.0
        return max(0.1, 1.0 - abs(value - center) / half_span)

    def _reset_histories() -> None:
        for tracker in trackers.values():
            tracker.objects.clear()
            tracker.disappeared.clear()
        for history_map in position_history.values():
            history_map.clear()
        for key in list(smoothed_positions_cache.keys()):
            smoothed_positions_cache[key] = []
        if pinned_reference is not None:
            smoothed_positions_cache["mouth"] = [pinned_reference]

    def _update_position_history(obj_type: str) -> None:
        tracker = trackers[obj_type]
        history_map = position_history[obj_type]
        active_ids = set(tracker.objects.keys())
        for object_id in list(history_map.keys()):
            if object_id not in active_ids:
                history_map.pop(object_id, None)
        for object_id, detection in tracker.objects.items():
            if detection is None:
                continue
            history = history_map.setdefault(object_id, deque(maxlen=smoothing_window))
            history.append(detection["centroid"])

    def _get_smoothed_positions(obj_type: str) -> list[tuple[float, float]]:
        positions: list[tuple[float, float]] = []
        for history in position_history[obj_type].values():
            if not history:
                continue
            arr = np.array(history, dtype=float)
            mean = arr.mean(axis=0)
            positions.append((float(mean[0]), float(mean[1])))
        return positions

    def _nearest_distance(point: tuple[float, float], positions: list[tuple[float, float]]) -> float:
        if not positions:
            return float("inf")
        return min(_distance(point, pos) for pos in positions)

    def _get_reference_positions(name: str | None) -> list[tuple[float, float]]:
        if not name:
            return []
        positions = smoothed_positions_cache.get(name, [])
        if name == "mouth" and (not positions) and pinned_reference is not None:
            return [pinned_reference]
        return positions

    def _compute_detection_score(obj_type: str, detection: dict, config: dict) -> float:
        score = 1.0
        score *= _range_score(detection.get("area"), config.get("min_area"), config.get("max_area"))
        score *= _range_score(
            detection.get("aspect_ratio"),
            config.get("aspect_ratio_min"),
            config.get("aspect_ratio_max"),
        )
        score *= _range_score(
            detection.get("eccentricity"),
            config.get("eccentricity_min"),
            config.get("eccentricity_max"),
        )
        score *= _range_score(
            detection.get("solidity"),
            config.get("solidity_min"),
            config.get("solidity_max"),
        )

        centroid = detection.get("centroid")
        if centroid is None:
            return 0.0

        active_positions = smoothed_positions_cache.get(obj_type, [])
        track_radius = config.get("track_search_radius")
        if active_positions and track_radius:
            dist = _nearest_distance(centroid, active_positions)
            if dist > track_radius:
                return 0.0
            score *= max(0.1, 1.0 - dist / max(track_radius, 1e-6))
        else:
            search_radius = config.get("search_radius")
            ref_name = config.get("reference_object")
            if search_radius and ref_name:
                ref_positions = _get_reference_positions(ref_name)
                if ref_positions:
                    dist = _nearest_distance(centroid, ref_positions)
                    if dist > search_radius:
                        return 0.0
                    score *= max(0.1, 1.0 - dist / max(search_radius, 1e-6))

        ownership_radius = config.get("ownership_radius")
        if active_positions and ownership_radius:
            dist = _nearest_distance(centroid, active_positions)
            score *= max(0.1, 1.0 - dist / max(ownership_radius, 1e-6))

        exclude_objects = config.get("exclude_objects") or {}
        for other_type, min_distance in exclude_objects.items():
            ref_positions = _get_reference_positions(other_type)
            if not ref_positions or min_distance is None:
                continue
            dist = _nearest_distance(centroid, ref_positions)
            if dist < min_distance:
                penalty_weight = config.get("overlap_penalty_weight", 0.5)
                score *= max(0.0, penalty_weight * (dist / max(min_distance, 1e-6)))

        return score

    def _build_priority_order() -> list[str]:
        order: list[str] = []
        seen: set[str] = set()
        for name in params.object_priority:
            if name in enabled_types and name not in seen:
                order.append(name)
                seen.add(name)
        for name in enabled_types:
            if name not in seen:
                order.append(name)
        return order

    priority_order = _build_priority_order()
    enabled_types = priority_order

    # Tracking variables
    frame_idx = 0
    prev_state = bg_processor.get_state()

    print("Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames is not None and frame_idx >= max_frames:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff, mask, is_ready = bg_processor.process_frame(frame_idx, gray)
        
        state = bg_processor.get_state()
        if state != prev_state:
            if state in (RotationState.ROTATING, RotationState.TRANSITION):
                _reset_histories()
            elif state == RotationState.STATIC:
                _reset_histories()
            prev_state = state

        if not is_ready:
            for tracker in trackers.values():
                tracker.update([])
            frame_idx += 1
            continue

        for key, positions in list(smoothed_positions_cache.items()):
            rotated_positions: list[tuple[float, float]] = []
            for pos in positions:
                rotated = bg_processor.rotate_point_if_needed(pos)
                rotated_positions.append(rotated if rotated is not None else pos)
            smoothed_positions_cache[key] = rotated_positions

        if pinned_reference is not None and not smoothed_positions_cache.get("mouth"):
            smoothed_positions_cache["mouth"] = [pinned_reference]

        component_stats = compute_component_stats(mask, pixel_size_mm=pixel_size_mm)

        component_candidates: dict[str, list[dict]] = {}
        component_scores: dict[int, dict[str, float]] = {}
        for obj_type in priority_order:
            config = params.get_object_config(obj_type)
            if config is None:
                continue
            detections = detect_objects_with_shape_filtering(
                None,
                config,
                pixel_size_mm,
                components=component_stats,
            )
            scored: list[dict] = []
            for det in detections:
                candidate = det.copy()
                score = _compute_detection_score(obj_type, candidate, config)
                if score <= 0:
                    continue
                candidate["score"] = score
                comp_id = candidate.get("component_id")
                if comp_id is not None:
                    component_scores.setdefault(comp_id, {})[obj_type] = score
                scored.append(candidate)
            component_candidates[obj_type] = scored

        claimed_components: set[int] = set()
        for obj_type in priority_order:
            config = params.get_object_config(obj_type)
            if config is None:
                continue
            detections: list[dict] = []
            score_margin = config.get("score_margin", 0.0) or 0.0
            for det in component_candidates.get(obj_type, []):
                comp_id = det.get("component_id")
                if comp_id is not None and comp_id in claimed_components:
                    continue
                comp_map = component_scores.get(comp_id, {}) if comp_id is not None else {}
                if comp_map:
                    best_type, best_score = max(comp_map.items(), key=lambda kv: kv[1])
                    if best_type != obj_type:
                        current_score = det["score"]
                        if best_score >= current_score * (1.0 + score_margin):
                            continue
                detections.append(det)

            trackers[obj_type].update(detections)
            _update_position_history(obj_type)
            smoothed_positions_cache[obj_type] = _get_smoothed_positions(obj_type)
            if obj_type == "mouth" and not smoothed_positions_cache[obj_type] and pinned_reference is not None:
                smoothed_positions_cache["mouth"] = [pinned_reference]

            for detection in trackers[obj_type].objects.values():
                if detection is None:
                    continue
                comp_id = detection.get("component_id")
                if comp_id is not None:
                    claimed_components.add(comp_id)
        
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")
    
    cap.release()
    
    # Report rotation episodes if adaptive background was used
    get_rotation_episodes = getattr(bg_processor, "get_rotation_episodes", None)
    if params.adaptive_background and callable(get_rotation_episodes):
        episodes_obj = get_rotation_episodes()
        if isinstance(episodes_obj, ABCIterable):
            episodes = list(episodes_obj)
        elif episodes_obj:
            episodes = [episodes_obj]
        else:
            episodes = []
        if episodes:
            print(f"\nDetected {len(episodes)} rotation episode(s):")
            for i, ep in enumerate(episodes):
                start = getattr(ep, "start_frame", "?")
                end = getattr(ep, "end_frame", "?")
                rotation_value = getattr(ep, "total_rotation_deg", None)
                if isinstance(rotation_value, (int, float)):
                    rotation_text = f"{rotation_value:.1f}"
                else:
                    rotation_text = str(rotation_value)
                print(
                    f"  Episode {i + 1}: frames {start}-{end}, "
                    f"total rotation: {rotation_text} deg"
                )
    
    # Collect tracking results
    results = {}
    for obj_type, tracker in trackers.items():
        tracking_data = tracker.get_tracking_data()
        results[obj_type] = tracking_data
        print(
            f"{obj_type.capitalize()} tracking complete: "
            f"{tracking_data.n_tracks} tracks, {tracking_data.n_frames} frames"
        )
    
    return results, fps


def create_roi_mask(height: int, width: int, roi_mode: str = "auto",
                  center: Optional[Tuple[float, float]] = None,
                  radius: Optional[float] = None,
                  points: Optional[List[Tuple[float, float]]] = None,
                  bbox: Optional[Tuple[float, float, float, float]] = None) -> np.ndarray:
    """
    Create ROI mask based on mode and parameters.
    
    This is a convenience wrapper around the roi_selector module functions.
    """
    from .roi_selector import create_roi_mask as roi_create
    
    return roi_create(height, width, roi_mode, center, radius, points, bbox)


def get_roi_mask_for_video(
    params: TrackingParameters,
    height: int,
    width: int,
    video_type: str,
) -> np.ndarray:
    """Build an ROI mask from tracking parameters and video type."""

    roi_params = params.get_roi_params()
    roi_mode = (roi_params.get("mode") or "auto").lower()

    if roi_mode in {"circle", "polygon", "bounding_box"}:
        roi_kwargs = {
            "height": height,
            "width": width,
            "roi_mode": roi_mode,
            "center": roi_params.get("center"),
            "radius": roi_params.get("radius"),
            "points": roi_params.get("points"),
        }
        bbox_value = roi_params.get("bbox")
        if bbox_value is not None:
            roi_kwargs["bbox"] = bbox_value
        return create_roi_mask(**roi_kwargs)

    if video_type.lower() == "rotating":
        center = (width / 2.0, height / 2.0)
        radius = min(width, height) / 2.0
        return create_roi_mask(
            height,
            width,
            "circle",
            center=center,
            radius=radius,
        )

    # Non-rotating default: full frame
    return np.full((height, width), 255, dtype=np.uint8)
