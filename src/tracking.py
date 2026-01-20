import cv2
import numpy as np
from skimage.measure import label, regionprops
from src.tracker import RobustTracker, TrackingData, TrackingParameters
from src.adaptive_background import BackgroundProcessor


def rotate_point(point: tuple[float, float], angle_deg: float, center: tuple[float, float]) -> tuple[float, float]:
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


def detect_objects_with_stats(
    binary_mask: np.ndarray,
    pixel_size_mm: float = 0.01,
    min_area: int = 35,
    max_area: int = 160,
) -> list[dict]:
    """
    Detects objects and calculates region properties.

    Args:
        binary_mask: Thresholded boolean or uint8 image (0=bg, 1=object).
        pixel_size_mm: For converting pixel lengths to mm.
        min_area: Minimum area in pixels to consider as valid object.
        max_area: Maximum area in pixels to consider as valid object.

    Returns:
        list: A list of dictionaries containing stats for each detected object.
    """
    # Label connected components
    label_img = label(binary_mask)

    # Extract properties
    props = regionprops(label_img)

    detections = []

    for prop in props:
        area = prop.area

        # Filter by area
        if min_area < area < max_area:
            # Centroid comes as (row, col) -> (y, x). We want (x, y).
            y, x = prop.centroid

            # Major Axis Length (in pixels)
            major_axis = prop.axis_major_length

            detections.append({
                'centroid': (x, y),
                'area': area,
                'major_axis_length_mm': major_axis * pixel_size_mm,
                'bounding_box': prop.bbox,  # (min_row, min_col, max_row, max_col)
            })

    return detections


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
    background = compute_background(video_path, num_samples=background_samples, max_frames=max_frames)
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

        current_stats = detect_objects_with_stats(mask, min_area=min_area, max_area=max_area)

        # Update Tracker
        tracker.update(current_stats)

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    # Get data-oriented tracking results
    tracking_data = tracker.get_tracking_data()
    print(f"Tracking complete: {tracking_data.n_tracks} tracks, {tracking_data.n_frames} frames")

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

    Uses a rolling buffer of recent frames for background computation.

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (None for all)
        background_buffer_size: Number of recent frames to use for background (default 10)
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

    mouth_tracker = RobustTracker(max_disappeared=mouth_max_disappeared, max_distance=mouth_max_distance)
    bulb_tracker = RobustTracker(max_disappeared=bulb_max_disappeared, max_distance=bulb_max_distance)

    # Get video dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize background processor (centralizes background subtraction logic)
    bg_mode = "adaptive" if adaptive_background else "rolling"
    print(f"Initializing {bg_mode} background processor (buffer_size={background_buffer_size})...")
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
    )

    roi_center, roi_radius = bg_processor.get_roi_params()
    print(f"ROI mask: circle at ({roi_center[0]}, {roi_center[1]}) with radius {roi_radius}")

    frame_idx = 0
    last_mouth_position = None  # Track last known mouth position for search radius

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Process frame through background subtraction
        diff, mask, is_ready = bg_processor.process_frame(frame_idx, gray)

        if not is_ready:
            # Not enough frames yet, skip processing
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"Filling background buffer...")
            continue

        # Rotate last mouth position if rotation is detected
        last_mouth_position = bg_processor.rotate_point_if_needed(last_mouth_position)

        # Detect mouth (larger objects)
        mouth_stats = detect_objects_with_stats(mask, min_area=mouth_min_area, max_area=mouth_max_area)

        # Filter mouth detections by distance from last known position if enabled
        if mouth_search_radius is not None and last_mouth_position is not None:
            mouth_stats = [
                m for m in mouth_stats
                if np.linalg.norm(np.array(m['centroid']) - np.array(last_mouth_position)) <= mouth_search_radius
            ]

        mouth_tracker.update(mouth_stats)

        # Get current mouth position for spatial filtering of bulbs and update last known position
        mouth_position = None
        if mouth_tracker.objects:
            # Use the first (and typically only) mouth object
            first_mouth = next(iter(mouth_tracker.objects.values()))
            mouth_position = first_mouth['centroid']
            last_mouth_position = mouth_position  # Update last known position

        # Detect bulbs (smaller objects)
        bulb_stats = detect_objects_with_stats(mask, min_area=bulb_min_area, max_area=bulb_max_area)

        # Filter bulbs by distance from mouth position
        # - If mouth is tracked: use bulb_search_radius from current position
        # - If mouth is lost: use mouth_search_radius from last known position
        if mouth_position is not None and bulb_search_radius is not None:
            bulb_stats = [
                b for b in bulb_stats
                if np.linalg.norm(np.array(b['centroid']) - np.array(mouth_position)) <= bulb_search_radius
            ]
        elif last_mouth_position is not None and mouth_search_radius is not None:
            bulb_stats = [
                b for b in bulb_stats
                if np.linalg.norm(np.array(b['centroid']) - np.array(last_mouth_position)) <= mouth_search_radius
            ]

        bulb_tracker.update(bulb_stats)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    # Report rotation episodes if adaptive background was used
    episodes = bg_processor.get_rotation_episodes()
    if episodes:
        print(f"\nDetected {len(episodes)} rotation episode(s):")
        for i, ep in enumerate(episodes):
            print(f"  Episode {i+1}: frames {ep.start_frame}-{ep.end_frame}, "
                  f"total rotation: {ep.total_rotation_deg:.1f} deg")

    # Get data-oriented tracking results
    mouth_tracking = mouth_tracker.get_tracking_data()
    bulb_tracking = bulb_tracker.get_tracking_data()

    print(f"Mouth tracking complete: {mouth_tracking.n_tracks} tracks, {mouth_tracking.n_frames} frames")
    print(f"Bulb tracking complete: {bulb_tracking.n_tracks} tracks, {bulb_tracking.n_frames} frames")

    return mouth_tracking, bulb_tracking, fps, params


def merge_mouth_tracks(mouth_tracking: TrackingData) -> TrackingData:
    """
    Merge multiple mouth track segments into a single continuous track.

    The mouth may be temporarily lost (e.g., due to occlusion) and reacquired,
    creating multiple non-overlapping track segments. This function links them
    into one unified track.

    For frames where multiple tracks have data (overlap), the track with the
    larger area is used.

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
            # Multiple tracks have data (overlap) - pick the one with largest area
            overlap_frames += 1
            areas = [mouth_tracking.area[t, frame_idx] for t in valid_tracks]
            best_idx = valid_tracks[np.argmax(areas)]

        # Copy data from best track
        tracks_used.add(best_idx)
        merged_x[frame_idx] = mouth_tracking.x[best_idx, frame_idx]
        merged_y[frame_idx] = mouth_tracking.y[best_idx, frame_idx]
        merged_area[frame_idx] = mouth_tracking.area[best_idx, frame_idx]
        merged_major_axis[frame_idx] = mouth_tracking.major_axis_length_mm[best_idx, frame_idx]
        merged_bbox_min_row[frame_idx] = mouth_tracking.bbox_min_row[best_idx, frame_idx]
        merged_bbox_min_col[frame_idx] = mouth_tracking.bbox_min_col[best_idx, frame_idx]
        merged_bbox_max_row[frame_idx] = mouth_tracking.bbox_max_row[best_idx, frame_idx]
        merged_bbox_max_col[frame_idx] = mouth_tracking.bbox_max_col[best_idx, frame_idx]
        merged_frame[frame_idx] = mouth_tracking.frame[best_idx, frame_idx]

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
