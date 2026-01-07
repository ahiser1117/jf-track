import cv2
import numpy as np
from skimage.measure import label, regionprops

from src.tracker import RobustTracker, TrackingData
from dataclasses import replace


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
    background_samples: int = 5,
    threshold: int = 10,
    # Mouth (large object) parameters
    mouth_min_area: int = 35,
    mouth_max_area: int = 160,
    mouth_max_disappeared: int = 15,
    mouth_max_distance: int = 50,
    # Bulb (small object) parameters
    bulb_min_area: int = 5,
    bulb_max_area: int = 35,
    bulb_max_disappeared: int = 10,
    bulb_max_distance: int = 30,
) -> tuple[TrackingData, TrackingData, float]:
    """
    Run two-pass tracking: first for the mouth (larger object), then for bulbs (smaller objects).

    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (None for all)
        background_samples: Number of frames to sample for background computation
        threshold: Binary threshold for background subtraction
        mouth_min_area: Minimum area for mouth detection
        mouth_max_area: Maximum area for mouth detection
        mouth_max_disappeared: Max frames mouth can disappear
        mouth_max_distance: Max distance for mouth track association
        bulb_min_area: Minimum area for bulb detection
        bulb_max_area: Maximum area for bulb detection
        bulb_max_disappeared: Max frames bulb can disappear
        bulb_max_distance: Max distance for bulb track association

    Returns:
        mouth_tracking: TrackingData for the mouth
        bulb_tracking: TrackingData for the bulbs
        fps: Video frame rate
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    mouth_tracker = RobustTracker(max_disappeared=mouth_max_disappeared, max_distance=mouth_max_distance)
    bulb_tracker = RobustTracker(max_disappeared=bulb_max_disappeared, max_distance=bulb_max_distance)

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

        # Detect mouth (larger objects)
        mouth_stats = detect_objects_with_stats(mask, min_area=mouth_min_area, max_area=mouth_max_area)
        mouth_tracker.update(mouth_stats)

        # Detect bulbs (smaller objects)
        bulb_stats = detect_objects_with_stats(mask, min_area=bulb_min_area, max_area=bulb_max_area)
        bulb_tracker.update(bulb_stats)

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames")

        if max_frames is not None and frame_idx >= max_frames:
            break

    cap.release()

    # Get data-oriented tracking results
    mouth_tracking = mouth_tracker.get_tracking_data()
    bulb_tracking = bulb_tracker.get_tracking_data()

    print(f"Mouth tracking complete: {mouth_tracking.n_tracks} tracks, {mouth_tracking.n_frames} frames")
    print(f"Bulb tracking complete: {bulb_tracking.n_tracks} tracks, {bulb_tracking.n_frames} frames")

    return mouth_tracking, bulb_tracking, fps


def select_best_mouth_track(mouth_tracking: TrackingData) -> TrackingData:
    """
    Select the single best mouth track from multiple candidates.

    Selection criteria (in order of priority):
    1. Most valid (non-NaN) frames - longest track duration
    2. Tie-breaker: highest average area

    Args:
        mouth_tracking: TrackingData potentially containing multiple mouth tracks

    Returns:
        TrackingData with only the single best mouth track
    """
    if mouth_tracking.n_tracks == 0:
        print("Warning: No mouth tracks found")
        return mouth_tracking

    if mouth_tracking.n_tracks == 1:
        print("Single mouth track found, no selection needed")
        return mouth_tracking

    # Calculate valid frame counts for each track
    valid_counts = np.sum(~np.isnan(mouth_tracking.x), axis=1)

    # Calculate average areas for tie-breaking
    avg_areas = np.nanmean(mouth_tracking.area, axis=1)

    # Find track with most valid frames
    max_valid = np.max(valid_counts)
    candidates = np.where(valid_counts == max_valid)[0]

    if len(candidates) == 1:
        best_idx = candidates[0]
    else:
        # Tie-breaker: highest average area among candidates
        candidate_areas = avg_areas[candidates]
        best_idx = candidates[np.argmax(candidate_areas)]

    best_track_id = mouth_tracking.track_ids[best_idx]
    print(f"Selected mouth track {best_track_id} from {mouth_tracking.n_tracks} candidates")
    print(f"  Valid frames: {valid_counts[best_idx]}, Avg area: {avg_areas[best_idx]:.1f} px")

    # Create new TrackingData with only the selected track
    return TrackingData(
        n_tracks=1,
        n_frames=mouth_tracking.n_frames,
        track_ids=np.array([best_track_id]),
        x=mouth_tracking.x[best_idx:best_idx+1, :],
        y=mouth_tracking.y[best_idx:best_idx+1, :],
        area=mouth_tracking.area[best_idx:best_idx+1, :],
        major_axis_length_mm=mouth_tracking.major_axis_length_mm[best_idx:best_idx+1, :],
        bbox_min_row=mouth_tracking.bbox_min_row[best_idx:best_idx+1, :],
        bbox_min_col=mouth_tracking.bbox_min_col[best_idx:best_idx+1, :],
        bbox_max_row=mouth_tracking.bbox_max_row[best_idx:best_idx+1, :],
        bbox_max_col=mouth_tracking.bbox_max_col[best_idx:best_idx+1, :],
        frame=mouth_tracking.frame[best_idx:best_idx+1, :],
    )
