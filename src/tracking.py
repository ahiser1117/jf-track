import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

from src.tracker import RobustTracker, TrackingData


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


def _extract_midline_skeleton(region_mask: np.ndarray, bbox, n_points: int = 5):
    """
    Get a simple midline skeleton using skeletonization + B-spline sampling.
    Returns a list of (x, y) coordinates in image space with fixed length.
    """
    # region_mask is already cropped to the bounding box
    skeleton = skeletonize(region_mask)
    coords = np.column_stack(np.nonzero(skeleton))  # (row, col)

    # Offset to full-image coordinates
    min_row, min_col, _, _ = bbox
    if len(coords) > 0:
        coords[:, 0] += min_row
        coords[:, 1] += min_col
        points = coords[:, [1, 0]].astype(float)  # (x, y)
    else:
        # Fallback: use bbox center
        cx = (bbox[1] + bbox[3]) / 2.0
        cy = (bbox[0] + bbox[2]) / 2.0
        return [(cx, cy)] * n_points

    # Order points along principal axis for a clean spline fit
    if len(points) > 1:
        centered = points - points.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0]
        ordering = np.argsort(centered @ axis)
        ordered = points[ordering]
    else:
        ordered = points

    # Fit spline if possible, otherwise linear interpolation
    u_existing = np.linspace(0, 1, len(ordered))
    u_sample = np.linspace(0, 1, n_points)

    if len(ordered) >= 4:
        try:
            tck, _ = splprep(ordered.T, s=0, k=min(3, len(ordered) - 1))
            x_new, y_new = splev(u_sample, tck)
            samples = list(zip(x_new, y_new))
        except Exception:
            x_new = np.interp(u_sample, u_existing, ordered[:, 0])
            y_new = np.interp(u_sample, u_existing, ordered[:, 1])
            samples = list(zip(x_new, y_new))
    elif len(ordered) >= 2:
        x_new = np.interp(u_sample, u_existing, ordered[:, 0])
        y_new = np.interp(u_sample, u_existing, ordered[:, 1])
        samples = list(zip(x_new, y_new))
    else:
        samples = [(ordered[0, 0], ordered[0, 1])] * n_points

    return samples


def detect_worms_with_stats(binary_mask, pixel_size_mm=0.01):
    """
    Detects worms and calculates MATLAB-equivalent region properties.
    
    Args:
        binary_mask (np.array): Thresholded boolean or uint8 image (0=bg, 1=worm).
        pixel_size_mm (float): For converting pixel lengths to mm.

    Returns:
        list: A list of dictionaries containing stats for each worm.
    """
    # 1. Label connected components (equivalent to MATLAB's bwconncomp)
    label_img = label(binary_mask)
    
    # 2. Extract properties (equivalent to MATLAB's regionprops)
    # We specifically need 'eccentricity', 'area', 'centroid', 'major_axis_length'
    props = regionprops(label_img)
    
    detections = []
    
    for prop in props:
        area = prop.area
        
        # Filter by area (Heuristic from define_preferences.m or user config)
        # MATLAB: if(STATS(pp).Area <= Prefs.MaxWormArea)
        if 35 < area < 160: 
            
            # MATLAB Eccentricity: Ratio of the distance between the foci of the 
            # ellipse and its major axis length. The value is between 0 and 1.
            # 0 = Circle, 1 = Line segment.
            ecc = prop.eccentricity 
            
            # Centroid comes as (row, col) -> (y, x). We usually want (x, y).
            y, x = prop.centroid
            
            # Major Axis Length (in pixels)
            major_axis = prop.axis_major_length
            
            # Midline skeleton (5 sampled points)
            skeleton_points = _extract_midline_skeleton(prop.image, prop.bbox, n_points=5)

            detections.append({
                'centroid': (x, y),
                'area': area,
                'eccentricity': ecc,
                'major_axis_length_mm': major_axis * pixel_size_mm,
                'bounding_box': prop.bbox,  # (min_row, min_col, max_row, max_col)
                'skeleton': skeleton_points,
            })
            
    return detections

def run_tracking(video_path, max_frames=None, background_samples: int = 5) -> tuple[TrackingData, float]:
    """
    Run tracking on a video and return data-oriented tracking results.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to process (None for all)
    
    Returns:
        tracking_data: TrackingData object with arrays of shape (n_tracks, n_frames)
        fps: Video frame rate
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tracker = RobustTracker(max_disappeared=15, max_distance=50)

    # Calculate background (mean of sampled frames)
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
        _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        current_stats = detect_worms_with_stats(mask)

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
