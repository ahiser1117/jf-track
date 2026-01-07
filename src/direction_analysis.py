import numpy as np
from dataclasses import dataclass
from scipy.ndimage import uniform_filter1d

from src.tracker import TrackingData


def _smooth_with_nans(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply moving average smoothing while handling NaN values.

    Uses a simple approach: interpolate NaNs, smooth, then restore NaN positions.
    """
    if window_size <= 1:
        return data.copy()

    result = data.copy()
    valid_mask = ~np.isnan(data)

    if np.sum(valid_mask) < 2:
        return result

    # Interpolate NaN values for smoothing
    indices = np.arange(len(data))
    valid_indices = indices[valid_mask]
    valid_values = data[valid_mask]

    # Linear interpolation for gaps
    interpolated = np.interp(indices, valid_indices, valid_values)

    # Apply uniform filter (moving average)
    smoothed = uniform_filter1d(interpolated, size=window_size, mode='nearest')

    # Only keep smoothed values where we had valid data originally
    # (or extend slightly into gaps for continuity)
    result[valid_mask] = smoothed[valid_mask]

    # For frames that were NaN but are surrounded by valid data, use smoothed values
    # This helps with brief dropouts
    for i in range(len(data)):
        if np.isnan(data[i]):
            # Check if there are valid values within window_size on both sides
            left_valid = np.any(valid_mask[max(0, i - window_size):i])
            right_valid = np.any(valid_mask[i + 1:min(len(data), i + window_size + 1)])
            if left_valid and right_valid:
                result[i] = smoothed[i]

    return result


@dataclass
class DirectionAnalysis:
    """
    Per-frame analysis of direction from bulb center of mass to mouth.
    All arrays have shape (n_frames,) with NaN for frames without valid data.
    """
    n_frames: int

    # Mouth position per frame
    mouth_x: np.ndarray
    mouth_y: np.ndarray

    # Bulb center of mass per frame (average of all tracked bulbs)
    bulb_com_x: np.ndarray
    bulb_com_y: np.ndarray
    bulb_count: np.ndarray  # Number of bulbs used for CoM calculation

    # Direction vector from bulb CoM to mouth (not normalized)
    direction_x: np.ndarray
    direction_y: np.ndarray

    # Direction magnitude and angle
    direction_magnitude: np.ndarray
    direction_angle_deg: np.ndarray  # Angle in degrees (0 = up, 90 = right, etc.)


def compute_direction_analysis(
    mouth_tracking: TrackingData,
    bulb_tracking: TrackingData,
    mouth_track_idx: int = 0,
    smooth_window: int = 5,
) -> DirectionAnalysis:
    """
    Compute direction analysis from bulb center of mass to mouth position.

    Args:
        mouth_tracking: TrackingData for the mouth (should typically have 1 track)
        bulb_tracking: TrackingData for the bulbs (multiple tracks)
        mouth_track_idx: Index of the mouth track to use (default 0, the first/primary track)
        smooth_window: Window size for temporal smoothing of positions (default 5 frames).
                       Set to 1 to disable smoothing.

    Returns:
        DirectionAnalysis with per-frame direction data
    """
    n_frames = max(mouth_tracking.n_frames, bulb_tracking.n_frames)

    # Initialize output arrays
    mouth_x = np.full(n_frames, np.nan)
    mouth_y = np.full(n_frames, np.nan)
    bulb_com_x = np.full(n_frames, np.nan)
    bulb_com_y = np.full(n_frames, np.nan)
    bulb_count = np.zeros(n_frames, dtype=np.int32)
    direction_x = np.full(n_frames, np.nan)
    direction_y = np.full(n_frames, np.nan)
    direction_magnitude = np.full(n_frames, np.nan)
    direction_angle_deg = np.full(n_frames, np.nan)

    # Extract mouth position for each frame
    if mouth_tracking.n_tracks > mouth_track_idx:
        mouth_x_data = mouth_tracking.x[mouth_track_idx, :]
        mouth_y_data = mouth_tracking.y[mouth_track_idx, :]
        # Copy data up to available frames
        n_mouth_frames = min(n_frames, len(mouth_x_data))
        mouth_x[:n_mouth_frames] = mouth_x_data[:n_mouth_frames]
        mouth_y[:n_mouth_frames] = mouth_y_data[:n_mouth_frames]

    # Compute bulb center of mass for each frame
    if bulb_tracking.n_tracks > 0:
        bulb_x_all = bulb_tracking.x  # (n_bulb_tracks, n_frames)
        bulb_y_all = bulb_tracking.y

        n_bulb_frames = min(n_frames, bulb_x_all.shape[1])

        for frame_idx in range(n_bulb_frames):
            # Get all valid bulb positions at this frame
            x_vals = bulb_x_all[:, frame_idx]
            y_vals = bulb_y_all[:, frame_idx]

            valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
            n_valid = np.sum(valid_mask)

            if n_valid > 0:
                bulb_com_x[frame_idx] = np.mean(x_vals[valid_mask])
                bulb_com_y[frame_idx] = np.mean(y_vals[valid_mask])
                bulb_count[frame_idx] = n_valid

    # Apply temporal smoothing to positions
    if smooth_window > 1:
        mouth_x_smooth = _smooth_with_nans(mouth_x, smooth_window)
        mouth_y_smooth = _smooth_with_nans(mouth_y, smooth_window)
        bulb_com_x_smooth = _smooth_with_nans(bulb_com_x, smooth_window)
        bulb_com_y_smooth = _smooth_with_nans(bulb_com_y, smooth_window)
    else:
        mouth_x_smooth = mouth_x
        mouth_y_smooth = mouth_y
        bulb_com_x_smooth = bulb_com_x
        bulb_com_y_smooth = bulb_com_y

    # Compute direction vector from smoothed bulb CoM to smoothed mouth
    for frame_idx in range(n_frames):
        if not np.isnan(mouth_x_smooth[frame_idx]) and not np.isnan(bulb_com_x_smooth[frame_idx]):
            dx = mouth_x_smooth[frame_idx] - bulb_com_x_smooth[frame_idx]
            dy = mouth_y_smooth[frame_idx] - bulb_com_y_smooth[frame_idx]

            direction_x[frame_idx] = dx
            direction_y[frame_idx] = dy
            direction_magnitude[frame_idx] = np.sqrt(dx * dx + dy * dy)

            # Compute angle (atan2 gives angle from positive x-axis, counter-clockwise)
            # Convert to degrees, 0 = right, 90 = up
            angle_rad = np.arctan2(-dy, dx)  # Negate y because image y-axis is inverted
            direction_angle_deg[frame_idx] = np.degrees(angle_rad) % 360.0

    # Store smoothed positions in output (for visualization consistency)
    return DirectionAnalysis(
        n_frames=n_frames,
        mouth_x=mouth_x_smooth,
        mouth_y=mouth_y_smooth,
        bulb_com_x=bulb_com_x_smooth,
        bulb_com_y=bulb_com_y_smooth,
        bulb_count=bulb_count,
        direction_x=direction_x,
        direction_y=direction_y,
        direction_magnitude=direction_magnitude,
        direction_angle_deg=direction_angle_deg,
    )


def compute_direction_analysis_from_zarr(
    mouth_zarr_path: str,
    bulb_zarr_path: str,
    mouth_track_id: int | None = None,
    smooth_window: int = 5,
) -> DirectionAnalysis:
    """
    Compute direction analysis from zarr stores.

    Args:
        mouth_zarr_path: Path to zarr store with mouth tracking data
        bulb_zarr_path: Path to zarr store with bulb tracking data
        mouth_track_id: Specific track ID to use for mouth (None = use first track)
        smooth_window: Window size for temporal smoothing of positions (default 5 frames).
                       Set to 1 to disable smoothing.

    Returns:
        DirectionAnalysis with per-frame direction data
    """
    import zarr

    mouth_root = zarr.open_group(mouth_zarr_path, mode='r')
    bulb_root = zarr.open_group(bulb_zarr_path, mode='r')

    mouth_track_ids = np.array(mouth_root['track'])
    mouth_x_all = np.array(mouth_root['x'])
    mouth_y_all = np.array(mouth_root['y'])

    bulb_x_all = np.array(bulb_root['x'])
    bulb_y_all = np.array(bulb_root['y'])

    # Determine mouth track index
    if mouth_track_id is not None:
        mouth_track_idx = int(np.where(mouth_track_ids == mouth_track_id)[0][0])
    else:
        mouth_track_idx = 0

    n_frames = max(mouth_x_all.shape[1], bulb_x_all.shape[1])

    # Initialize output arrays
    mouth_x = np.full(n_frames, np.nan)
    mouth_y = np.full(n_frames, np.nan)
    bulb_com_x = np.full(n_frames, np.nan)
    bulb_com_y = np.full(n_frames, np.nan)
    bulb_count = np.zeros(n_frames, dtype=np.int32)
    direction_x = np.full(n_frames, np.nan)
    direction_y = np.full(n_frames, np.nan)
    direction_magnitude = np.full(n_frames, np.nan)
    direction_angle_deg = np.full(n_frames, np.nan)

    # Extract mouth position
    n_mouth_frames = min(n_frames, mouth_x_all.shape[1])
    mouth_x[:n_mouth_frames] = mouth_x_all[mouth_track_idx, :n_mouth_frames]
    mouth_y[:n_mouth_frames] = mouth_y_all[mouth_track_idx, :n_mouth_frames]

    # Compute bulb center of mass
    n_bulb_frames = min(n_frames, bulb_x_all.shape[1])
    for frame_idx in range(n_bulb_frames):
        x_vals = bulb_x_all[:, frame_idx]
        y_vals = bulb_y_all[:, frame_idx]

        valid_mask = ~np.isnan(x_vals) & ~np.isnan(y_vals)
        n_valid = np.sum(valid_mask)

        if n_valid > 0:
            bulb_com_x[frame_idx] = np.mean(x_vals[valid_mask])
            bulb_com_y[frame_idx] = np.mean(y_vals[valid_mask])
            bulb_count[frame_idx] = n_valid

    # Apply temporal smoothing to positions
    if smooth_window > 1:
        mouth_x_smooth = _smooth_with_nans(mouth_x, smooth_window)
        mouth_y_smooth = _smooth_with_nans(mouth_y, smooth_window)
        bulb_com_x_smooth = _smooth_with_nans(bulb_com_x, smooth_window)
        bulb_com_y_smooth = _smooth_with_nans(bulb_com_y, smooth_window)
    else:
        mouth_x_smooth = mouth_x
        mouth_y_smooth = mouth_y
        bulb_com_x_smooth = bulb_com_x
        bulb_com_y_smooth = bulb_com_y

    # Compute direction vector from smoothed positions
    for frame_idx in range(n_frames):
        if not np.isnan(mouth_x_smooth[frame_idx]) and not np.isnan(bulb_com_x_smooth[frame_idx]):
            dx = mouth_x_smooth[frame_idx] - bulb_com_x_smooth[frame_idx]
            dy = mouth_y_smooth[frame_idx] - bulb_com_y_smooth[frame_idx]

            direction_x[frame_idx] = dx
            direction_y[frame_idx] = dy
            direction_magnitude[frame_idx] = np.sqrt(dx * dx + dy * dy)

            angle_rad = np.arctan2(-dy, dx)
            direction_angle_deg[frame_idx] = np.degrees(angle_rad) % 360.0

    return DirectionAnalysis(
        n_frames=n_frames,
        mouth_x=mouth_x_smooth,
        mouth_y=mouth_y_smooth,
        bulb_com_x=bulb_com_x_smooth,
        bulb_com_y=bulb_com_y_smooth,
        bulb_count=bulb_count,
        direction_x=direction_x,
        direction_y=direction_y,
        direction_magnitude=direction_magnitude,
        direction_angle_deg=direction_angle_deg,
    )
