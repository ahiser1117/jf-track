import numpy as np
import zarr

from src.tracker import TrackingData
from src.direction_analysis import DirectionAnalysis


def save_tracking_to_zarr(tracking_data: TrackingData, output_path: str, fps: float):
    """
    Save tracking data to zarr format.

    All arrays have shape (n_tracks, n_frames) with NaN padding for missing data.

    Args:
        tracking_data: TrackingData object from tracker
        output_path: Path to save zarr store
        fps: Video frame rate
    """
    print(f"Saving results to {output_path}...")

    n_tracks = tracking_data.n_tracks
    n_frames = tracking_data.n_frames

    # Open zarr group with v2 format for xarray compatibility
    root = zarr.open_group(output_path, mode='w', zarr_format=2)

    # Store metadata
    root.attrs['fps'] = fps
    root.attrs['n_tracks'] = n_tracks
    root.attrs['n_frames'] = n_frames

    def create_array_with_dims(name, data, dims=('track', 'frame')):
        arr = root.create_array(name, data=data)
        arr.attrs['_ARRAY_DIMENSIONS'] = list(dims)
        return arr

    # Save track IDs as coordinate
    track_id_arr = root.create_array('track', data=tracking_data.track_ids)
    track_id_arr.attrs['_ARRAY_DIMENSIONS'] = ['track']

    # Save frame indices as coordinate (0 to n_frames-1)
    frame_coord = root.create_array('frame_coord', data=np.arange(n_frames))
    frame_coord.attrs['_ARRAY_DIMENSIONS'] = ['frame']

    # Save raw tracking data (n_tracks, n_frames)
    create_array_with_dims('frame', tracking_data.frame)
    create_array_with_dims('x', tracking_data.x)
    create_array_with_dims('y', tracking_data.y)
    create_array_with_dims('area', tracking_data.area)
    create_array_with_dims('major_axis_length_mm', tracking_data.major_axis_length_mm)
    create_array_with_dims('bbox_min_row', tracking_data.bbox_min_row)
    create_array_with_dims('bbox_min_col', tracking_data.bbox_min_col)
    create_array_with_dims('bbox_max_row', tracking_data.bbox_max_row)
    create_array_with_dims('bbox_max_col', tracking_data.bbox_max_col)

    print(f"Saved {n_tracks} tracks x {n_frames} frames to {output_path}")


def save_two_pass_tracking_to_zarr(
    mouth_tracking: TrackingData,
    bulb_tracking: TrackingData,
    direction_analysis: DirectionAnalysis,
    output_path: str,
    fps: float,
):
    """
    Save two-pass tracking data (mouth + bulbs) and direction analysis to zarr format.

    Args:
        mouth_tracking: TrackingData for the mouth
        bulb_tracking: TrackingData for the bulbs
        direction_analysis: DirectionAnalysis with direction vectors
        output_path: Path to save zarr store
        fps: Video frame rate
    """
    print(f"Saving two-pass tracking results to {output_path}...")

    n_frames = direction_analysis.n_frames

    # Open zarr group with v2 format for xarray compatibility
    root = zarr.open_group(output_path, mode='w', zarr_format=2)

    # Store metadata
    root.attrs['fps'] = fps
    root.attrs['n_mouth_tracks'] = mouth_tracking.n_tracks
    root.attrs['n_bulb_tracks'] = bulb_tracking.n_tracks
    root.attrs['n_frames'] = n_frames

    def create_array_with_dims(name, data, dims):
        arr = root.create_array(name, data=data)
        arr.attrs['_ARRAY_DIMENSIONS'] = list(dims)
        return arr

    # Save frame coordinate
    frame_coord = root.create_array('frame_coord', data=np.arange(n_frames))
    frame_coord.attrs['_ARRAY_DIMENSIONS'] = ['frame']

    # === Save mouth tracking data ===
    mouth_group = root.create_group('mouth')
    mouth_group.attrs['n_tracks'] = mouth_tracking.n_tracks
    mouth_group.attrs['n_frames'] = mouth_tracking.n_frames

    # Mouth track IDs
    mouth_track_arr = mouth_group.create_array('track', data=mouth_tracking.track_ids)
    mouth_track_arr.attrs['_ARRAY_DIMENSIONS'] = ['mouth_track']

    # Mouth tracking arrays (mouth_track, frame)
    def create_mouth_array(name, data):
        arr = mouth_group.create_array(name, data=data)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['mouth_track', 'frame']
        return arr

    create_mouth_array('x', mouth_tracking.x)
    create_mouth_array('y', mouth_tracking.y)
    create_mouth_array('area', mouth_tracking.area)
    create_mouth_array('major_axis_length_mm', mouth_tracking.major_axis_length_mm)
    create_mouth_array('bbox_min_row', mouth_tracking.bbox_min_row)
    create_mouth_array('bbox_min_col', mouth_tracking.bbox_min_col)
    create_mouth_array('bbox_max_row', mouth_tracking.bbox_max_row)
    create_mouth_array('bbox_max_col', mouth_tracking.bbox_max_col)
    create_mouth_array('frame', mouth_tracking.frame)

    # === Save bulb tracking data ===
    bulb_group = root.create_group('bulb')
    bulb_group.attrs['n_tracks'] = bulb_tracking.n_tracks
    bulb_group.attrs['n_frames'] = bulb_tracking.n_frames

    # Bulb track IDs
    bulb_track_arr = bulb_group.create_array('track', data=bulb_tracking.track_ids)
    bulb_track_arr.attrs['_ARRAY_DIMENSIONS'] = ['bulb_track']

    # Bulb tracking arrays (bulb_track, frame)
    def create_bulb_array(name, data):
        arr = bulb_group.create_array(name, data=data)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['bulb_track', 'frame']
        return arr

    create_bulb_array('x', bulb_tracking.x)
    create_bulb_array('y', bulb_tracking.y)
    create_bulb_array('area', bulb_tracking.area)
    create_bulb_array('major_axis_length_mm', bulb_tracking.major_axis_length_mm)
    create_bulb_array('bbox_min_row', bulb_tracking.bbox_min_row)
    create_bulb_array('bbox_min_col', bulb_tracking.bbox_min_col)
    create_bulb_array('bbox_max_row', bulb_tracking.bbox_max_row)
    create_bulb_array('bbox_max_col', bulb_tracking.bbox_max_col)
    create_bulb_array('frame', bulb_tracking.frame)

    # === Save direction analysis ===
    direction_group = root.create_group('direction')
    direction_group.attrs['n_frames'] = direction_analysis.n_frames

    def create_direction_array(name, data):
        arr = direction_group.create_array(name, data=data)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['frame']
        return arr

    create_direction_array('mouth_x', direction_analysis.mouth_x)
    create_direction_array('mouth_y', direction_analysis.mouth_y)
    create_direction_array('bulb_com_x', direction_analysis.bulb_com_x)
    create_direction_array('bulb_com_y', direction_analysis.bulb_com_y)
    create_direction_array('bulb_count', direction_analysis.bulb_count)
    create_direction_array('direction_x', direction_analysis.direction_x)
    create_direction_array('direction_y', direction_analysis.direction_y)
    create_direction_array('direction_magnitude', direction_analysis.direction_magnitude)
    create_direction_array('direction_angle_deg', direction_analysis.direction_angle_deg)

    print(f"Saved mouth ({mouth_tracking.n_tracks} tracks) + bulbs ({bulb_tracking.n_tracks} tracks) x {n_frames} frames")
    print(f"Direction analysis included with {np.sum(~np.isnan(direction_analysis.direction_magnitude))} valid frames")
