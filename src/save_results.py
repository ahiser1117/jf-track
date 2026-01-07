import numpy as np
import zarr

from src.tracker import TrackingData
from src.featureExtraction import WormFeatureExtractor, FeatureData


def save_results_to_zarr(tracking_data: TrackingData, feature_data: FeatureData, 
                         output_path: str, fps: float):
    """
    Save tracking and feature data to zarr format.
    
    All arrays have shape (n_tracks, n_frames) with NaN padding for missing data.
    
    Args:
        tracking_data: TrackingData object from tracker
        feature_data: FeatureData object from feature extraction
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
    create_array_with_dims('eccentricity', tracking_data.eccentricity)
    create_array_with_dims('major_axis_length_mm', tracking_data.major_axis_length_mm)
    create_array_with_dims('bbox_min_row', tracking_data.bbox_min_row)
    create_array_with_dims('bbox_min_col', tracking_data.bbox_min_col)
    create_array_with_dims('bbox_max_row', tracking_data.bbox_max_row)
    create_array_with_dims('bbox_max_col', tracking_data.bbox_max_col)
    create_array_with_dims(
        'skeleton_x',
        tracking_data.skeleton_x,
        dims=('track', 'frame', 'skeleton_point'),
    )
    create_array_with_dims(
        'skeleton_y',
        tracking_data.skeleton_y,
        dims=('track', 'frame', 'skeleton_point'),
    )
    
    # Save extracted features (n_tracks, n_frames)
    create_array_with_dims('SmoothX', feature_data.smooth_x)
    create_array_with_dims('SmoothY', feature_data.smooth_y)
    create_array_with_dims('Direction', feature_data.direction)
    create_array_with_dims('Speed', feature_data.speed)
    create_array_with_dims('AngSpeed', feature_data.ang_speed)
    create_array_with_dims('is_Reversal', feature_data.is_reversal)
    create_array_with_dims('is_Omega', feature_data.is_omega)
    create_array_with_dims('is_Upsilon', feature_data.is_upsilon)
    create_array_with_dims('Reversal_Category', feature_data.reversal_category)
    
    print(f"Saved {n_tracks} tracks x {n_frames} frames to {output_path}")


def extract_and_save(tracking_data: TrackingData, output_path: str, 
                     fps: float, pixel_size_mm: float = 0.01):
    """
    Extract features and save everything to zarr.
    
    Args:
        tracking_data: TrackingData object from tracker
        output_path: Path to save zarr store
        fps: Video frame rate
        pixel_size_mm: Pixel size in mm
    """
    # Extract features
    extractor = WormFeatureExtractor(pixel_size_mm=pixel_size_mm, frame_rate=fps)
    feature_data = extractor.extract_features(tracking_data)
    
    # Save to zarr
    save_results_to_zarr(tracking_data, feature_data, output_path, fps)
