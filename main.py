from src.collision_merging import apply_merges_to_zarr, combine_merge_results, find_collision_merges, find_directional_merges
from src.refined_features import compute_refined_features, save_refined_features_to_zarr
from src.visualizations import save_labeled_video, visualize_single_track
from src.tracking import run_tracking
from src.save_results import extract_and_save


if __name__ == "__main__":
    video_path = "/home/alex/Downloads/rotateAndHoldCCW_12112025_jelly1-12112025091630-0000.avi/rotateAndHoldCCW_12112025_jelly1-12112025091630-0000.avi"

    save_base_path = "./saved_videos"
    zarr_path = "tracking_results_jf.zarr"
    zarr_merged_path = "tracking_results_merged_jf.zarr"

    # Generate tracks - returns TrackingData object
    print("Generating tracks...")
    tracking_data, fps = run_tracking(video_path, max_frames=1000)
    
    # Extract features and save to Zarr
    
    extract_and_save(tracking_data, zarr_path, fps)
    refined = compute_refined_features(zarr_path, pause_speed_fraction=0.05, contact_distance_fraction=0.5)
    save_refined_features_to_zarr(zarr_path, refined)

    collision_merges = find_collision_merges(zarr_path, area_threshold=80)
    directional_merges = find_directional_merges(
        zarr_path,
        max_frame_gap=180,
        max_distance_px=40,
        distance_per_gap_px=6,
        direction_tolerance_deg=35,
    )
    merges = combine_merge_results([collision_merges, directional_merges])
    apply_merges_to_zarr(
        zarr_path,
        merges,
        output_path=zarr_merged_path,
        array_prefix="",  # merged arrays are written with prefix; originals preserved
        keep_original=False,
    )

    # Visualize a single track
    idx = 0
    visualize_single_track(
        video_path,
        zarr_path=zarr_merged_path,
        track_id=idx,
        output_path=f"{save_base_path}/track_{idx:02d}_visualization.mp4",
        padding=50,
        scale_factor=8,
        show_refined_motion=False,
        show_eccentricity=False,
        show_binary_mask=True,
        show_skeleton=False,
    )

    # Save labeled video with all tracks
    output_path = f"{save_base_path}/labeled_merged_video_jf.mp4"
    save_labeled_video(video_path, zarr_merged_path, output_path)
