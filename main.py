from src.tracking import run_two_pass_tracking, select_best_mouth_track
from src.direction_analysis import compute_direction_analysis
from src.save_results import save_two_pass_tracking_to_zarr
from src.visualizations import save_two_pass_labeled_video
from src.adaptive_background import visualize_adaptive_background

if __name__ == "__main__":
    video_path = "/home/alex/Downloads/rotateAndHoldCCW_12112025_jelly1-12112025091630-0000.avi/rotateAndHoldCCW_12112025_jelly1-12112025091630-0000.avi"
    video_path = "/home/alex/Downloads/rotateAndHoldCCW_12112025_jelly1-12112025091630-0000.avi/clip.avi"

    save_base_path = "./saved_videos"
    zarr_path = "two_pass_tracking.zarr"

    # Run two-pass tracking: mouth (large) + bulbs (small)
    # print("Running two-pass tracking...")
    # mouth_tracking_raw, bulb_tracking, fps = run_two_pass_tracking(
    #     video_path,
    #     max_frames=None,
    #     # Mouth detection parameters
    #     mouth_min_area=35,
    #     mouth_max_area=160,
    #     mouth_max_disappeared=15,
    #     mouth_max_distance=50,
    #     # Bulb detection parameters (smaller objects)
    #     bulb_min_area=5,
    #     bulb_max_area=35,
    #     bulb_max_disappeared=10,
    #     bulb_max_distance=30,
    #     # Adaptive background for rotating videos (set to True if background rotates)
    #     adaptive_background=True,
    #     rotation_start_threshold_deg=0.5,
    #     rotation_stop_threshold_deg=0.1,
    #     rotation_center=None,  # Auto-detect, or set to (cx, cy)
    # )

    # # Select the single best mouth track (in case bulbs overlapped and created false positives)
    # print("\nSelecting best mouth track...")
    # mouth_tracking = select_best_mouth_track(mouth_tracking_raw)

    # # Compute direction from bulb center of mass to mouth (with temporal smoothing)
    # print("\nComputing direction analysis...")
    # direction_analysis = compute_direction_analysis(
    #     mouth_tracking,
    #     bulb_tracking,
    #     mouth_track_idx=0,  # Use first (and only) mouth track
    #     smooth_window=5,    # Smooth positions over 5 frames to reduce flickering
    # )

    # # Save all results to zarr
    # save_two_pass_tracking_to_zarr(
    #     mouth_tracking,
    #     bulb_tracking,
    #     direction_analysis,
    #     zarr_path,
    #     fps,
    # )

    # # Print summary
    # print("\n=== Tracking Summary ===")
    # print(f"Mouth tracks: {mouth_tracking.n_tracks}")
    # print(f"Bulb tracks: {bulb_tracking.n_tracks}")
    # print(f"Total frames: {direction_analysis.n_frames}")

    # # Direction statistics
    # import numpy as np
    # valid_directions = ~np.isnan(direction_analysis.direction_magnitude)
    # n_valid = np.sum(valid_directions)
    # print(f"Frames with valid direction: {n_valid} ({100*n_valid/direction_analysis.n_frames:.1f}%)")

    # if n_valid > 0:
    #     avg_magnitude = np.nanmean(direction_analysis.direction_magnitude)
    #     avg_bulb_count = np.mean(direction_analysis.bulb_count[valid_directions])
    #     print(f"Average direction magnitude: {avg_magnitude:.1f} px")
    #     print(f"Average bulb count per frame: {avg_bulb_count:.1f}")

    # # Save labeled visualization video
    # print("\n=== Generating Visualization ===")
    # output_video_path = f"{save_base_path}/two_pass_labeled_video.mp4"
    # save_two_pass_labeled_video(
    #     video_path,
    #     zarr_path,
    #     output_video_path,
    #     max_frames=None,
    #     show_direction_vector=True,
    #     show_bulb_com=True,
    #     background_mode="diff"
    # )

    # Save adaptive background visualization (if used)
    print("\nGenerating adaptive background visualization...")
    adaptive_bg_video_path = f"{save_base_path}/adaptive_background_video.mp4"
    visualize_adaptive_background(
        video_path,
        adaptive_bg_video_path,
        max_frames=None,
        rotation_start_threshold_deg=0.01,
        rotation_stop_threshold_deg=0.005,
        rotation_center=None,  # Auto-detect, or set to (cx, cy)
        diff_threshold=10
    )
