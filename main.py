from src.tracking import run_two_pass_tracking, merge_mouth_tracks
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
    # print("=== Running Two-Pass Tracking ===")
    # mouth_tracking_raw, bulb_tracking, fps = run_two_pass_tracking(
    #     video_path,
    #     max_frames=3000,
    #     # Background buffer: uses rolling average of last N frames
    #     background_buffer_size=10,
    #     # Mouth detection parameters
    #     mouth_min_area=35,
    #     mouth_max_area=160,
    #     mouth_max_disappeared=15,
    #     mouth_max_distance=50,
    #     # Bulb detection parameters (smaller objects, detect after mouth in second pass)
    #     bulb_min_area=5,
    #     bulb_max_area=35,
    #     bulb_max_disappeared=10,
    #     bulb_max_distance=30,
    #     # Adaptive background for rotating videos (set to True if background rotates)
    #     adaptive_background=True,
    #     rotation_start_threshold_deg=0.02,
    #     rotation_stop_threshold_deg=0.005,
    #     rotation_center=None,  # Auto-detect, or set to (cx, cy)
    # )

    # # Merge mouth track segments (mouth may be lost and reacquired due to occlusion)
    # print("\n=== Merging Mouth Track Segments ===")
    # mouth_tracking = merge_mouth_tracks(mouth_tracking_raw)

    # # Compute jellyfish direction from bulb center of mass to mouth (with temporal smoothing)
    # print("\n=== Computing Direction Analysis ===")
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

    # Save labeled visualization video
    print("\n=== Generating Label Visualization ===")
    output_video_path = f"{save_base_path}/labeled_video_clip.mp4"
    save_two_pass_labeled_video(
        video_path,
        zarr_path,
        output_video_path,
        max_frames=None,
        show_direction_vector=True,
        show_bulb_com=True,
        background_mode="diff",
        background_buffer_size=10,  # Match tracking buffer size
    )


    # Save adaptive background visualization (if used)
    # print("\n=== Generating Adaptive Background Visualization ===")
    # adaptive_bg_video_path = f"{save_base_path}/adaptive_bg_vis_clip.mp4"
    # visualize_adaptive_background(
    #     video_path,
    #     adaptive_bg_video_path,
    #     max_frames=None,
    #     buffer_size=10,  # Use rolling average of last N frames
    #     rotation_start_threshold_deg=0.02,
    #     rotation_stop_threshold_deg=0.005,
    #     rotation_center=None,  # Auto-detect, or set to (cx, cy)
    #     diff_threshold=10
    # )
