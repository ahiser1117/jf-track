import os
from pathlib import Path
# Import the functions from the existing src package
from src.tracking import run_two_pass_tracking, merge_mouth_tracks
from src.direction_analysis import compute_direction_analysis
from src.save_results import save_two_pass_tracking_to_zarr
from src.visualizations import save_two_pass_labeled_video

def process_single_video(video_path, save_base_path):
    video_path = Path(video_path)
    video_stem = video_path.stem
    
    # Define paths specific to this video
    zarr_path = f"{video_stem}_tracking.zarr"
    frames_to_analyze = 5*60*20 
    
    print(f"\n>>> Starting: {video_stem}")

    # 1. Tracking
    mouth_raw, bulbs, fps, params = run_two_pass_tracking(
        str(video_path),
        max_frames=frames_to_analyze,
        background_buffer_size=25,
        mouth_min_area=35,
        mouth_max_area=160,
        mouth_max_disappeared=15,
        mouth_max_distance=50,
        mouth_search_radius=200,
        bulb_min_area=5,
        bulb_max_area=35,
        bulb_max_disappeared=10,
        bulb_max_distance=30,
        bulb_search_radius=150,
        adaptive_background=True,
        rotation_start_threshold_deg=0.02,
        rotation_stop_threshold_deg=0.005,
    )

    # 2. Processing
    mouth_tracking = merge_mouth_tracks(mouth_raw)
    direction_data = compute_direction_analysis(
        mouth_tracking, bulbs, mouth_track_idx=0, smooth_window=5
    )

    # 3. Saving Data
    save_two_pass_tracking_to_zarr(
        mouth_tracking, bulbs, direction_data, zarr_path, fps, parameters=params
    )

    # 4. Generating Visuals
    # Original Background
    out_orig = save_base_path / f"{video_stem}_labeled.mp4"
    save_two_pass_labeled_video(
        str(video_path), zarr_path, str(out_orig),
        max_frames=frames_to_analyze, show_direction_vector=True, show_bulb_com=True,
        background_mode="original"
    )

    # Diff Background
    out_diff = save_base_path / f"{video_stem}_diff.mp4"
    save_two_pass_labeled_video(
        str(video_path), zarr_path, str(out_diff),
        max_frames=frames_to_analyze, show_direction_vector=True, show_bulb_com=True,
        background_mode="diff"
    )
    
    print(f">>> Completed: {video_stem}")

if __name__ == "__main__":
    # Point this to your target directory
    input_folder = Path("/run/media/alex/AKUSB/weissbourd_lab/amitRotatationVideos/2026-01-20/")
    output_folder = Path("./batch_results")
    output_folder.mkdir(exist_ok=True)

    # Find all .avi files
    videos = list(input_folder.glob("*.avi"))

    for v in videos:
        try:
            process_single_video(v, output_folder)
        except Exception as e:
            print(f"Error processing {v.name}: {e}")