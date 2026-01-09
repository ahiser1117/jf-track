import cv2
import numpy as np
import zarr

from src.tracking import get_roi_params, create_circular_roi_mask, rotate_point
from src.adaptive_background import RollingBackgroundManager, AdaptiveBackgroundManager, RotationState


def save_two_pass_labeled_video(
    video_path: str,
    zarr_path: str,
    output_path: str,
    max_frames: int | None = None,
    show_direction_vector: bool = True,
    show_bulb_com: bool = True,
    background_mode: str = "original",
    diff_threshold: int = 10,
    background_buffer_size: int = 10,
    bulb_search_radius: int | None = None,
    mouth_search_radius: int | None = None,
    adaptive_background: bool = False,
    rotation_start_threshold_deg: float = 0.01,
    rotation_stop_threshold_deg: float = 0.005,
    rotation_center: tuple[float, float] | None = None,
):
    """
    Save a labeled video with two-pass tracking annotations (mouth + bulbs).

    Args:
        video_path: Path to source video
        zarr_path: Path to zarr store with two-pass tracking results
        output_path: Output video path
        max_frames: Maximum number of frames to process
        show_direction_vector: Draw direction vector from bulb CoM to mouth
        show_bulb_com: Show bulb center of mass marker
        background_mode: What to display as background:
            - "original": Original video frames (default)
            - "diff": Background subtraction difference image (uses rolling median)
            - "mask": Binary mask after thresholding (uses rolling median)
        diff_threshold: Threshold value for binary mask (only used when background_mode="mask")
        background_buffer_size: Number of recent frames for rolling background (default 10)
        bulb_search_radius: If set, draw circle showing bulb search area around mouth
        mouth_search_radius: If set, draw circle showing mouth search area when mouth is lost
        adaptive_background: Enable rotation-compensated search area visualization
        rotation_start_threshold_deg: Degrees/frame to trigger rotation detection
        rotation_stop_threshold_deg: Degrees/frame to consider rotation stopped
        rotation_center: Fixed rotation center (cx, cy), or None for auto-detection
    """
    root = zarr.open_group(zarr_path, mode='r')

    # Load mouth tracking data
    mouth_group = root['mouth']
    mouth_track_ids = np.array(mouth_group['track'])
    mouth_x = np.array(mouth_group['x'])
    mouth_y = np.array(mouth_group['y'])
    n_mouth_tracks = len(mouth_track_ids)

    # Load bulb tracking data
    bulb_group = root['bulb']
    bulb_track_ids = np.array(bulb_group['track'])
    bulb_x = np.array(bulb_group['x'])
    bulb_y = np.array(bulb_group['y'])
    n_bulb_tracks = len(bulb_track_ids)

    # Load direction analysis
    direction_group = root['direction']
    dir_mouth_x = np.array(direction_group['mouth_x'])
    dir_mouth_y = np.array(direction_group['mouth_y'])
    bulb_com_x = np.array(direction_group['bulb_com_x'])
    bulb_com_y = np.array(direction_group['bulb_com_y'])

    n_frames = len(dir_mouth_x)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize background manager
    bg_manager = None
    adaptive_bg_manager = None
    roi_mask = None

    if background_mode in ("diff", "mask"):
        print(f"Initializing rolling background (buffer_size={background_buffer_size})...")
        bg_manager = RollingBackgroundManager(buffer_size=background_buffer_size)
        roi_mask = create_circular_roi_mask(height, width)

    # Initialize adaptive background manager for rotation detection (used for search area rotation)
    if adaptive_background:
        print(f"Initializing adaptive background for rotation detection...")
        adaptive_bg_manager = AdaptiveBackgroundManager(
            video_path,
            buffer_size=background_buffer_size,
            rotation_start_threshold_deg=rotation_start_threshold_deg,
            rotation_stop_threshold_deg=rotation_stop_threshold_deg,
        )
        adaptive_bg_manager.initialize(
            max_frames=max_frames,
            initial_center_estimate=rotation_center,
        )

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Saving two-pass labeled video to {output_path}...")
    print(f"  Mouth tracks: {n_mouth_tracks}, Bulb tracks: {n_bulb_tracks}")
    print(f"  Background mode: {background_mode}")

    # Colors
    MOUTH_COLOR = (0, 165, 255)  # Orange (BGR)
    BULB_COLOR = (255, 200, 0)   # Cyan (BGR)
    COM_COLOR = (0, 255, 255)    # Yellow (BGR)
    DIRECTION_COLOR = (0, 255, 0)  # Green (BGR)
    ROI_COLOR = (128, 128, 128)  # Gray (BGR)
    BULB_SEARCH_COLOR = (255, 200, 0)    # Cyan (BGR) - matches bulb color
    MOUTH_SEARCH_COLOR = (0, 100, 255)   # Dark orange (BGR) - indicates searching

    # Get ROI circle parameters
    roi_center, roi_radius = get_roi_params(height, width)

    if max_frames is None:
        max_frames = n_frames

    # Track last known mouth position for search radius visualization
    last_mouth_position = None

    for frame_idx in range(min(n_frames, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply background mode transformation using rolling background
        if background_mode in ("diff", "mask") and bg_manager is not None:
            bg_manager.update(gray)

            if bg_manager.is_ready():
                gray_bg = bg_manager.get_background()
                diff = cv2.absdiff(gray, gray_bg)

                if background_mode == "diff":
                    # Scale diff to use full dynamic range for better visibility
                    diff_scaled = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
                    frame = cv2.cvtColor(diff_scaled, cv2.COLOR_GRAY2BGR)
                else:  # mask mode
                    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
                    # Apply ROI mask to match what tracking sees
                    mask = cv2.bitwise_and(mask, roi_mask)
                    frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Update adaptive background manager and rotate search position if rotation detected
        if adaptive_bg_manager is not None:
            adaptive_bg_manager.get_background(frame_idx, gray)  # Updates rotation state
            if last_mouth_position is not None:
                if adaptive_bg_manager.get_state() in (RotationState.ROTATING, RotationState.TRANSITION):
                    rotation_angle = adaptive_bg_manager.get_last_smoothed_angle()
                    rotation_center_pt = adaptive_bg_manager.get_rotation_center()
                    if rotation_center_pt is not None and abs(rotation_angle) > 0.001:
                        last_mouth_position = rotate_point(last_mouth_position, rotation_angle, rotation_center_pt)
                        last_mouth_position = (int(last_mouth_position[0]), int(last_mouth_position[1]))

        # Draw bulb tracks (smaller circles, cyan)
        for track_idx in range(n_bulb_tracks):
            if frame_idx < bulb_x.shape[1]:
                x = bulb_x[track_idx, frame_idx]
                y = bulb_y[track_idx, frame_idx]

                if not np.isnan(x):
                    track_id = bulb_track_ids[track_idx]
                    # Small filled circle for bulbs
                    cv2.circle(frame, (int(x), int(y)), 3, BULB_COLOR, -1)
                    cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), 1)
                    # Label
                    cv2.putText(frame, f"B{track_id}", (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, BULB_COLOR, 1)

        # Draw bulb center of mass
        if show_bulb_com and not np.isnan(bulb_com_x[frame_idx]):
            com_x = int(bulb_com_x[frame_idx])
            com_y = int(bulb_com_y[frame_idx])
            # Draw crosshair for CoM
            cv2.drawMarker(frame, (com_x, com_y), COM_COLOR,
                          cv2.MARKER_CROSS, 12, 2)
            cv2.putText(frame, "CoM", (com_x + 8, com_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COM_COLOR, 1)

        # Draw mouth tracks (larger circles, orange)
        current_mouth_position = None
        for track_idx in range(n_mouth_tracks):
            if frame_idx < mouth_x.shape[1]:
                x = mouth_x[track_idx, frame_idx]
                y = mouth_y[track_idx, frame_idx]

                if not np.isnan(x):
                    current_mouth_position = (int(x), int(y))
                    track_id = mouth_track_ids[track_idx]
                    # Larger filled circle for mouth
                    cv2.circle(frame, (int(x), int(y)), 6, MOUTH_COLOR, -1)
                    cv2.circle(frame, (int(x), int(y)), 6, (255, 255, 255), 2)
                    # Label
                    cv2.putText(frame, f"Mouth", (int(x) - 20, int(y) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, MOUTH_COLOR, 2)

        # Update last known mouth position and draw search radius circles
        if current_mouth_position is not None:
            last_mouth_position = current_mouth_position
            # Draw bulb search radius around current mouth position
            if bulb_search_radius is not None:
                cv2.circle(frame, current_mouth_position, bulb_search_radius, BULB_SEARCH_COLOR, 1)
        elif last_mouth_position is not None and mouth_search_radius is not None:
            # Mouth is lost - draw search radius around last known position
            # This area is used for both mouth and bulb search
            cv2.circle(frame, last_mouth_position, mouth_search_radius, MOUTH_SEARCH_COLOR, 2)
            cv2.putText(frame, "Search area", (last_mouth_position[0] - 35, last_mouth_position[1] - mouth_search_radius - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, MOUTH_SEARCH_COLOR, 1)

        # Draw direction vector from bulb CoM to mouth
        if show_direction_vector:
            if not np.isnan(bulb_com_x[frame_idx]) and not np.isnan(dir_mouth_x[frame_idx]):
                com_x = int(bulb_com_x[frame_idx])
                com_y = int(bulb_com_y[frame_idx])
                mouth_px = int(dir_mouth_x[frame_idx])
                mouth_py = int(dir_mouth_y[frame_idx])
                # Draw arrow from CoM to mouth
                cv2.arrowedLine(frame, (com_x, com_y), (mouth_px, mouth_py),
                               DIRECTION_COLOR, 2, tipLength=0.15)

        # Draw ROI circle boundary
        cv2.circle(frame, roi_center, roi_radius, ROI_COLOR, 1)

        out.write(frame)

        if (frame_idx + 1) % 100 == 0:
            print(f"Writing frame {frame_idx + 1}")

    cap.release()
    out.release()
    print("Done.")


def save_labeled_video(video_path: str, zarr_path: str, output_path: str,
                       max_frames: int = 500):
    """
    Save a labeled video with tracking annotations.

    Args:
        video_path: Path to source video
        zarr_path: Path to zarr store with tracking results
        output_path: Output video path
        max_frames: Maximum number of frames to process
    """
    # Load tracking data from zarr
    root = zarr.open_group(zarr_path, mode='r')

    track_ids = np.array(root['track'])
    n_tracks = len(track_ids)
    x_vals = np.array(root['x'])
    y_vals = np.array(root['y'])
    n_frames = x_vals.shape[1]

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Saving labeled video to {output_path}...")

    # Generate distinct colors for each track
    colors = []
    for i in range(n_tracks):
        hue = int(180 * i / max(n_tracks, 1))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(int(c) for c in color_bgr))

    for frame_idx in range(min(n_frames, max_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw all tracks that have data at this frame
        for track_idx in range(n_tracks):
            x = x_vals[track_idx, frame_idx]
            y = y_vals[track_idx, frame_idx]

            if not np.isnan(x):
                track_id = track_ids[track_idx]
                color = colors[track_idx]

                text = f"ID {track_id}"
                cv2.putText(frame, text, (int(x) - 10, int(y) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)

        out.write(frame)

        if (frame_idx + 1) % 100 == 0:
            print(f"Writing frame {frame_idx + 1}")

    cap.release()
    out.release()
    print("Done.")


def visualize_single_track(video_path: str, zarr_path: str, track_id: int, output_path: str,
                           padding: int = 50, scale_factor: int = 4,
                           show_binary_mask: bool = False):
    """
    Visualize a single track with position and area info.

    Args:
        video_path: Path to source video
        zarr_path: Path to zarr store with tracking results
        track_id: The track ID to visualize
        output_path: Output video path
        padding: Pixels to pad around the object bounding box (in original resolution)
        scale_factor: Factor to upscale the video for higher quality output
        show_binary_mask: Whether to display the binary mask instead of the original video
    """
    # Load tracking data directly from zarr
    root = zarr.open_group(zarr_path, mode='r')

    # Get track index
    track_ids = np.array(root['track'])
    if track_id not in track_ids:
        raise ValueError(f"Track {track_id} not found. Available: {track_ids}")

    track_idx = int(np.where(track_ids == track_id)[0][0])

    # Extract data for this track (all arrays are (n_tracks, n_frames))
    x_vals = np.array(root['x'])[track_idx, :]
    y_vals = np.array(root['y'])[track_idx, :]
    area = np.array(root['area'])[track_idx, :]

    # Find valid (non-NaN) frame indices - these are 0-indexed
    valid_mask = ~np.isnan(x_vals)
    valid_frame_indices = np.where(valid_mask)[0]

    if len(valid_frame_indices) == 0:
        print(f"Track {track_id} has no valid data")
        return

    cap = cv2.VideoCapture(video_path)
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate background if showing binary mask
    gray_bg = None
    if show_binary_mask:
        print("Calculating background for binary mask...")
        frame_indices_bg = np.linspace(0, total_frames - 1, 50).astype(int)
        bg_frames = []
        for idx in frame_indices_bg:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                bg_frames.append(frame)
        background = np.median(bg_frames, axis=0).astype(np.uint8)
        gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

    # Original crop size (before upscaling)
    crop_size_orig = padding * 2 + 100

    # Upscaled sizes for high quality output
    crop_size_scaled = crop_size_orig * scale_factor
    info_panel_width = 300
    out_width = crop_size_scaled + info_panel_width
    out_height = crop_size_scaled

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    # Frame range (0-indexed)
    start_frame = int(valid_frame_indices.min())
    end_frame = int(valid_frame_indices.max())

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Visualizing track {track_id} from frame {start_frame + 1} to {end_frame + 1}...")
    print(f"Output resolution: {out_width}x{out_height} (scale factor: {scale_factor}x)")

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames where this track has no data
        if not valid_mask[frame_idx]:
            continue

        # Convert to binary mask if requested
        if show_binary_mask and gray_bg is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, gray_bg)
            _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        cx, cy = int(x_vals[frame_idx]), int(y_vals[frame_idx])

        # Crop around object (original resolution)
        x1 = max(0, cx - crop_size_orig // 2)
        y1 = max(0, cy - crop_size_orig // 2)
        x2 = min(orig_width, x1 + crop_size_orig)
        y2 = min(orig_height, y1 + crop_size_orig)

        cropped = frame[y1:y2, x1:x2].copy()

        # Calculate local centroid position before resizing
        local_cx_orig = cx - x1
        local_cy_orig = cy - y1

        # Upscale the cropped region
        cropped = cv2.resize(cropped, (crop_size_scaled, crop_size_scaled),
                            interpolation=cv2.INTER_NEAREST if show_binary_mask else cv2.INTER_CUBIC)

        # Scale local centroid to match upscaled image
        scale_x = crop_size_scaled / (x2 - x1) if (x2 - x1) > 0 else scale_factor
        scale_y = crop_size_scaled / (y2 - y1) if (y2 - y1) > 0 else scale_factor
        local_cx = int(local_cx_orig * scale_x)
        local_cy = int(local_cy_orig * scale_y)

        # Draw annotations on upscaled image
        if 0 <= local_cx < crop_size_scaled and 0 <= local_cy < crop_size_scaled:
            color = (0, 255, 0)  # Green

            # Scaled annotation sizes
            centroid_radius = 3 * scale_factor // 2

            cv2.circle(cropped, (local_cx, local_cy), centroid_radius, color, -1)
            cv2.circle(cropped, (local_cx, local_cy), centroid_radius + 2, (255, 255, 255), 2)

        # Create info panel
        info_panel = np.zeros((crop_size_scaled, info_panel_width, 3), dtype=np.uint8)
        info_panel[:] = (30, 30, 30)

        cv2.line(info_panel, (0, 0), (0, crop_size_scaled), (80, 80, 80), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        title_font_scale = 1.2
        label_font_scale = 0.9
        value_font_scale = 0.75
        line_height = 50
        section_gap = 20

        y_text = 50
        x_label = 20

        # Title
        cv2.putText(info_panel, f"Track {track_id}", (x_label, y_text),
                   font, title_font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        y_text += line_height + section_gap

        cv2.line(info_panel, (x_label, y_text - 15), (info_panel_width - 20, y_text - 15), (80, 80, 80), 1)

        # Frame info (1-indexed for display)
        cv2.putText(info_panel, "Frame", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        y_text += int(line_height * 0.6)
        cv2.putText(info_panel, f"{frame_idx + 1}", (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += line_height

        # Position
        cv2.putText(info_panel, "Position", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        y_text += int(line_height * 0.6)
        cv2.putText(info_panel, f"({cx}, {cy})", (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += line_height

        # Area
        cv2.putText(info_panel, "Area", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        y_text += int(line_height * 0.6)
        area_val = area[frame_idx]
        cv2.putText(info_panel, f"{area_val:.1f} px", (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Combine
        combined = np.hstack([cropped, info_panel])
        out.write(combined)

        if (frame_idx + 1) % 100 == 0:
            print(f"Processing frame {frame_idx + 1}")

    cap.release()
    out.release()
    print(f"Done. Saved to {output_path}")
