import cv2
import numpy as np
import zarr


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
    
    # Load behavior data for coloring
    is_reversal = np.array(root['is_Reversal'])
    is_omega = np.array(root['is_Omega'])
    is_upsilon = np.array(root['is_Upsilon'])
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Saving labeled video to {output_path}...")
    
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
                
                # Color based on behavior state
                if is_omega[track_idx, frame_idx]:
                    color = (0, 0, 255)  # Red for omega
                elif is_upsilon[track_idx, frame_idx]:
                    color = (0, 165, 255)  # Orange for upsilon
                elif is_reversal[track_idx, frame_idx]:
                    color = (0, 255, 255)  # Yellow for reversal
                else:
                    color = (0, 255, 0)  # Green for normal
                
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
                           show_eccentricity: bool = False, show_binary_mask: bool = False,
                           show_skeleton: bool = False,
                           show_refined_motion: bool = False):
    """
    Visualize a single track with feature overlays.
    
    Args:
        video_path: Path to source video
        zarr_path: Path to zarr store with tracking results
        track_id: The track ID to visualize
        output_path: Output video path
        padding: Pixels to pad around the worm bounding box (in original resolution)
        scale_factor: Factor to upscale the video for higher quality output
        show_eccentricity: Whether to visualize eccentricity as an ellipse overlay on the worm
        show_binary_mask: Whether to display the binary mask used for tracking instead of the original video
        show_skeleton: Whether to overlay sampled skeleton points (if available in the zarr file)
        show_refined_motion: Whether to overlay refined front/back markers and motion labels (if present)
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
    speed = np.array(root['Speed'])[track_idx, :]
    direction = np.array(root['Direction'])[track_idx, :]
    ang_speed = np.array(root['AngSpeed'])[track_idx, :]
    is_reversal = np.array(root['is_Reversal'])[track_idx, :]
    is_omega = np.array(root['is_Omega'])[track_idx, :]
    is_upsilon = np.array(root['is_Upsilon'])[track_idx, :]
    eccentricity = np.array(root['eccentricity'])[track_idx, :]
    reversal_category = np.array(root['Reversal_Category'])[track_idx, :]
    
    skeleton_x = skeleton_y = None
    if show_skeleton:
        if 'skeleton_x' in root and 'skeleton_y' in root:
            skeleton_x = np.array(root['skeleton_x'])[track_idx, :, :]
            skeleton_y = np.array(root['skeleton_y'])[track_idx, :, :]
        else:
            print("Skeleton data not found in zarr; disabling skeleton overlay.")
            show_skeleton = False

    refined_front_x = refined_front_y = refined_tail_x = refined_tail_y = None
    refined_motion_state = refined_projected_speed = None
    if show_refined_motion:
        keys_needed = ['refined_front_x', 'refined_front_y', 'refined_tail_x', 
                       'refined_tail_y', 'refined_motion_state', 'refined_projected_speed']
        if all(k in root for k in keys_needed):
            refined_front_x = np.array(root['refined_front_x'])[track_idx, :]
            refined_front_y = np.array(root['refined_front_y'])[track_idx, :]
            refined_tail_x = np.array(root['refined_tail_x'])[track_idx, :]
            refined_tail_y = np.array(root['refined_tail_y'])[track_idx, :]
            refined_motion_state = np.array(root['refined_motion_state'])[track_idx, :]
            refined_projected_speed = np.array(root['refined_projected_speed'])[track_idx, :]
        else:
            print("Refined motion data not found in zarr; disabling refined overlay.")
            show_refined_motion = False
    
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
    info_panel_width = 500
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
        
        # Crop around worm (original resolution)
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
            # Color based on behavior state
            if is_omega[frame_idx]:
                color = (0, 0, 255)  # Red for omega
            elif is_upsilon[frame_idx]:
                color = (0, 165, 255)  # Orange for upsilon
            elif is_reversal[frame_idx]:
                color = (0, 255, 255)  # Yellow for reversal
            else:
                color = (0, 255, 0)  # Green for normal
            
            # Scaled annotation sizes
            centroid_radius = 3 * scale_factor // 2
            arrow_len = 10 * scale_factor // 2
            arrow_thickness = max(2, scale_factor)
            
            # Draw eccentricity ellipse if enabled
            if show_eccentricity:
                ecc_val = eccentricity[frame_idx]
                major_axis = 25 * scale_factor // 2
                minor_axis = int(major_axis * np.sqrt(1 - ecc_val**2))
                ellipse_angle = direction[frame_idx] - 90
                cv2.ellipse(cropped, (local_cx, local_cy), (major_axis, minor_axis),
                           ellipse_angle, 0, 360, (255, 255, 0), 2)
            
            cv2.circle(cropped, (local_cx, local_cy), centroid_radius, color, -1)
            cv2.circle(cropped, (local_cx, local_cy), centroid_radius + 2, (255, 255, 255), 2)
            
            # Draw direction arrow
            dir_rad = np.radians(direction[frame_idx])
            if dir_rad is None or np.isnan(dir_rad):
                dir_rad = 0.0
            end_x = int(local_cx + arrow_len * np.sin(dir_rad))
            end_y = int(local_cy - arrow_len * np.cos(dir_rad))
            cv2.arrowedLine(cropped, (local_cx, local_cy), (end_x, end_y), 
                           (255, 255, 0), arrow_thickness, tipLength=0.3)
        
        if show_skeleton and skeleton_x is not None and skeleton_y is not None:
            skel_points = []
            for px, py in zip(skeleton_x[frame_idx], skeleton_y[frame_idx]):
                if np.isnan(px) or np.isnan(py):
                    continue
                local_px = int((px - x1) * scale_x)
                local_py = int((py - y1) * scale_y)
                if 0 <= local_px < crop_size_scaled and 0 <= local_py < crop_size_scaled:
                    skel_points.append((local_px, local_py))
            if len(skel_points) >= 2:
                cv2.polylines(cropped, [np.array(skel_points, dtype=np.int32)], False, (255, 0, 255), max(2, scale_factor - 1))
            for pt in skel_points:
                cv2.circle(cropped, pt, max(2, scale_factor - 1), (255, 0, 128), -1)

        if show_refined_motion and refined_front_x is not None and refined_front_y is not None:
            fx = refined_front_x[frame_idx]
            fy = refined_front_y[frame_idx]
            tx = refined_tail_x[frame_idx] # type: ignore
            ty = refined_tail_y[frame_idx] # type: ignore
            if not np.isnan([fx, fy, tx, ty]).any():
                local_fx = int((fx - x1) * scale_x)
                local_fy = int((fy - y1) * scale_y)
                local_tx = int((tx - x1) * scale_x)
                local_ty = int((ty - y1) * scale_y)
                cv2.circle(cropped, (local_tx, local_ty), max(3, scale_factor), (0, 200, 255), -1)
                cv2.circle(cropped, (local_fx, local_fy), max(3, scale_factor), (255, 0, 255), -1)
                cv2.line(cropped, (local_tx, local_ty), (local_fx, local_fy), (128, 128, 255), max(2, scale_factor - 1))
        
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
        
        # Speed
        cv2.putText(info_panel, "Speed", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        y_text += int(line_height * 0.6)
        cv2.putText(info_panel, f"{speed[frame_idx]:.4f} mm/s", (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += line_height
        
        # Direction
        cv2.putText(info_panel, "Direction", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        y_text += int(line_height * 0.6)
        cv2.putText(info_panel, f"{direction[frame_idx]:.1f} deg", (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += line_height
        
        # Angular Speed
        cv2.putText(info_panel, "Angular Speed", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        y_text += int(line_height * 0.6)
        cv2.putText(info_panel, f"{ang_speed[frame_idx]:.1f} deg/s", (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += line_height
        
        # Eccentricity
        cv2.putText(info_panel, "Eccentricity", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
        y_text += int(line_height * 0.6)
        cv2.putText(info_panel, f"{eccentricity[frame_idx]:.4f}", (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        y_text += line_height + section_gap
        
        cv2.line(info_panel, (x_label, y_text - 15), (info_panel_width - 20, y_text - 15), (80, 80, 80), 1)
        
        # Behavior state
        state_text = "Forward"
        state_color = (0, 255, 0)
        if is_omega[frame_idx]:
            state_text = "OMEGA TURN"
            state_color = (0, 0, 255)
        elif is_upsilon[frame_idx]:
            state_text = "UPSILON TURN"
            state_color = (0, 165, 255)
        elif is_reversal[frame_idx]:
            state_text = "REVERSAL"
            state_color = (0, 255, 255)

        if show_refined_motion and refined_motion_state is not None:
            r_state = str(refined_motion_state[frame_idx])
            if r_state:
                state_text = f"REFINED: {r_state.upper()}"
                if r_state == 'forward':
                    state_color = (0, 200, 0)
                elif r_state == 'reverse':
                    state_color = (255, 200, 0)
                elif r_state == 'paused':
                    state_color = (180, 180, 180)
                else:
                    state_color = (100, 149, 237)
        
        box_y1 = y_text
        box_y2 = y_text + 50
        cv2.rectangle(info_panel, (x_label, box_y1), (info_panel_width - 20, box_y2), state_color, -1)
        cv2.rectangle(info_panel, (x_label, box_y1), (info_panel_width - 20, box_y2), (255, 255, 255), 2)
        
        text_size = cv2.getTextSize(state_text, font, 0.8, 2)[0]
        text_x = x_label + (info_panel_width - 40 - text_size[0]) // 2
        text_y = box_y1 + (50 + text_size[1]) // 2
        cv2.putText(info_panel, state_text, (text_x, text_y), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        y_text = box_y2 + section_gap + 10

        if show_refined_motion and refined_projected_speed is not None:
            cv2.putText(info_panel, "Proj Speed", (x_label, y_text), font, label_font_scale, (180, 180, 180), 1, cv2.LINE_AA)
            y_text += int(line_height * 0.6)
            val = refined_projected_speed[frame_idx]
            if np.isnan(val):
                val_str = "nan"
            else:
                val_str = f"{val:.4f} (signed)"
            cv2.putText(info_panel, val_str, (x_label, y_text), font, value_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            y_text += line_height + section_gap
        
        # Reversal Category
        rev_cat = str(reversal_category[frame_idx]) if reversal_category[frame_idx] else 'None'
        
        rev_color = (100, 100, 100)
        if rev_cat != 'None' and rev_cat != '':
            if 'Omega' in rev_cat:
                rev_color = (128, 0, 128)
            elif 'Upsilon' in rev_cat:
                rev_color = (255, 0, 128)
            elif 'pure' in rev_cat:
                rev_color = (255, 255, 0)
            else:
                rev_color = (0, 200, 200)
        
        box_y1 = y_text
        box_y2 = y_text + 50
        cv2.rectangle(info_panel, (x_label, box_y1), (info_panel_width - 20, box_y2), rev_color, -1)
        cv2.rectangle(info_panel, (x_label, box_y1), (info_panel_width - 20, box_y2), (255, 255, 255), 2)
        
        rev_display = rev_cat.upper() if rev_cat != 'None' and rev_cat != '' else 'NO REVERSAL'
        text_size = cv2.getTextSize(rev_display, font, 0.7, 2)[0]
        text_x = x_label + (info_panel_width - 40 - text_size[0]) // 2
        text_y = box_y1 + (50 + text_size[1]) // 2
        cv2.putText(info_panel, rev_display, (text_x, text_y), font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Combine
        combined = np.hstack([cropped, info_panel])
        out.write(combined)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"Processing frame {frame_idx + 1}")
    
    cap.release()
    out.release()
    print(f"Done. Saved to {output_path}")
