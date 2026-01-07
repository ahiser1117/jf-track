import numpy as np
import zarr
from dataclasses import dataclass


@dataclass
class RefinedFeatureData:
    """
    Stores refined behavior labels based on skeleton orientation.
    Shapes:
        front_x, front_y, tail_x, tail_y, projected_speed, is_forward,
        is_reverse, is_paused, motion_state: (n_tracks, n_frames)
        front_endpoint_index: (n_tracks,) with 0 or last point chosen as head
    """
    front_endpoint_index: np.ndarray
    front_x: np.ndarray
    front_y: np.ndarray
    tail_x: np.ndarray
    tail_y: np.ndarray
    projected_speed: np.ndarray
    is_forward: np.ndarray
    is_reverse: np.ndarray
    is_paused: np.ndarray
    motion_state: np.ndarray


def _choose_front_endpoint(skel_x: np.ndarray, skel_y: np.ndarray,
                           vel_dx: np.ndarray, vel_dy: np.ndarray) -> int:
    """
    Decide which skeleton endpoint is the head based on aggregate movement
    alignment across the track. Returns 0 or last index.
    """
    n_points = skel_x.shape[1]
    first = np.column_stack((skel_x[:, 0], skel_y[:, 0]))
    last = np.column_stack((skel_x[:, -1], skel_y[:, -1]))
    orient = first - last  # candidate where first is head

    vel = np.column_stack((vel_dx, vel_dy))
    vel_mag = np.linalg.norm(vel, axis=1)

    orient_norm = np.linalg.norm(orient, axis=1)
    valid = (~np.isnan(orient_norm)) & (orient_norm > 1e-6) & (vel_mag > 1e-6)

    if not np.any(valid):
        return 0  # Default to first endpoint

    orient_unit = np.zeros_like(orient)
    orient_unit[valid] = orient[valid] / orient_norm[valid, None]
    vel_unit = np.zeros_like(vel)
    vel_unit[valid] = vel[valid] / vel_mag[valid, None]

    score_first = np.nansum(np.einsum('ij,ij->i', vel_unit, orient_unit))
    score_last = -score_first  # Flipping endpoints flips sign

    return 0 if score_first >= score_last else n_points - 1


def _register_skeleton_sequence(skel_x: np.ndarray, skel_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce consistent endpoint ordering across frames by flipping sequences
    when endpoints appear swapped relative to the previous valid frame.
    """
    aligned_x = skel_x.copy()
    aligned_y = skel_y.copy()
    n_frames, n_points = aligned_x.shape

    prev_first = prev_last = None

    for frame_idx in range(n_frames):
        fx = aligned_x[frame_idx, 0]
        fy = aligned_y[frame_idx, 0]
        lx = aligned_x[frame_idx, n_points - 1]
        ly = aligned_y[frame_idx, n_points - 1]

        if np.isnan([fx, fy, lx, ly]).any():
            continue

        if prev_first is None:
            prev_first = (fx, fy)
            prev_last = (lx, ly)
            continue

        dist_same = np.hypot(fx - prev_first[0], fy - prev_first[1]) + np.hypot(lx - prev_last[0], ly - prev_last[1]) # type: ignore
        dist_swap = np.hypot(fx - prev_last[0], fy - prev_last[1]) + np.hypot(lx - prev_first[0], ly - prev_first[1]) # type: ignore

        if dist_swap < dist_same:
            aligned_x[frame_idx] = aligned_x[frame_idx][::-1]
            aligned_y[frame_idx] = aligned_y[frame_idx][::-1]
            fx, fy, lx, ly = aligned_x[frame_idx, 0], aligned_y[frame_idx, 0], aligned_x[frame_idx, n_points - 1], aligned_y[frame_idx, n_points - 1]

        prev_first = (fx, fy)
        prev_last = (lx, ly)

    return aligned_x, aligned_y


def _get_non_contact_episodes(contact_mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Given a boolean mask where True indicates endpoints are in contact,
    return list of (start, end) indices for contiguous non-contact segments.
    """
    non_contact = ~contact_mask
    episodes = []
    in_seg = False
    start = 0
    for idx, val in enumerate(non_contact):
        if val and not in_seg:
            start = idx
            in_seg = True
        elif not val and in_seg:
            episodes.append((start, idx - 1))
            in_seg = False
    if in_seg:
        episodes.append((start, len(contact_mask) - 1))
    return episodes


def compute_refined_features(
    zarr_path: str,
    pause_speed_fraction: float = 0.05,
    contact_distance_fraction: float = 0.25,
    contact_min_distance_px: float = 3.0,
) -> RefinedFeatureData:
    """
    Compute refined behavior labels using skeleton orientation to infer worm head.

    Args:
        zarr_path: Path to existing tracking/feature zarr store.
        pause_speed_fraction: Fraction of median speed used as pause threshold.
        contact_distance_fraction: Fraction of median endpoint distance used to mark head-tail contact.
        contact_min_distance_px: Absolute minimum distance (pixels) to mark contact.
    """
    root = zarr.open_group(zarr_path, mode='r')

    required = ('skeleton_x', 'skeleton_y')
    for key in required:
        if key not in root:
            raise ValueError(f"{key} not found in zarr. Regenerate tracking results with skeletons.")

    x = np.array(root['x'])
    y = np.array(root['y'])
    skeleton_x = np.array(root['skeleton_x'])
    skeleton_y = np.array(root['skeleton_y'])
    n_tracks, n_frames = x.shape

    smooth_x = np.array(root['SmoothX']) if 'SmoothX' in root else x
    smooth_y = np.array(root['SmoothY']) if 'SmoothY' in root else y
    speed_mm = np.array(root['Speed']) if 'Speed' in root else None

    all_speeds = speed_mm if speed_mm is not None else np.sqrt(
        np.square(np.diff(smooth_x, axis=1, prepend=np.nan)) +
        np.square(np.diff(smooth_y, axis=1, prepend=np.nan))
    )
    median_speed = np.nanmedian(all_speeds)
    pause_thresh = pause_speed_fraction * median_speed if not np.isnan(median_speed) else 0.01

    front_endpoint_index = np.zeros(n_tracks, dtype=np.int32)
    front_x = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
    front_y = np.full_like(front_x, np.nan)
    tail_x = np.full_like(front_x, np.nan)
    tail_y = np.full_like(front_x, np.nan)
    projected_speed = np.full_like(front_x, np.nan)
    is_forward = np.full((n_tracks, n_frames), False, dtype=bool)
    is_reverse = np.full_like(is_forward, False)
    is_paused = np.full_like(is_forward, False)
    motion_state = np.full((n_tracks, n_frames), 'unknown', dtype='U12')

    for track_idx in range(n_tracks):
        # Enforce consistent endpoint ordering across frames
        track_skel_x, track_skel_y = _register_skeleton_sequence(
            skeleton_x[track_idx], skeleton_y[track_idx]
        )

        # Endpoint distances to detect head-tail contact frames
        endpoint_dist = np.hypot(
            track_skel_x[:, 0] - track_skel_x[:, -1],
            track_skel_y[:, 0] - track_skel_y[:, -1],
        )
        median_dist = np.nanmedian(endpoint_dist)
        contact_thresh = max(contact_distance_fraction * median_dist, contact_min_distance_px) if not np.isnan(median_dist) else contact_min_distance_px
        contact_mask = (endpoint_dist < contact_thresh) & ~np.isnan(endpoint_dist)

        episodes = _get_non_contact_episodes(contact_mask)
        if not episodes:
            valid = np.where(~np.isnan(endpoint_dist))[0]
            if len(valid) > 0:
                episodes = [(valid.min(), valid.max())]
            else:
                continue

        sx = smooth_x[track_idx]
        sy = smooth_y[track_idx]
        dx = np.diff(sx, prepend=np.nan)
        dy = np.diff(sy, prepend=np.nan)

        speed_track = speed_mm[track_idx] if speed_mm is not None else np.sqrt(dx**2 + dy**2)
        first_head_idx = None

        for ep_start, ep_end in episodes:
            ep_skel_x = track_skel_x[ep_start:ep_end + 1]
            ep_skel_y = track_skel_y[ep_start:ep_end + 1]
            ep_dx = dx[ep_start:ep_end + 1]
            ep_dy = dy[ep_start:ep_end + 1]

            head_idx = _choose_front_endpoint(ep_skel_x, ep_skel_y, ep_dx, ep_dy)
            tail_idx = ep_skel_x.shape[1] - 1 - head_idx
            if first_head_idx is None:
                first_head_idx = head_idx

            for frame_idx in range(ep_start, ep_end + 1):
                fx = track_skel_x[frame_idx, head_idx]
                fy = track_skel_y[frame_idx, head_idx]
                tx = track_skel_x[frame_idx, tail_idx]
                ty = track_skel_y[frame_idx, tail_idx]

                if np.isnan([fx, fy, tx, ty]).any():
                    continue

                front_x[track_idx, frame_idx] = fx
                front_y[track_idx, frame_idx] = fy
                tail_x[track_idx, frame_idx] = tx
                tail_y[track_idx, frame_idx] = ty

                orient = np.array([fx - tx, fy - ty], dtype=np.float64)
                orient_norm = np.linalg.norm(orient)
                if orient_norm < 1e-6 or np.isnan(orient_norm):
                    continue
                orient_unit = orient / orient_norm

                vx = dx[frame_idx]
                vy = dy[frame_idx]
                if np.isnan(vx) or np.isnan(vy):
                    continue
                vel_mag_px = np.hypot(vx, vy)
                if vel_mag_px < 1e-8:
                    continue
                vel_unit = np.array([vx, vy]) / vel_mag_px

                speed_mag = speed_track[frame_idx]
                if np.isnan(speed_mag):
                    continue
                vel_vec_mm = vel_unit * speed_mag

                proj = float(np.dot(vel_vec_mm, orient_unit))
                projected_speed[track_idx, frame_idx] = proj

                if abs(proj) < pause_thresh:
                    is_paused[track_idx, frame_idx] = True
                    motion_state[track_idx, frame_idx] = 'paused'
                elif proj > 0:
                    is_forward[track_idx, frame_idx] = True
                    motion_state[track_idx, frame_idx] = 'forward'
                else:
                    is_reverse[track_idx, frame_idx] = True
                    motion_state[track_idx, frame_idx] = 'reverse'

        if first_head_idx is not None:
            front_endpoint_index[track_idx] = first_head_idx

    return RefinedFeatureData(
        front_endpoint_index=front_endpoint_index,
        front_x=front_x,
        front_y=front_y,
        tail_x=tail_x,
        tail_y=tail_y,
        projected_speed=projected_speed,
        is_forward=is_forward,
        is_reverse=is_reverse,
        is_paused=is_paused,
        motion_state=motion_state,
    )


def save_refined_features_to_zarr(zarr_path: str, refined: RefinedFeatureData):
    """
    Persist refined features into an existing zarr store (adds new arrays).
    Existing arrays with the same name are replaced.
    """
    root = zarr.open_group(zarr_path, mode='a')

    def write_array(name, data, dims):
        if name in root:
            del root[name]
        arr = root.create_array(name, data=data)
        arr.attrs['_ARRAY_DIMENSIONS'] = list(dims)

    write_array('refined_front_endpoint_index', refined.front_endpoint_index, dims=('track',))
    write_array('refined_front_x', refined.front_x, dims=('track', 'frame'))
    write_array('refined_front_y', refined.front_y, dims=('track', 'frame'))
    write_array('refined_tail_x', refined.tail_x, dims=('track', 'frame'))
    write_array('refined_tail_y', refined.tail_y, dims=('track', 'frame'))
    write_array('refined_projected_speed', refined.projected_speed, dims=('track', 'frame'))
    write_array('refined_is_forward', refined.is_forward, dims=('track', 'frame'))
    write_array('refined_is_reverse', refined.is_reverse, dims=('track', 'frame'))
    write_array('refined_is_paused', refined.is_paused, dims=('track', 'frame'))
    write_array('refined_motion_state', refined.motion_state, dims=('track', 'frame'))
