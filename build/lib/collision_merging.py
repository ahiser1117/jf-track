import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import zarr


@dataclass
class TrackStats:
    track_id: int
    track_idx: int
    start_frame: int
    end_frame: int
    start_pos: Tuple[float, float]
    end_pos: Tuple[float, float]
    start_dir: float
    end_dir: float


@dataclass
class MergeCandidate:
    parent_track_id: int
    child_track_id: int
    collision_frame: int | None
    gap_frames: int
    distance_px: float
    direction_diff_deg: float
    score: float
    merge_type: str = "collision"


@dataclass
class CollisionMergeResult:
    merges: List[MergeCandidate]
    params: Dict[str, float | int]


def _first_valid_index(arr: np.ndarray) -> int | None:
    idx = np.where(~np.isnan(arr))[0]
    return int(idx[0]) if idx.size else None


def _last_valid_index(arr: np.ndarray) -> int | None:
    idx = np.where(~np.isnan(arr))[0]
    return int(idx[-1]) if idx.size else None


def _angle_diff_deg(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return math.inf
    diff = (a - b + 180.0) % 360.0 - 180.0
    return abs(diff)


def _circular_mean_deg(angles: np.ndarray) -> float:
    valid = angles[~np.isnan(angles)]
    if valid.size == 0:
        return math.nan
    radians = np.deg2rad(valid)
    mean_angle = math.atan2(np.nanmean(np.sin(radians)), np.nanmean(np.cos(radians)))
    return float(np.degrees(mean_angle) % 360.0)


def _heading_from_points(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """
    Compute heading (deg) from p1->p2 using the same convention as featureExtraction
    (x forward, y inverted to keep "up" as positive).
    """
    dx = p2[0] - p1[0]
    dy = -(p2[1] - p1[1])
    return float(np.degrees(np.arctan2(dx, dy)) % 360.0)


def _nan_mask(vals: np.ndarray) -> np.ndarray:
    """Return mask of valid entries, handling non-float dtypes gracefully."""
    if np.issubdtype(vals.dtype, np.floating):
        return ~np.isnan(vals)
    return np.ones_like(vals, dtype=bool)


def _combine_track_frames(data: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Combine multiple track rows (frame-major) by taking the first valid value
    at each frame, preserving later tracks when earlier ones are NaN/empty.
    """
    sample = data[0]
    if data.dtype == bool:
        combined = np.zeros_like(sample, dtype=bool)
        for idx in indices:
            combined |= data[idx]
        return combined
    if data.dtype.kind in ("U", "S", "O"):
        combined = np.full_like(sample, "", dtype=data.dtype)
        for idx in indices:
            vals = data[idx]
            write_mask = (combined == "") & (vals != "")
            combined[write_mask] = vals[write_mask]
        return combined

    combined = np.full_like(sample, np.nan, dtype=np.float64)
    for idx in indices:
        vals = data[idx]
        mask = _nan_mask(vals)
        write_mask = mask & np.isnan(combined)
        combined[write_mask] = vals[write_mask]
    return combined


def _combine_track_frames_3d(data: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Combine multi-point track data (e.g., skeleton) frame-wise using the same
    first-valid rule as `_combine_track_frames`.
    """
    sample = data[0]
    if data.dtype == bool:
        combined = np.zeros_like(sample, dtype=bool)
        for idx in indices:
            combined |= data[idx]
        return combined
    if data.dtype.kind in ("U", "S", "O"):
        combined = np.full_like(sample, "", dtype=data.dtype)
        for idx in indices:
            vals = data[idx]
            write_mask = (combined == "") & (vals != "")
            combined[write_mask] = vals[write_mask]
        return combined

    combined = np.full_like(sample, np.nan, dtype=np.float64)
    for idx in indices:
        vals = data[idx]
        mask = _nan_mask(vals)
        write_mask = mask & np.isnan(combined)
        combined[write_mask] = vals[write_mask]
    return combined


def _edge_direction_summary(
    direction_row: np.ndarray,
    x_row: np.ndarray,
    y_row: np.ndarray,
    frame_idx: int,
    window: int = 5,
    forward: bool = False,
) -> float:
    """
    Aggregate direction near a track endpoint. Uses circular mean of the
    requested window; if unavailable, falls back to heading between the first
    and last valid coordinates inside the same window.
    """
    n = direction_row.shape[0]
    if forward:
        start = frame_idx
        end = min(n, frame_idx + window)
    else:
        start = max(0, frame_idx - window + 1)
        end = frame_idx + 1

    dir_mean = _circular_mean_deg(direction_row[start:end])
    if not math.isnan(dir_mean):
        return dir_mean

    x_seg = x_row[start:end]
    y_seg = y_row[start:end]
    valid = np.where(~np.isnan(x_seg) & ~np.isnan(y_seg))[0]
    if valid.size >= 2:
        p1 = (float(x_seg[valid[0]]), float(y_seg[valid[0]]))
        p2 = (float(x_seg[valid[-1]]), float(y_seg[valid[-1]]))
        return _heading_from_points(p1, p2)

    return math.nan


def _collect_track_stats(
    x: np.ndarray,
    y: np.ndarray,
    direction: np.ndarray,
    track_ids: np.ndarray,
) -> List[TrackStats]:
    stats: List[TrackStats] = []
    for idx, track_id in enumerate(track_ids):
        start = _first_valid_index(x[idx])
        end = _last_valid_index(x[idx])
        if start is None or end is None:
            continue

        start_pos = (float(x[idx, start]), float(y[idx, start]))
        end_pos = (float(x[idx, end]), float(y[idx, end]))
        start_dir = float(direction[idx, start]) if not np.isnan(direction[idx, start]) else math.nan
        end_dir = float(direction[idx, end]) if not np.isnan(direction[idx, end]) else math.nan

        stats.append(
            TrackStats(
                track_id=int(track_id),
                track_idx=int(idx),
                start_frame=int(start),
                end_frame=int(end),
                start_pos=start_pos,
                end_pos=end_pos,
                start_dir=start_dir,
                end_dir=end_dir,
            )
        )
    return stats


def _collision_frames_from_area(area: np.ndarray, area_threshold: float) -> np.ndarray:
    """
    Identify frames where any track shows an area spike (collision indicator).
    Returns sorted unique frame indices (0-indexed).
    """
    mask = np.any(area > area_threshold, axis=0)
    return np.where(mask)[0].astype(int)


def _nearest_collision_frame(
    collision_frames: np.ndarray,
    parent_end: int,
    child_start: int,
    padding: int,
) -> int | None:
    """
    Find a collision frame close to both parent end and child start.
    """
    if collision_frames.size == 0:
        return None
    lo = parent_end - padding
    hi = child_start + padding
    window = collision_frames[(collision_frames >= lo) & (collision_frames <= hi)]
    if window.size == 0:
        return None
    # Prefer collisions between parent end and child start, else nearest within padding
    mid_window = window[(window >= parent_end) & (window <= child_start)]
    if mid_window.size:
        return int(mid_window[np.argmin(np.abs(mid_window - parent_end))])
    return int(window[np.argmin(np.minimum(np.abs(window - parent_end), np.abs(window - child_start)))])


def find_collision_merges(
    zarr_path: str,
    area_threshold: float = 80.0,
    max_frame_gap: int = 90,
    max_distance_px: float = 50.0,
    direction_tolerance_deg: float = 90.0,
    collision_padding: int = 8,
) -> CollisionMergeResult:
    """
    Detect likely child->parent track merges caused by collision-induced track loss.

    A conserved track shows the area spike (> area_threshold). The parent track
    is the one that disappears at the collision, and the child track is the new
    ID that emerges after separation. A valid merge must satisfy:
        - A collision frame (any track area spike) sits between the parent end
          and child start, or within `collision_padding` frames of them.
        - Child appears within max_frame_gap frames after the parent ended.
        - The gap between the parent's last point and the child's first point
          is within max_distance_px.
        - Motion direction from parent->child roughly matches the parent's end
          direction (and/or the child's start direction) within the tolerance.
    """
    root = zarr.open_group(zarr_path, mode="r")

    track_ids = np.array(root["track"]).astype(int)
    x = np.array(root["x"])
    y = np.array(root["y"])
    area = np.array(root["area"])

    # Compute direction from position
    dx = np.diff(x, axis=1, prepend=np.nan)
    dy = -np.diff(y, axis=1, prepend=np.nan)
    direction = (np.degrees(np.arctan2(dx, dy)) % 360.0).astype(float)

    stats = _collect_track_stats(x, y, direction, track_ids)
    stats_by_id = {s.track_id: s for s in stats}
    collision_frames = _collision_frames_from_area(area, area_threshold)

    merges: List[MergeCandidate] = []

    for child in stats:
        # Only consider tracks that appear mid-video
        if child.start_frame <= 0:
            continue

        best: MergeCandidate | None = None
        for parent in stats:
            if parent.track_id == child.track_id:
                continue
            if parent.end_frame >= child.start_frame:
                continue

            gap = child.start_frame - parent.end_frame
            if gap <= 0 or gap > max_frame_gap:
                continue

            collision_frame = _nearest_collision_frame(collision_frames, parent.end_frame, child.start_frame, collision_padding)
            if collision_frame is None:
                continue

            distance = float(np.hypot(child.start_pos[0] - parent.end_pos[0], child.start_pos[1] - parent.end_pos[1]))
            if distance > max_distance_px:
                continue

            vec_heading = _heading_from_points(parent.end_pos, child.start_pos)
            dir_diffs = [
                _angle_diff_deg(parent.end_dir, vec_heading),
                _angle_diff_deg(child.start_dir, vec_heading),
            ]
            valid_diffs = [d for d in dir_diffs if not math.isinf(d)]
            direction_diff = min(valid_diffs) if valid_diffs else 0.0
            if direction_diff > direction_tolerance_deg:
                continue

            collision_delta_parent = abs(parent.end_frame - collision_frame)
            collision_delta_child = abs(child.start_frame - collision_frame)

            score = (
                (gap / max_frame_gap)
                + (distance / max_distance_px)
                + (direction_diff / direction_tolerance_deg)
                + ((collision_delta_parent + collision_delta_child) / max(1, 2 * collision_padding))
            )

            candidate = MergeCandidate(
                parent_track_id=parent.track_id,
                child_track_id=child.track_id,
                collision_frame=collision_frame,
                gap_frames=int(gap),
                distance_px=distance,
                direction_diff_deg=direction_diff,
                score=score,
                merge_type="collision",
            )
            if best is None or candidate.score < best.score:
                best = candidate

        if best is not None:
            merges.append(best)

    merges.sort(key=lambda m: (stats_by_id[m.child_track_id].start_frame, m.score))

    for merge in merges:
        collision_label = merge.collision_frame if merge.collision_frame is not None else "-"
        print(
            f"[{merge.merge_type}] Child {merge.child_track_id} <- Parent {merge.parent_track_id} | "
            f"Gap: {merge.gap_frames} frames | Distance: {merge.distance_px:.1f}px | "
            f"DirDiff: {merge.direction_diff_deg:.1f}° | Collision Frame: {collision_label} | "
            f"Score: {merge.score:.3f}"
        )

    return CollisionMergeResult(
        merges=merges,
        params={
            "area_threshold": area_threshold,
            "max_frame_gap": max_frame_gap,
            "max_distance_px": max_distance_px,
            "direction_tolerance_deg": direction_tolerance_deg,
            "collision_padding": collision_padding,
        },
    )


def find_directional_merges(
    zarr_path: str,
    max_frame_gap: int = 30,
    max_distance_px: float = 40.0,
    distance_per_gap_px: float = 6.0,
    direction_tolerance_deg: float = 35.0,
    direction_window: int = 5,
    max_score: float = 2.0,
) -> CollisionMergeResult:
    """
    Detect child->parent merges where a short gap exists but the motion
    direction is consistent across the gap (e.g., track briefly disappears).

    A valid merge must satisfy:
        - Child appears within `max_frame_gap` frames after parent ends.
        - The start/end displacement is within a dynamic distance threshold.
        - The heading between parent end and child start is aligned with the
          parent's terminal direction and/or the child's initial direction
          within the tolerance.
    """
    root = zarr.open_group(zarr_path, mode="r")

    track_ids = np.array(root["track"]).astype(int)
    x = np.array(root["x"])
    y = np.array(root["y"])

    # Compute direction from position if not stored
    dx = np.diff(x, axis=1, prepend=np.nan)
    dy = -np.diff(y, axis=1, prepend=np.nan)
    direction = (np.degrees(np.arctan2(dx, dy)) % 360.0).astype(float)

    stats = _collect_track_stats(x, y, direction, track_ids)
    stats_by_id = {s.track_id: s for s in stats}

    end_dirs = {
        s.track_id: _edge_direction_summary(direction[s.track_idx], x[s.track_idx], y[s.track_idx], s.end_frame, window=direction_window, forward=False)
        for s in stats
    }
    start_dirs = {
        s.track_id: _edge_direction_summary(direction[s.track_idx], x[s.track_idx], y[s.track_idx], s.start_frame, window=direction_window, forward=True)
        for s in stats
    }

    merges: List[MergeCandidate] = []

    for child in stats:
        if child.start_frame <= 0:
            continue

        child_start_dir = start_dirs.get(child.track_id, math.nan)
        best: MergeCandidate | None = None

        for parent in stats:
            if parent.track_id == child.track_id:
                continue
            if parent.end_frame >= child.start_frame:
                continue

            gap = child.start_frame - parent.end_frame
            if gap <= 0 or gap > max_frame_gap:
                continue

            allowed_distance = max_distance_px + distance_per_gap_px * max(0, gap - 1)
            distance = float(np.hypot(child.start_pos[0] - parent.end_pos[0], child.start_pos[1] - parent.end_pos[1]))
            if distance > allowed_distance:
                continue

            parent_end_dir = end_dirs.get(parent.track_id, math.nan)
            if math.isnan(parent_end_dir) and math.isnan(child_start_dir):
                continue

            vec_heading = _heading_from_points(parent.end_pos, child.start_pos)

            dir_diffs = []
            if not math.isnan(parent_end_dir):
                dir_diffs.append(_angle_diff_deg(parent_end_dir, vec_heading))
            if not math.isnan(child_start_dir):
                dir_diffs.append(_angle_diff_deg(child_start_dir, vec_heading))
            if not math.isnan(parent_end_dir) and not math.isnan(child_start_dir):
                dir_diffs.append(_angle_diff_deg(parent_end_dir, child_start_dir))

            direction_diff = min(dir_diffs) if dir_diffs else math.inf
            if direction_diff > direction_tolerance_deg:
                continue

            score = (
                (gap / max_frame_gap)
                + (distance / max(1.0, allowed_distance))
                + (direction_diff / max(1.0, direction_tolerance_deg))
            )
            if score > max_score:
                continue

            candidate = MergeCandidate(
                parent_track_id=parent.track_id,
                child_track_id=child.track_id,
                collision_frame=None,
                gap_frames=int(gap),
                distance_px=distance,
                direction_diff_deg=direction_diff,
                score=score,
                merge_type="direction",
            )

            if best is None or candidate.score < best.score:
                best = candidate

        if best is not None:
            merges.append(best)

    merges.sort(key=lambda m: (stats_by_id[m.child_track_id].start_frame, m.score))

    for merge in merges:
        collision_label = merge.collision_frame if merge.collision_frame is not None else "-"
        print(
            f"[{merge.merge_type}] Child {merge.child_track_id} <- Parent {merge.parent_track_id} | "
            f"Gap: {merge.gap_frames} frames | Distance: {merge.distance_px:.1f}px | "
            f"DirDiff: {merge.direction_diff_deg:.1f}° | Collision Frame: {collision_label} | "
            f"Score: {merge.score:.3f}"
        )

    return CollisionMergeResult(
        merges=merges,
        params={
            "max_frame_gap": max_frame_gap,
            "max_distance_px": max_distance_px,
            "distance_per_gap_px": distance_per_gap_px,
            "direction_tolerance_deg": direction_tolerance_deg,
            "direction_window": direction_window,
            "max_score": max_score,
        },
    )


def combine_merge_results(results: List[CollisionMergeResult]) -> CollisionMergeResult:
    """
    Combine multiple merge result sets, keeping the best parent per child
    track (lowest score wins).
    """
    merged: Dict[int, MergeCandidate] = {}
    for res in results:
        if res is None:
            continue
        for merge in res.merges:
            existing = merged.get(merge.child_track_id)
            if existing is None or merge.score < existing.score:
                merged[merge.child_track_id] = merge

    merged_list = sorted(merged.values(), key=lambda m: (m.child_track_id, m.score))
    return CollisionMergeResult(
        merges=merged_list,
        params={
            "combined_results": len([r for r in results if r is not None]),
            "combined_merges": len(merged_list),
        },
    )


def apply_merges_to_zarr(
    zarr_path: str,
    merge_result: CollisionMergeResult,
    output_path: str | None = None,
    array_prefix: str = "merged_",
    keep_original: bool = True,
    min_track_length: int = 0,
) -> None:
    """
    Apply merges by stitching tracks together into new merged tracks while
    preserving all original tracks (no data is dropped). Merged tracks are
    written with an optional prefix; originals can be copied into a dedicated
    subgroup so they remain untouched.

    Args:
        zarr_path: Source zarr with tracking data.
        merge_result: Output of `find_collision_merges`.
        output_path: Destination zarr path (if None, appends `_merged.zarr`).
        array_prefix: Optional prefix for merged track arrays in the new store.
        keep_original: If True, copies all original arrays to `original/`
                       inside the destination store for lossless reference.
        min_track_length: Minimum number of valid frames for a track to be kept.
                          Tracks shorter than this threshold are removed.
    """
    if not merge_result.merges:
        print("No merge candidates found; writing passthrough copy with originals preserved.")

    if output_path is None:
        base = zarr_path[:-5] if zarr_path.endswith(".zarr") else zarr_path
        output_path = f"{base}_merged.zarr"

    root = zarr.open_group(zarr_path, mode="r")
    track_ids = np.array(root["track"]).astype(int)
    id_to_index = {tid: idx for idx, tid in enumerate(track_ids)}

    if "x" not in root:
        raise ValueError("Required array 'x' not found in zarr store.")
    track_frame_arrays: Dict[str, np.ndarray] = {}
    track_only_arrays: Dict[str, np.ndarray] = {}
    for name in root.array_keys():
        if array_prefix and name.startswith(array_prefix):
            continue
        if name == "track":
            continue
        arr = root[name]
        dims = arr.attrs.get("_ARRAY_DIMENSIONS", [])
        if not dims or dims[0] != "track":
            continue
        if arr.shape[0] != track_ids.size:
            continue
        if arr.ndim >= 2:
            track_frame_arrays[name] = np.array(arr)
        elif arr.ndim == 1:
            track_only_arrays[name] = np.array(arr)
    dest = zarr.open_group(output_path, mode="w", zarr_format=getattr(root, "zarr_format", 2))
    dest.attrs.update(root.attrs)
    dest.attrs["merged_from"] = zarr_path
    dest.attrs["merged_prefix"] = array_prefix
    dest.attrs["n_original_tracks"] = int(track_ids.size)

    # Optionally keep untouched originals
    if keep_original:
        original = dest.create_group("original")
        for name in root.array_keys():
            arr = root[name]
            copied = original.create_array(name, data=np.array(arr))
            dims = arr.attrs.get("_ARRAY_DIMENSIONS", [])
            if dims:
                copied.attrs["_ARRAY_DIMENSIONS"] = list(dims)

    # Build merge groups (connected components of parent/child links)
    class UnionFind:
        def __init__(self, items: np.ndarray):
            self.parent = {int(i): int(i) for i in items}

        def find(self, x: int) -> int:
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]

        def union(self, a: int, b: int) -> None:
            ra, rb = self.find(a), self.find(b)
            if ra != rb:
                self.parent[rb] = ra

    uf = UnionFind(track_ids)
    applied_merges: List[MergeCandidate] = []
    for merge in merge_result.merges:
        if merge.parent_track_id not in id_to_index or merge.child_track_id not in id_to_index:
            continue
        uf.union(int(merge.parent_track_id), int(merge.child_track_id))
        applied_merges.append(merge)

    groups: Dict[int, List[int]] = {}
    for tid in track_ids:
        root_tid = uf.find(int(tid))
        groups.setdefault(root_tid, []).append(int(tid))

    x_array = track_frame_arrays.get("x")
    start_frames: Dict[int, int] = {}
    if x_array is not None:
        for tid in track_ids:
            idx = id_to_index[int(tid)]
            sf = _first_valid_index(x_array[idx])
            start_frames[int(tid)] = sf if sf is not None else math.inf

    def _order_key(track_id: int) -> tuple[float, int]:
        return (start_frames.get(track_id, math.inf), track_id)

    grouped_tracks = []
    for root_tid, tids in groups.items():
        ordered = sorted(tids, key=_order_key)
        first_frame = start_frames.get(ordered[0], math.inf)
        grouped_tracks.append((first_frame, ordered))

    grouped_tracks.sort(key=lambda t: (t[0], t[1][0]))

    merged_track_ids: List[int] = []
    merged_arrays: Dict[str, List[np.ndarray]] = {name: [] for name in track_frame_arrays}
    merged_track_only: Dict[str, List[np.ndarray]] = {name: [] for name in track_only_arrays}
    merged_sources: List[List[int]] = []

    for _, tids in grouped_tracks:
        indices = [id_to_index[tid] for tid in tids]
        merged_track_ids.append(int(tids[0]))
        merged_sources.append([int(t) for t in tids])

        for name, data in track_frame_arrays.items():
            if data.ndim == 2:
                combined = _combine_track_frames(data, indices)
            elif data.ndim == 3:
                combined = _combine_track_frames_3d(data, indices)
            else:
                raise ValueError(f"Unsupported array dimensions for {name}: {data.shape}")
            merged_arrays[name].append(combined)

        for name, data in track_only_arrays.items():
            merged_track_only[name].append(data[indices[0]])

    # Stack lists into arrays
    merged_arrays_stacked: Dict[str, np.ndarray] = {}
    for name, rows in merged_arrays.items():
        merged_arrays_stacked[name] = np.stack(rows, axis=0)
    merged_track_only_stacked: Dict[str, np.ndarray] = {}
    for name, vals in merged_track_only.items():
        merged_track_only_stacked[name] = np.array(vals)

    merged_track_ids_arr = np.array(merged_track_ids, dtype=int)

    # Filter out short tracks if min_track_length is specified
    if min_track_length > 0 and "x" in merged_arrays_stacked:
        x_merged = merged_arrays_stacked["x"]
        # Count valid (non-NaN) frames per track
        valid_counts = np.sum(~np.isnan(x_merged), axis=1)
        keep_mask = valid_counts >= min_track_length

        n_before = len(merged_track_ids_arr)
        n_removed = n_before - np.sum(keep_mask)
        if n_removed > 0:
            print(f"Removing {n_removed} tracks with fewer than {min_track_length} frames")

        # Filter all arrays
        for name in merged_arrays_stacked:
            merged_arrays_stacked[name] = merged_arrays_stacked[name][keep_mask]
        for name in merged_track_only_stacked:
            merged_track_only_stacked[name] = merged_track_only_stacked[name][keep_mask]
        merged_track_ids_arr = merged_track_ids_arr[keep_mask]
        merged_sources = [src for src, keep in zip(merged_sources, keep_mask) if keep]

    dest.attrs["merged_n_tracks"] = int(merged_track_ids_arr.size)

    # Persist merged track arrays with optional prefix
    for name, data in merged_arrays_stacked.items():
        out_name = f"{array_prefix}{name}" if array_prefix else name
        arr = dest.create_array(out_name, data=data)
        dims = root[name].attrs.get("_ARRAY_DIMENSIONS", [])
        if dims:
            arr.attrs["_ARRAY_DIMENSIONS"] = list(dims)
    for name, data in merged_track_only_stacked.items():
        out_name = f"{array_prefix}{name}" if array_prefix else name
        arr = dest.create_array(out_name, data=data)
        dims = root[name].attrs.get("_ARRAY_DIMENSIONS", [])
        if dims:
            arr.attrs["_ARRAY_DIMENSIONS"] = list(dims)

    # Save merged track ids (coordinate)
    track_out_name = f"{array_prefix}track" if array_prefix else "track"
    track_arr = dest.create_array(track_out_name, data=merged_track_ids_arr.astype(int))
    track_arr.attrs["_ARRAY_DIMENSIONS"] = ["track"]

    # Copy non-track arrays unchanged (e.g., frame_coord)
    written = set()
    for name in merged_arrays_stacked:
        written.add(f"{array_prefix}{name}" if array_prefix else name)
    for name in merged_track_only_stacked:
        written.add(f"{array_prefix}{name}" if array_prefix else name)
    written.add(track_out_name)

    for name in root.array_keys():
        if name in track_frame_arrays or name in track_only_arrays or name == "track":
            continue
        if array_prefix and name.startswith(array_prefix):
            continue
        if (array_prefix and f"{array_prefix}{name}" in written) or name in written:
            continue
        arr_data = np.array(root[name])
        arr = dest.create_array(name, data=arr_data)
        dims = root[name].attrs.get("_ARRAY_DIMENSIONS", [])
        if dims:
            arr.attrs["_ARRAY_DIMENSIONS"] = list(dims)

    # Store merge bookkeeping
    merge_group = dest.create_group("merge_results")
    merge_group.attrs.update(merge_result.params)
    merge_group.attrs["merged_track_ids"] = [int(tid) for tid in merged_track_ids_arr]
    merge_group.attrs["merged_sources"] = [
        {"merged_track_id": int(mid), "source_track_ids": srcs} for mid, srcs in zip(merged_track_ids_arr, merged_sources)
    ]

    parent_arr = merge_group.create_array("parent_track_id", data=np.array([m.parent_track_id for m in applied_merges], dtype=np.int32))
    child_arr = merge_group.create_array("child_track_id", data=np.array([m.child_track_id for m in applied_merges], dtype=np.int32))
    collision_arr = merge_group.create_array(
        "collision_frame",
        data=np.array([m.collision_frame if m.collision_frame is not None else -1 for m in applied_merges], dtype=np.int32),
    )
    gap_arr = merge_group.create_array("gap_frames", data=np.array([m.gap_frames for m in applied_merges], dtype=np.int32))
    distance_arr = merge_group.create_array("distance_px", data=np.array([m.distance_px for m in applied_merges], dtype=np.float32))
    dir_arr = merge_group.create_array("direction_diff_deg", data=np.array([m.direction_diff_deg for m in applied_merges], dtype=np.float32))
    score_arr = merge_group.create_array("score", data=np.array([m.score for m in applied_merges], dtype=np.float32))
    merge_type_arr = merge_group.create_array("merge_type", data=np.array([m.merge_type for m in applied_merges], dtype="U16"))

    for arr, name in [
        (parent_arr, "parent_track_id"),
        (child_arr, "child_track_id"),
        (collision_arr, "collision_frame"),
        (gap_arr, "gap_frames"),
        (distance_arr, "distance_px"),
        (dir_arr, "direction_diff_deg"),
        (score_arr, "score"),
        (merge_type_arr, "merge_type"),
    ]:
        arr.attrs["_ARRAY_DIMENSIONS"] = ["merge"]
