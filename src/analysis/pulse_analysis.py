from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr

from src.save_results import load_multi_object_tracking_from_zarr
from src.tracker import TrackingData


@dataclass
class PulseAnalysisResult:
    frame_indices: np.ndarray
    time_seconds: np.ndarray
    mean_distance_px: np.ndarray
    fps: float
    object_type: str


def _read_fps(zarr_path: str | Path) -> float:
    root = zarr.open_group(str(zarr_path), mode="r")
    fps = root.attrs.get("fps")
    try:
        return float(fps)
    except (TypeError, ValueError):
        return 0.0


def compute_pulse_distance_from_tracking(
    tracking: TrackingData,
    *,
    fps: float,
    object_type: str,
) -> PulseAnalysisResult:
    """Compute pulse distance statistics from an in-memory TrackingData object."""

    x = tracking.x.astype(float)
    y = tracking.y.astype(float)
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    counts = valid_mask.sum(axis=0)

    sum_x = np.nansum(x, axis=0)
    sum_y = np.nansum(y, axis=0)
    center_x = np.divide(
        sum_x,
        counts,
        out=np.full_like(sum_x, np.nan, dtype=float),
        where=counts > 0,
    )
    center_y = np.divide(
        sum_y,
        counts,
        out=np.full_like(sum_y, np.nan, dtype=float),
        where=counts > 0,
    )

    dx = x - center_x
    dy = y - center_y
    distances = np.sqrt(dx**2 + dy**2)
    distances[~valid_mask] = np.nan

    with np.errstate(all="ignore"):
        mean_distance = np.nanmean(distances, axis=0)

    mean_distance[counts == 0] = np.nan
    frame_indices = np.arange(tracking.n_frames, dtype=int)
    fps_value = float(fps) if isinstance(fps, (int, float)) else 0.0
    if fps_value > 0:
        time_axis = frame_indices / fps_value
    else:
        time_axis = frame_indices.astype(float)

    return PulseAnalysisResult(
        frame_indices=frame_indices,
        time_seconds=time_axis,
        mean_distance_px=mean_distance,
        fps=fps_value,
        object_type=object_type,
    )


def compute_pulse_distance_series(
    zarr_path: str | Path,
    *,
    object_type: str = "tentacle_bulb",
) -> PulseAnalysisResult:
    """Compute mean bulb distance from center-of-mass for every frame."""

    zarr_path = str(zarr_path)
    tracking_results = load_multi_object_tracking_from_zarr(zarr_path)
    if object_type not in tracking_results:
        available = ", ".join(sorted(tracking_results.keys()))
        raise ValueError(
            f"Object type '{object_type}' not found in tracking results. Available: {available}"
        )

    tracking = tracking_results[object_type]
    fps = _read_fps(zarr_path)
    return compute_pulse_distance_from_tracking(
        tracking,
        fps=fps,
        object_type=object_type,
    )


def save_pulse_plot(
    result: PulseAnalysisResult,
    *,
    output_path: str | Path | None,
    show: bool,
) -> None:
    """Render and optionally save the pulse distance plot."""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(result.time_seconds, result.mean_distance_px, label=result.object_type)
    ax.set_ylabel("Mean distance (px)")
    ax.set_xlabel("Time (s)" if result.fps > 0 else "Frame")
    ax.set_title("Tentacle bulb pulse distance")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Pulse plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_pulse_csv(result: PulseAnalysisResult, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack(
        [result.frame_indices, result.time_seconds, result.mean_distance_px]
    )
    header = "frame,time_seconds,mean_distance_px"
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")
    print(f"Pulse data saved to {output_path}")
