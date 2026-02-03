from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import math

from src.roi_selector import _compute_circle_from_three_points


DEFAULT_MAX_PREVIEW_FRAMES = 300
DEFAULT_PREVIEW_MAX_PIXELS = 512 * 512


def _determine_preview_shape(
    width: int,
    height: int,
    max_pixels: int,
) -> tuple[int, int, float, float]:
    """Determine preview dimensions and coordinate scaling factors."""

    total_pixels = width * height
    if total_pixels <= max_pixels or max_pixels <= 0:
        return width, height, 1.0, 1.0

    scale = math.sqrt(total_pixels / max_pixels)
    preview_width = max(1, int(round(width / scale)))
    preview_height = max(1, int(round(height / scale)))
    scale_x = width / preview_width
    scale_y = height / preview_height
    return preview_width, preview_height, scale_x, scale_y


def compute_median_intensity_projection(
    video_path: str,
    max_frames: int | None = None,
    max_preview_frames: int = DEFAULT_MAX_PREVIEW_FRAMES,
    preview_max_pixels: int = DEFAULT_PREVIEW_MAX_PIXELS,
) -> tuple[np.ndarray, float, float]:
    """Compute a memory-efficient median intensity projection for preview."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    preview_width, preview_height, scale_x, scale_y = _determine_preview_shape(
        width, height, preview_max_pixels
    )

    if max_frames is not None and total_frames:
        total_frames = min(total_frames, max_frames)

    if total_frames <= 0:
        # Fallback: manually count frames up to max_frames
        total_frames = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            if max_frames is not None and total_frames >= max_frames:
                break
            ret, _ = cap.read()
            if not ret:
                break
            total_frames += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not reopen video: {video_path}")

    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no readable frames")

    sample_count = total_frames
    if max_preview_frames is not None:
        sample_count = min(sample_count, max_preview_frames)

    indices = np.linspace(0, total_frames - 1, sample_count).astype(int)
    samples = np.empty((sample_count, preview_height, preview_width), dtype=np.uint8)

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to read frame while computing median preview")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if preview_width != width or preview_height != height:
            gray = cv2.resize(gray, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
        samples[i] = gray

    cap.release()

    median_frame = np.median(samples, axis=0).astype(np.uint8)
    return median_frame, scale_x, scale_y


def select_bounding_box_from_frame(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Show OpenCV ROI selector on the provided frame and return (x, y, w, h)."""

    if frame.ndim == 2:
        display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        display = frame.copy()

    roi = cv2.selectROI(
        "Select Bounding Box", display, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Select Bounding Box")
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        raise ValueError("Bounding box selection cancelled or invalid.")
    return int(x), int(y), int(w), int(h)


def run_bounding_box_selection(video_path: str, max_frames: int | None = None) -> dict:
    """Compute median projection and prompt the user to select a bounding box."""

    median_frame, scale_x, scale_y = compute_median_intensity_projection(
        video_path, max_frames=max_frames
    )
    bbox_preview = select_bounding_box_from_frame(median_frame)

    x, y, w, h = bbox_preview
    x_orig = int(round(x * scale_x))
    y_orig = int(round(y * scale_y))
    w_orig = int(round(w * scale_x))
    h_orig = int(round(h * scale_y))

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "mode": "bounding_box",
        "height": height,
        "width": width,
        "bbox": (x_orig, y_orig, w_orig, h_orig),
    }


def _draw_instructions(frame: np.ndarray, lines: list[str]) -> None:
    y = 24
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        y += 26


def select_circle_roi_from_frame(frame: np.ndarray) -> tuple[tuple[int, int], int]:
    """Interactive circle selector using three-point boundary definition."""

    if frame.ndim == 2:
        base = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        base = frame.copy()

    window = "Select Circular ROI"
    points: list[tuple[int, int]] = []
    center: tuple[float, float] | None = None
    radius: float | None = None
    confirmed = False
    cancelled = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, center, radius, confirmed
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            if len(points) == 3:
                computed_center, computed_radius = _compute_circle_from_three_points(points)
                if computed_center is None or computed_radius is None:
                    print("Points are collinear; please select three non-collinear points.")
                    points.clear()
                    center = None
                    radius = None
                else:
                    center = computed_center
                    radius = computed_radius
                    confirmed = True

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    try:
        while True:
            overlay = base.copy()
            for px, py in points:
                cv2.circle(overlay, (int(px), int(py)), 4, (0, 0, 255), -1)
            if len(points) >= 2:
                pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(overlay, [pts], False, (0, 200, 255), 1)

            if center is not None and radius is not None:
                center_int = (int(round(center[0])), int(round(center[1])))
                radius_int = max(int(round(radius)), 1)
                cv2.circle(overlay, center_int, radius_int, (0, 255, 0), 2)
                cv2.circle(overlay, center_int, 3, (0, 0, 255), -1)

            _draw_instructions(
                overlay,
                [
                    "Circle ROI: click 3 edge points",
                    f"Points selected: {len(points)}/3",
                    "Press 'z' to undo, 'r' to reset, ESC to cancel",
                ],
            )

            cv2.imshow(window, overlay)
            key = cv2.waitKey(20) & 0xFF
            if key == 27:  # ESC
                cancelled = True
                break
            elif key in (ord('r'), ord('R')):
                points.clear()
                center = None
                radius = None
                confirmed = False
            elif key in (ord('z'), ord('Z'), 8, 127):  # Undo/backspace
                if points:
                    points.pop()
                center = None
                radius = None
                confirmed = False

            if confirmed:
                break

        if cancelled:
            raise ValueError("Circular ROI selection cancelled")

        if center is None or radius is None or radius <= 0:
            raise ValueError("Circular ROI selection incomplete")
        return (int(round(center[0])), int(round(center[1]))), int(round(radius))
    finally:
        cv2.destroyWindow(window)


def run_circle_roi_selection(video_path: str, max_frames: int | None = None) -> dict:
    """Compute median projection and prompt the user to select a circular ROI."""

    median_frame, scale_x, scale_y = compute_median_intensity_projection(
        video_path, max_frames=max_frames
    )
    center_preview, radius_preview = select_circle_roi_from_frame(median_frame)

    # Scale back to original coordinates
    cx, cy = center_preview
    cx_orig = float(cx * scale_x)
    cy_orig = float(cy * scale_y)
    scale_radius = (scale_x + scale_y) / 2.0
    radius_orig = float(max(1.0, radius_preview * scale_radius))

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "mode": "circle",
        "height": height,
        "width": width,
        "center": (cx_orig, cy_orig),
        "radius": radius_orig,
    }
