from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import click
from skimage.measure import regionprops

from .tracker import TrackingParameters
from .tracking import get_roi_mask_for_video


# Valid feature labels for annotation
FEATURE_TYPES = ("mouth", "gonad", "tentacle_bulb")
AREA_MARGIN = 0.3
ASPECT_MARGIN = 0.2
ECCENTRICITY_MARGIN = 0.1
SOLIDITY_MARGIN = 0.1
SEARCH_RADIUS_MARGIN = 1.5
THRESHOLD_BUFFER = 2


@dataclass
class FeatureSample:
    feature_type: str
    frame_idx: int
    centroid: Tuple[float, float]
    area: float
    bbox: Tuple[int, int, int, int]
    aspect_ratio: float
    eccentricity: float
    solidity: float


def compute_median_background(
    video_path: str,
    num_samples: int = 60,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, int, int, int]:
    """Compute a simple median background from evenly spaced frames."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_frames == 0:
        raise ValueError("Video contains no frames")

    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    num_samples = min(num_samples, total_frames)
    indices = np.linspace(0, total_frames - 1, num_samples).astype(int)

    sample_stack: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sample_stack.append(gray.astype(np.float32))

    cap.release()

    if not sample_stack:
        raise ValueError("Unable to compute median background")

    median_bg = np.median(np.stack(sample_stack, axis=0), axis=0).astype(np.uint8)
    return median_bg, width, height, int(fps)


def _extract_feature_sample(
    component_mask: np.ndarray,
    feature_type: str,
    frame_idx: int,
) -> FeatureSample | None:
    props = regionprops(component_mask.astype(np.uint8))
    if not props:
        return None
    region = props[0]
    min_row, min_col, max_row, max_col = region.bbox
    bbox = (
        int(min_col),
        int(min_row),
        int(max_col - min_col),
        int(max_row - min_row),
    )
    aspect_ratio = (
        float(region.axis_major_length / region.axis_minor_length)
        if region.axis_minor_length > 0
        else float("inf")
    )

    return FeatureSample(
        feature_type=feature_type,
        frame_idx=frame_idx,
        centroid=(float(region.centroid[1]), float(region.centroid[0])),
        area=float(region.area),
        bbox=bbox,
        aspect_ratio=aspect_ratio,
        eccentricity=float(region.eccentricity),
        solidity=float(region.solidity),
    )


class FeatureSamplingUI:
    def __init__(
        self,
        video_path: str,
        params: TrackingParameters,
        max_frames: Optional[int] = None,
    ) -> None:
        self.video_path = video_path
        self.params = params
        self.max_frames = max_frames

        self.background, self.width, self.height, _ = compute_median_background(
            video_path, max_frames=max_frames
        )

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if self.max_frames is not None:
            self.total_frames = min(self.total_frames, self.max_frames)

        self.current_frame_idx = 0
        self.gray_frame: np.ndarray | None = None
        self.color_frame: np.ndarray | None = None
        self.diff: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.labels: np.ndarray | None = None

        self.video_type = getattr(params, "video_type", "non_rotating") or "non_rotating"
        self.roi_mask = get_roi_mask_for_video(params, self.height, self.width, self.video_type)

        self.threshold_value = max(0, min(255, int(params.threshold)))

        self.mouth_pinned = params.mouth_pinned
        self.available_features = [
            ft for ft in FEATURE_TYPES if not (self.mouth_pinned and ft == "mouth")
        ]
        if not self.available_features:
            self.available_features = ["gonad"]
        self.current_feature = self.available_features[0]
        self.samples: Dict[str, List[FeatureSample]] = {ft: [] for ft in FEATURE_TYPES}
        self.status_message = "Click red regions to label features"
        self._display_shape: tuple[int, int] = (self.width * 3, self.height)
        self._cursor_screen_pos: tuple[int, int] | None = None
        self._cursor_image_pos: tuple[int, int] | None = None
        self._cursor_mask_value: int | None = None

        self.window_name = "Feature Sampling"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._handle_mouse)
        cv2.createTrackbar(
            "Threshold",
            self.window_name,
            self.threshold_value,
            255,
            self._handle_trackbar,
        )

        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self._load_frame(0)

    @property
    def threshold(self) -> int:
        return int(self.threshold_value)

    def _handle_trackbar(self, value: int) -> None:  # pragma: no cover - UI callback
        self.threshold_value = value
        self.params.threshold = self.threshold
        self._update_diff()
        self.status_message = f"Threshold set to {self.threshold}"

    def _load_frame(self, idx: int) -> None:
        idx = max(0, min(idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Unable to read frame for sampling")

        self.current_frame_idx = idx
        self.color_frame = frame
        self.gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._update_diff()

    def _update_diff(self) -> None:
        if self.gray_frame is None:
            return
        diff = cv2.absdiff(self.gray_frame, self.background)
        if self.roi_mask is not None:
            diff = cv2.bitwise_and(diff, self.roi_mask)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)
        self.diff = diff
        self.mask = mask
        if np.count_nonzero(mask) > 0:
            labels = cv2.connectedComponentsWithStats(mask, connectivity=8)
            self.labels = labels[1]
        else:
            self.labels = None

    def _handle_mouse(self, event, x, y, flags, param):  # pragma: no cover - UI callback
        self._cursor_screen_pos = (x, y)
        image_coords = self._map_screen_to_image_coords(x, y)

        if image_coords is None:
            self._cursor_image_pos = None
            self._cursor_mask_value = None
            if event == cv2.EVENT_LBUTTONDOWN:
                self.status_message = "Click inside the diff pane (right side)"
            return

        local_x, local_y = image_coords
        self._cursor_image_pos = (local_x, local_y)
        if self.mask is not None:
            self._cursor_mask_value = int(self.mask[local_y, local_x])
        else:
            self._cursor_mask_value = None

        if event == cv2.EVENT_LBUTTONDOWN:
            self._label_component(local_x, local_y)

    def _label_component(self, local_x: int, local_y: int) -> None:
        if self.mask is None or self.labels is None:
            self.status_message = "Diff mask not ready"
            return
        if local_x < 0 or local_x >= self.width or local_y < 0 or local_y >= self.height:
            return
        if self.mask[local_y, local_x] == 0:
            self.status_message = "Clicked region is below threshold"
            return

        label_id = int(self.labels[local_y, local_x])
        if label_id <= 0:
            self.status_message = "Clicked background"
            return

        component_mask = (self.labels == label_id).astype(np.uint8)
        sample = _extract_feature_sample(component_mask, self.current_feature, self.current_frame_idx)
        if sample is None:
            self.status_message = "Unable to measure selected region"
            return

        self.samples[self.current_feature].append(sample)
        self.status_message = f"Added {self.current_feature} sample (area={sample.area:.0f})"

    def _map_screen_to_image_coords(self, screen_x: int, screen_y: int) -> tuple[int, int] | None:
        if not hasattr(self, "_last_crop"):
            return None

        display_width, display_height = self._get_window_size()
        stacked_width = self.width * 3
        stacked_height = self.height

        array_x = int(screen_x * stacked_width / max(1, display_width))
        array_y = int(screen_y * stacked_height / max(1, display_height))

        pan_x, pan_y, view_width, view_height = self._last_crop

        orig_x = pan_x + (array_x * view_width / stacked_width)
        orig_y = pan_y + (array_y * view_height / stacked_height)

        if orig_x < 2 * self.width or orig_x >= 3 * self.width:
            return None
        if orig_y < 0 or orig_y >= self.height:
            return None

        local_x = int(orig_x - 2 * self.width)
        local_y = int(orig_y)
        local_x = max(0, min(self.width - 1, local_x))
        local_y = max(0, min(self.height - 1, local_y))
        return local_x, local_y

    def _image_to_display_coords(self, local_x: int, local_y: int) -> tuple[int, int] | None:
        if not hasattr(self, "_last_crop"):
            return None
        pan_x, pan_y, view_width, view_height = self._last_crop
        stacked_width = self.width * 3
        stacked_height = self.height

        orig_x = local_x + 2 * self.width
        orig_y = local_y

        if not (pan_x <= orig_x < pan_x + view_width):
            return None
        if not (pan_y <= orig_y < pan_y + view_height):
            return None

        array_x = int(((orig_x - pan_x) * stacked_width) / view_width)
        array_y = int(((orig_y - pan_y) * stacked_height) / view_height)
        array_x = max(0, min(self._display_shape[0] - 1, array_x))
        array_y = max(0, min(self._display_shape[1] - 1, array_y))
        return array_x, array_y

    def _get_window_size(self) -> tuple[int, int]:
        display_width, display_height = self._display_shape
        try:
            rect = cv2.getWindowImageRect(self.window_name)
            if rect[2] > 0 and rect[3] > 0:
                display_width = rect[2]
                display_height = rect[3]
        except Exception:  # pragma: no cover - OpenCV may not support this call
            pass
        return display_width, display_height

    def _cycle_threshold(self, step: int = 5) -> None:
        self.threshold_value = max(0, min(255, self.threshold_value + step))
        cv2.setTrackbarPos("Threshold", self.window_name, self.threshold_value)
        self.params.threshold = self.threshold
        self._update_diff()
        self.status_message = f"Threshold set to {self.threshold}"

    def _compose_display(self) -> np.ndarray:
        if self.gray_frame is None or self.diff is None or self.mask is None:
            return np.zeros((self.height, self.width * 3, 3), dtype=np.uint8)

        raw = cv2.cvtColor(self.gray_frame, cv2.COLOR_GRAY2BGR)
        bg = cv2.cvtColor(self.background, cv2.COLOR_GRAY2BGR)
        diff_display = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        diff_display[self.mask > 0] = (0, 0, 255)

        self._draw_annotations(raw)
        stacked = np.hstack([raw, bg, diff_display])
        view, crop_metadata = self._apply_zoom_pan(stacked)
        self._last_crop = crop_metadata
        self._display_shape = (view.shape[1], view.shape[0])

        overlay = view.copy()
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 110), (0, 0, 0), -1)
        blended = cv2.addWeighted(overlay, 0.4, view, 0.6, 0)

        def _draw_text(img, text, pos, color):
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        status_text = (
            f"Frame {self.current_frame_idx + 1}/{self.total_frames} | Feature: {self.current_feature.upper()}"
            f" | Threshold: {self.threshold}"
        )
        _draw_text(blended, status_text, (10, 35), (0, 255, 255))
        cursor_info = ""
        if self._cursor_mask_value is not None:
            cursor_info = f" | Mask:{self._cursor_mask_value}"
        _draw_text(blended, self.status_message + cursor_info, (10, 65), (255, 255, 255))
        feature_keys = "/".join(ft[0] for ft in self.available_features)
        instructions = (
            f"{feature_keys} change feature | n/p frame | slider/b threshold | +/- zoom | arrows pan | u undo | c clear | q quit"
        )
        _draw_text(blended, instructions, (10, 95), (173, 255, 47))

        if self._cursor_image_pos is not None:
            display_coords = self._image_to_display_coords(*self._cursor_image_pos)
            if display_coords is not None:
                cv2.drawMarker(
                    blended,
                    display_coords,
                    (0, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2,
                )

        return blended

    def _apply_zoom_pan(
        self, stacked: np.ndarray
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        zoom = max(1.0, min(self.zoom, 5.0))
        view_width = int(stacked.shape[1] / zoom)
        view_height = int(stacked.shape[0] / zoom)
        view_width = max(1, min(view_width, stacked.shape[1]))
        view_height = max(1, min(view_height, stacked.shape[0]))

        max_x = stacked.shape[1] - view_width
        max_y = stacked.shape[0] - view_height
        self.pan_x = max(0, min(int(self.pan_x), max_x))
        self.pan_y = max(0, min(int(self.pan_y), max_y))

        crop = stacked[
            self.pan_y : self.pan_y + view_height,
            self.pan_x : self.pan_x + view_width,
        ]
        resized = cv2.resize(
            crop,
            (stacked.shape[1], stacked.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        return resized, (self.pan_x, self.pan_y, view_width, view_height)

    def _adjust_zoom(self, factor: float) -> None:
        center_x = self.pan_x + int((self.width * 1.5) / self.zoom)
        center_y = self.pan_y + int((self.height) / self.zoom)
        self.zoom = max(1.0, min(5.0, self.zoom * factor))
        view_width = int((self.width * 3) / self.zoom)
        view_height = int(self.height / self.zoom)
        self.pan_x = center_x - view_width // 2
        self.pan_y = center_y - view_height // 2

    def _pan(self, dx: int, dy: int) -> None:
        self.pan_x += dx
        self.pan_y += dy

    def _draw_annotations(self, canvas: np.ndarray) -> None:
        for feature, samples in self.samples.items():
            color = {
                "mouth": (0, 165, 255),
                "gonad": (255, 0, 255),
                "tentacle_bulb": (255, 200, 0),
            }.get(feature, (255, 255, 255))
            for sample in samples:
                x, y = int(sample.centroid[0]), int(sample.centroid[1])
                cv2.circle(canvas, (x, y), 4, color, -1)
                cv2.putText(
                    canvas,
                    feature[0].upper(),
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
        self._draw_roi(canvas)

    def _draw_roi(self, canvas: np.ndarray) -> None:
        roi_params = self.params.get_roi_params()
        mode = roi_params.get("mode", "auto")
        if mode == "circle" and roi_params.get("center") and roi_params.get("radius"):
            center = tuple(map(int, map(round, roi_params["center"])))
            radius = max(int(round(roi_params["radius"])), 1)
            cv2.circle(canvas, center, radius, (128, 128, 128), 2)
        elif mode == "polygon" and roi_params.get("points"):
            pts = np.array(
                [tuple(int(round(v)) for v in pt) for pt in roi_params["points"]],
                dtype=np.int32,
            )
            cv2.polylines(canvas, [pts], True, (128, 128, 128), 2)
        elif self.video_type == "rotating":
            center = (self.width // 2, self.height // 2)
            radius = min(self.width, self.height) // 2
            cv2.circle(canvas, center, radius, (128, 128, 128), 1)

    def _change_feature(self, feature: str) -> None:
        if feature in self.available_features:
            self.current_feature = feature
            self.status_message = f"Feature: {feature}"
        else:
            self.status_message = f"Feature '{feature}' disabled"

    def _undo(self) -> None:
        samples = self.samples[self.current_feature]
        if samples:
            samples.pop()
            self.status_message = "Removed last sample"
        else:
            self.status_message = "No samples to remove"

    def _clear_feature(self) -> None:
        self.samples[self.current_feature].clear()
        self.status_message = f"Cleared {self.current_feature} samples"

    def run(self) -> Dict[str, List[FeatureSample]]:
        while True:
            display = self._compose_display()
            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                break
            if key == ord("b"):
                self._cycle_threshold(step=5)
            elif key == ord("B"):
                self._cycle_threshold(step=-5)
            elif key == ord("m"):
                if self.mouth_pinned:
                    self.status_message = "Mouth annotations disabled (pinned mode)"
                else:
                    self._change_feature("mouth")
            elif key == ord("g"):
                self._change_feature("gonad")
            elif key == ord("t"):
                self._change_feature("tentacle_bulb")
            elif key in (ord("n"), ord("]")):
                self._load_frame(self.current_frame_idx + 1)
            elif key in (ord("p"), ord("[")):
                self._load_frame(self.current_frame_idx - 1)
            elif key in (ord("+"), ord("=")):
                self._adjust_zoom(1.2)
            elif key == ord("-"):
                self._adjust_zoom(0.8)
            elif key in (ord("h"), ord("a")):
                self._pan(-40, 0)
            elif key in (ord("l"), ord("d")):
                self._pan(40, 0)
            elif key in (ord("k"), ord("w")):
                self._pan(0, -40)
            elif key in (ord("j"), ord("s")):
                self._pan(0, 40)
            elif key == ord("u"):
                self._undo()
            elif key == ord("c"):
                self._clear_feature()

        cv2.destroyWindow(self.window_name)
        self.cap.release()
        self.params.threshold = self.threshold
        return self.samples


def _apply_samples_to_parameters(
    params: TrackingParameters,
    samples: Dict[str, List[FeatureSample]],
) -> TrackingParameters:
    params.threshold = max(1, int(params.threshold - THRESHOLD_BUFFER))
    params.use_auto_threshold = False

    def _apply_feature(feature: str) -> None:
        feature_samples = samples.get(feature, [])
        if not feature_samples:
            return
        areas = [sample.area for sample in feature_samples]
        aspect_ratios = [
            sample.aspect_ratio
            for sample in feature_samples
            if math.isfinite(sample.aspect_ratio)
        ]
        eccentricities = [sample.eccentricity for sample in feature_samples]
        solidities = [sample.solidity for sample in feature_samples]

        config = params.object_types.setdefault(feature, {})
        config["min_area"] = max(1, int(math.floor(min(areas) * (1 - AREA_MARGIN))))
        config["max_area"] = int(math.ceil(max(areas) * (1 + AREA_MARGIN)))
        if aspect_ratios:
            config["aspect_ratio_min"] = round(min(aspect_ratios) * (1 - ASPECT_MARGIN), 2)
            config["aspect_ratio_max"] = round(max(aspect_ratios) * (1 + ASPECT_MARGIN), 2)
        if eccentricities:
            config["eccentricity_min"] = round(max(0.0, min(eccentricities) - ECCENTRICITY_MARGIN), 2)
            config["eccentricity_max"] = round(min(0.999, max(eccentricities) + ECCENTRICITY_MARGIN), 2)
        if solidities:
            config["solidity_min"] = round(max(0.0, min(solidities) - SOLIDITY_MARGIN), 2)

    for feature in FEATURE_TYPES:
        _apply_feature(feature)

    mouth_samples = samples.get("mouth", [])
    mouth_center = (
        np.mean([sample.centroid for sample in mouth_samples], axis=0)
        if mouth_samples
        else None
    )
    if mouth_center is None and params.mouth_pinned and params.pinned_mouth_point is not None:
        mouth_center = np.array(params.pinned_mouth_point, dtype=float)

    def _set_search_radius(feature: str) -> None:
        if mouth_center is None:
            return
        feature_samples = samples.get(feature, [])
        if not feature_samples:
            return
        dists = [math.dist(sample.centroid, mouth_center) for sample in feature_samples]
        radius = int(math.ceil(max(dists) * SEARCH_RADIUS_MARGIN))
        if radius > 0:
            params.object_types.setdefault(feature, {})["search_radius"] = radius

    _set_search_radius("gonad")
    _set_search_radius("tentacle_bulb")

    params.update_object_counts()
    return params


def run_interactive_feature_sampling(
    video_path: str,
    params: TrackingParameters | None = None,
    max_frames: int | None = None,
) -> TrackingParameters:
    """Launch the feature sampling UI and return updated tracking parameters."""

    if params is None:
        params = TrackingParameters()

    click.echo("\n=== Feature Sampling UI ===")
    click.echo("Left pane: raw | middle: background | right: diff (red = above threshold)")
    if getattr(params, "mouth_pinned", False):
        click.echo("Pinned mouth mode: mouth annotations are disabled; focus on gonads/bulbs.")
    click.echo("Use feature hotkeys, n/p to change frame, and the slider or b/B to adjust threshold")

    ui = FeatureSamplingUI(video_path, params, max_frames=max_frames)
    samples = ui.run()

    updated_params = _apply_samples_to_parameters(params, samples)
    click.echo("Feature sampling complete. Updated parameters:")
    click.echo(f"  Threshold: {updated_params.threshold}")
    for feature in FEATURE_TYPES:
        config = updated_params.object_types.get(feature, {})
        if config:
            click.echo(
                f"  {feature}: min_area={config.get('min_area')} max_area={config.get('max_area')} (samples={len(samples.get(feature, []))})"
            )

    return updated_params
