from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import click
from skimage.measure import regionprops

from .tracker import TrackingParameters

FULL_WEIGHT_SAMPLES = 5
MIN_MOUTH_SAMPLES = 1


@dataclass
class SampledFeature:
    object_type: str
    centroid: Tuple[float, float]
    area: float
    aspect_ratio: float
    eccentricity: float
    solidity: float
    major_axis_length_mm: float
    minor_axis_length_mm: float


@dataclass
class AnnotationRegion:
    feature_type: str
    frame_idx: int
    points: List[Tuple[float, float]]
    shape: str = "polygon"
    circle_center: Tuple[float, float] | None = None
    circle_radius: float | None = None


def compute_median_frame(
    video_path: str,
    sample_count: int = 200,
    max_frames: Optional[int] = None,
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames is not None and total_frames:
        total_frames = min(total_frames, max_frames)
    if total_frames <= 0:
        cap.release()
        raise ValueError("Video contains no frames")

    indices = np.linspace(0, total_frames - 1, min(sample_count, total_frames)).astype(int)
    samples: List[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(gray.astype(np.float32))

    cap.release()
    if not samples:
        raise ValueError("Unable to compute median frame")

    median_frame = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)
    return median_frame


class FeatureAnnotationApp:
    def __init__(
        self,
        video_path: str,
        params: TrackingParameters,
        pixel_size_mm: float,
    ) -> None:
        self.video_path = video_path
        self.params = params
        self.pixel_size_mm = pixel_size_mm
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0
        self.current_frame: np.ndarray | None = None
        self.current_diff: np.ndarray | None = None
        self.show_diff = True
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.window_name = "Feature Annotation"
        self.current_points: List[Tuple[float, float]] = []
        self.current_feature = "mouth"
        self.annotations: List[AnnotationRegion] = []
        self.status_message = "Polygon mode: click to add vertices, Enter to finalize"
        self.median_frame = compute_median_frame(video_path)
        self.drawing_mode = "polygon"  # or "circle"
        self.circle_center: Tuple[float, float] | None = None
        self.circle_radius: float | None = None
        self.is_drawing_circle = False
        self.threshold_options = [5, 10, 15, 20, 30, 40, 60, 80]
        if params.threshold not in self.threshold_options:
            self.threshold_options.append(params.threshold)
            self.threshold_options = sorted(set(self.threshold_options))
        self.threshold_index = max(
            0,
            self.threshold_options.index(
                min(self.threshold_options, key=lambda t: abs(t - params.threshold))
            ),
        )

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _read_frame(self, idx: int) -> bool:
        if idx < 0 or idx >= self.total_frames:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return False
        self.current_frame_idx = idx
        self.current_frame = frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, self.median_frame)
        norm_diff = np.zeros_like(diff, dtype=np.uint8)
        cv2.normalize(diff, dst=norm_diff, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        self.current_diff = cv2.applyColorMap(norm_diff, cv2.COLORMAP_MAGMA)
        self.current_points.clear()
        self.circle_center = None
        self.circle_radius = None
        self.is_drawing_circle = False
        return True

    def _apply_zoom_pan(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if self.zoom <= 1.0:
            self.pan_x = 0
            self.pan_y = 0
            return img
        scaled = cv2.resize(img, None, fx=self.zoom, fy=self.zoom, interpolation=cv2.INTER_LINEAR)
        view_w, view_h = w, h
        max_x = max(0, scaled.shape[1] - view_w)
        max_y = max(0, scaled.shape[0] - view_h)
        self.pan_x = int(min(max(self.pan_x, 0), max_x))
        self.pan_y = int(min(max(self.pan_y, 0), max_y))
        return scaled[self.pan_y : self.pan_y + view_h, self.pan_x : self.pan_x + view_w]

    def _map_display_to_image(self, x: int, y: int) -> Tuple[int, int]:
        if self.zoom <= 1.0:
            return x, y
        return int((self.pan_x + x) / self.zoom), int((self.pan_y + y) / self.zoom)

    def _draw_annotations(self, canvas: np.ndarray) -> None:
        for region in self.annotations:
            if region.frame_idx != self.current_frame_idx:
                continue
            color = self._color_for_type(region.feature_type)
            if region.shape == "circle" and region.circle_center and region.circle_radius:
                c = (int(round(region.circle_center[0])), int(round(region.circle_center[1])))
                r = max(1, int(round(region.circle_radius)))
                cv2.circle(canvas, c, r, color, 2)
                cv2.circle(canvas, c, 4, color, -1)
                cv2.putText(canvas, region.feature_type[0].upper(), (c[0] + 5, c[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                pts = np.array(region.points, dtype=np.int32)
                if pts.size == 0:
                    continue
                cv2.polylines(canvas, [pts], True, color, 2)
                mask = np.zeros((self.height, self.width), dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                moments = cv2.moments(mask)
                if moments["m00"]:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    cv2.circle(canvas, (cx, cy), 4, color, -1)
                    cv2.putText(canvas, region.feature_type[0].upper(), (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        if self.drawing_mode == "polygon" and self.current_points:
            pts = np.array(self.current_points, dtype=np.int32)
            color = self._color_for_type(self.current_feature)
            cv2.polylines(canvas, [pts], False, color, 1)
            for point in self.current_points:
                cv2.circle(canvas, (int(point[0]), int(point[1])), 3, color, -1)
        elif self.drawing_mode == "circle" and self.circle_center is not None and self.circle_radius is not None:
            color = self._color_for_type(self.current_feature)
            c = (int(round(self.circle_center[0])), int(round(self.circle_center[1])))
            r = max(1, int(round(self.circle_radius)))
            cv2.circle(canvas, c, r, color, 1)
            cv2.circle(canvas, c, 3, color, -1)
            cv2.putText(canvas, f"r={r}px", (c[0] + 5, c[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    @staticmethod
    def _color_for_type(feature_type: str) -> Tuple[int, int, int]:
        return {
            "mouth": (0, 165, 255),
            "gonad": (255, 0, 255),
            "tentacle_bulb": (0, 255, 255),
        }.get(feature_type, (255, 255, 255))

    def _render(self) -> None:
        if self.current_frame is None:
            return
        base = self.current_diff if self.show_diff and self.current_diff is not None else self.current_frame
        canvas = base.copy()
        self._draw_annotations(canvas)
        overlay = canvas
        if self.zoom > 1.0:
            overlay = self._apply_zoom_pan(canvas)
        cv2.putText(
            overlay,
            f"Frame {self.current_frame_idx + 1}/{self.total_frames} | Feature: {self.current_feature.upper()} | Zoom: {self.zoom:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        hud_lines = [
            self.status_message,
            f"Threshold: {self.params.threshold}px (press 'b' to cycle)",
            "Controls: m/g/t switch feature | o toggle circle mode",
            "Click to add vertices (polygon) or click+drag (circle)",
            "+/− zoom | h/j/k/l pan | d diff/raw | Enter finalize polygon | u undo | c clear | q quit",
        ]
        y = 50
        for line in hud_lines:
            cv2.putText(
                overlay,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )
            y += 20
        cv2.imshow(self.window_name, overlay)

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if self.drawing_mode == "circle":
            self._circle_mouse(event, x, y)
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = self._map_display_to_image(x, y)
            self.current_points.append((float(px), float(py)))

    def _circle_mouse(self, event, x, y) -> None:
        px, py = self._map_display_to_image(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.circle_center = (float(px), float(py))
            self.circle_radius = 0.0
            self.is_drawing_circle = True
        elif event == cv2.EVENT_MOUSEMOVE and self.is_drawing_circle and self.circle_center is not None:
            dx = px - self.circle_center[0]
            dy = py - self.circle_center[1]
            self.circle_radius = float(np.hypot(dx, dy))
        elif event == cv2.EVENT_LBUTTONUP and self.is_drawing_circle and self.circle_center is not None:
            dx = px - self.circle_center[0]
            dy = py - self.circle_center[1]
            self.circle_radius = float(np.hypot(dx, dy))
            self.is_drawing_circle = False
            self._finalize_circle()

    def _finalize_polygon(self) -> None:
        if len(self.current_points) < 3:
            self.status_message = "Need at least three points"
            return
        region = AnnotationRegion(
            feature_type=self.current_feature,
            frame_idx=self.current_frame_idx,
            points=self.current_points.copy(),
            shape="polygon",
        )
        self.annotations.append(region)
        self.current_points.clear()
        self.status_message = f"Added {region.feature_type} annotation"

    def _finalize_circle(self) -> None:
        if self.circle_center is None or self.circle_radius is None or self.circle_radius < 1.0:
            self.status_message = "Circle radius too small"
            self.circle_center = None
            self.circle_radius = None
            return
        region = AnnotationRegion(
            feature_type=self.current_feature,
            frame_idx=self.current_frame_idx,
            points=[],
            shape="circle",
            circle_center=self.circle_center,
            circle_radius=self.circle_radius,
        )
        self.annotations.append(region)
        self.circle_center = None
        self.circle_radius = None
        self.status_message = "Circle annotation added"

    def _remove_last_annotation(self) -> None:
        for idx in range(len(self.annotations) - 1, -1, -1):
            if self.annotations[idx].feature_type == self.current_feature and self.annotations[idx].frame_idx == self.current_frame_idx:
                self.annotations.pop(idx)
                self.status_message = "Removed last annotation"
                return
        self.status_message = "No annotation to remove"

    def run(self) -> List[AnnotationRegion]:
        if not self._read_frame(0):
            raise RuntimeError("Unable to read first frame")

        while True:
            self._render()
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                break
            elif key == ord("d"):
                self.show_diff = not self.show_diff
            elif key == ord("m"):
                self.current_feature = "mouth"
                self.status_message = "Feature: mouth"
            elif key == ord("g"):
                self.current_feature = "gonad"
                self.status_message = "Feature: gonad"
            elif key == ord("t"):
                self.current_feature = "tentacle_bulb"
                self.status_message = "Feature: tentacle bulb"
            elif key == ord("o"):
                if self.drawing_mode == "polygon":
                    self.drawing_mode = "circle"
                    self.current_points.clear()
                    self.status_message = "Circle mode: click and drag to define radius"
                else:
                    self.drawing_mode = "polygon"
                    self.circle_center = None
                    self.circle_radius = None
                    self.is_drawing_circle = False
                    self.status_message = "Polygon mode: click to add vertices"
            elif key == ord("b"):
                self._cycle_threshold()
            elif key in (ord("+"), ord("=")):
                self.zoom = min(self.zoom + 0.25, 5.0)
            elif key == ord("-"):
                self.zoom = max(1.0, self.zoom - 0.25)
            elif key == ord("h"):
                self.pan_x = max(self.pan_x - 40, 0)
            elif key == ord("l"):
                self.pan_x += 40
            elif key == ord("k"):
                self.pan_y = max(self.pan_y - 40, 0)
            elif key == ord("j"):
                self.pan_y += 40
            elif key in (ord("n"), ord("]")):
                self._read_frame(min(self.current_frame_idx + 1, self.total_frames - 1))
            elif key in (ord("p"), ord("[")):
                self._read_frame(max(self.current_frame_idx - 1, 0))
            elif key == 13:  # Enter
                if self.drawing_mode == "polygon":
                    self._finalize_polygon()
            elif key in (ord("u"), 8):
                if self.drawing_mode == "polygon":
                    if self.current_points:
                        self.current_points.pop()
                    else:
                        self._remove_last_annotation()
                else:
                    if self.is_drawing_circle:
                        self.circle_center = None
                        self.circle_radius = None
                        self.is_drawing_circle = False
                    else:
                        self._remove_last_annotation()
            elif key == ord("c"):
                if self.drawing_mode == "polygon":
                    self.current_points.clear()
                else:
                    self.circle_center = None
                    self.circle_radius = None
                    self.is_drawing_circle = False

        cv2.destroyWindow(self.window_name)
        self.cap.release()
        return self.annotations

    def _cycle_threshold(self) -> None:
        """Cycle through predefined threshold options."""

        if not self.threshold_options:
            return
        self.threshold_index = (self.threshold_index + 1) % len(self.threshold_options)
        self.params.threshold = int(self.threshold_options[self.threshold_index])
        self.status_message = f"Threshold set to {self.params.threshold}px"


def _annotations_to_samples(
    annotations: List[AnnotationRegion],
    frame_shape: Tuple[int, int],
    pixel_size_mm: float,
) -> List[SampledFeature]:
    samples: List[SampledFeature] = []
    height, width = frame_shape
    for region in annotations:
        mask = np.zeros((height, width), dtype=np.uint8)
        if region.shape == "circle" and region.circle_center and region.circle_radius:
            center = (int(round(region.circle_center[0])), int(round(region.circle_center[1])))
            radius = max(1, int(round(region.circle_radius)))
            cv2.circle(mask, center, radius, 255, -1)
        else:
            pts = np.array(region.points, dtype=np.int32)
            if pts.size == 0:
                continue
            cv2.fillPoly(mask, [pts], 255)
        props = regionprops(mask)
        if not props:
            continue
        prop = props[0]
        aspect_ratio = prop.axis_major_length / prop.axis_minor_length if prop.axis_minor_length > 0 else 1.0
        samples.append(
            SampledFeature(
                object_type=region.feature_type,
                centroid=(prop.centroid[1], prop.centroid[0]),
                area=float(prop.area),
                aspect_ratio=float(aspect_ratio),
                eccentricity=float(prop.eccentricity),
                solidity=float(prop.solidity),
                major_axis_length_mm=float(prop.axis_major_length * pixel_size_mm),
                minor_axis_length_mm=float(prop.axis_minor_length * pixel_size_mm),
            )
        )
    return samples


def _count_samples(samples: List[SampledFeature]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in samples:
        counts[sample.object_type] = counts.get(sample.object_type, 0) + 1
    return counts


def _validate_samples(samples: List[SampledFeature], params: TrackingParameters) -> None:
    counts = _count_samples(samples)
    mouth_count = counts.get("mouth", 0)
    if mouth_count < MIN_MOUTH_SAMPLES:
        raise click.ClickException("Please annotate at least one mouth region before continuing.")
    warnings = []
    if params.num_gonads and counts.get("gonad", 0) == 0:
        warnings.append("No gonad annotations detected; using default gonad parameters.")
    if params.num_tentacle_bulbs and counts.get("tentacle_bulb", 0) == 0:
        warnings.append("No tentacle bulb annotations detected; using default bulb parameters.")
    for msg in warnings:
        click.echo(f"Warning: {msg}")


def _blend_value(default: float | int | None, derived: float, sample_count: int) -> float:
    if default is None or not isinstance(default, (int, float)):
        return derived
    weight = min(1.0, sample_count / FULL_WEIGHT_SAMPLES)
    return default * (1.0 - weight) + derived * weight


def _print_sample_summary(samples: List[SampledFeature]) -> None:
    if not samples:
        return
    click.echo("\nFeature annotation summary:")
    counts = _count_samples(samples)
    for feature_type, count in counts.items():
        areas = [s.area for s in samples if s.object_type == feature_type]
        if not areas:
            continue
        click.echo(
            f"  {feature_type}: {count} sample(s), area median {np.median(areas):.1f}px, min {min(areas):.1f}px, max {max(areas):.1f}px"
        )


def _collect_annotation_points(annotations: List[AnnotationRegion]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for region in annotations:
        if region.shape == "circle" and region.circle_center and region.circle_radius:
            cx, cy = region.circle_center
            r = region.circle_radius
            points.extend(
                [
                    (cx, cy),
                    (cx + r, cy),
                    (cx - r, cy),
                    (cx, cy + r),
                    (cx, cy - r),
                ]
            )
        else:
            points.extend(region.points)
    return points


def _suggest_roi_from_annotations(
    annotations: List[AnnotationRegion],
    width: int,
    height: int,
) -> Dict[str, object] | None:
    points = _collect_annotation_points(annotations)
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = max(0.0, min(xs)), min(float(width), max(xs))
    min_y, max_y = max(0.0, min(ys)), min(float(height), max(ys))
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span_x = max_x - min_x
    span_y = max_y - min_y
    radius = 0.6 * max(span_x, span_y)
    radius = max(radius, 10.0)
    return {
        "mode": "circle",
        "center": (center_x, center_y),
        "radius": radius,
        "bbox": (min_x, min_y, max_x - min_x, max_y - min_y),
    }


def _suggest_parameters(
    samples: List[SampledFeature],
    params: TrackingParameters,
) -> TrackingParameters:
    if not samples:
        return params

    grouped: Dict[str, List[SampledFeature]] = {}
    for sample in samples:
        grouped.setdefault(sample.object_type, []).append(sample)

    for obj_type, feats in grouped.items():
        config = params.get_object_config(obj_type)
        if config is None or not feats:
            continue
        original_config = config.copy()
        sample_count = len(feats)
        areas = np.array([f.area for f in feats], dtype=np.float32)
        aspects = np.array([f.aspect_ratio for f in feats], dtype=np.float32)
        eccs = np.array([f.eccentricity for f in feats], dtype=np.float32)
        solids = np.array([f.solidity for f in feats], dtype=np.float32)

        def bounds(values: np.ndarray, min_width: float, pad: float = 0.25) -> Tuple[float, float]:
            if values.size == 0:
                return 0.0, 0.0
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            spread = max(mad * 1.4826, min_width)
            return median - max(spread, median * pad), median + max(spread, median * pad)

        area_min, area_max = bounds(areas, min_width=10.0)
        aspect_min, aspect_max = bounds(aspects, min_width=0.1)
        ecc_min, ecc_max = bounds(eccs, min_width=0.05)
        sol_min, sol_max = bounds(solids, min_width=0.05)

        derived_values = {
            "min_area": max(5, int(area_min)),
            "max_area": max(int(area_max), int(area_min) + 5),
            "aspect_ratio_min": max(0.5, aspect_min),
            "aspect_ratio_max": max(aspect_max, aspect_min + 0.1),
            "eccentricity_min": max(0.0, ecc_min),
            "eccentricity_max": min(1.0, max(ecc_max, ecc_min + 0.05)),
            "solidity_min": max(0.0, sol_min),
            "solidity_max": min(1.0, max(sol_max, sol_min + 0.05)),
        }

        if obj_type == "gonad":
            derived_values["aspect_ratio_min"] = max(1.2, derived_values["aspect_ratio_min"])
        elif obj_type == "tentacle_bulb":
            derived_values["aspect_ratio_max"] = min(1.8, derived_values["aspect_ratio_max"])

        for key, value in derived_values.items():
            blended = _blend_value(original_config.get(key), value, sample_count)
            if key in ("min_area", "max_area"):
                config[key] = int(blended)
            else:
                config[key] = float(blended)

    return params


def _apply_search_radius_from_samples(
    samples: List[SampledFeature],
    params: TrackingParameters,
) -> None:
    mouth_centroids = np.array([f.centroid for f in samples if f.object_type == "mouth"], dtype=np.float32)
    if mouth_centroids.size == 0:
        return
    mouth_center = np.mean(mouth_centroids, axis=0)

    for obj_type in ("gonad", "tentacle_bulb"):
        feats = [f for f in samples if f.object_type == obj_type]
        if not feats:
            continue
        distances = [float(np.linalg.norm(np.array(f.centroid) - mouth_center)) for f in feats]
        if not distances:
            continue
        radius = max(distances) * 1.3
        config = params.get_object_config(obj_type)
        if config is not None:
            config["search_radius"] = radius


def run_interactive_feature_sampling(
    video_path: str,
    params: TrackingParameters | None = None,
    pixel_size_mm: float = 0.01,
    apply_roi_mask: bool = True,
) -> TrackingParameters:
    params = params or TrackingParameters()
    app = FeatureAnnotationApp(video_path, params, pixel_size_mm)
    annotations = app.run()
    if not annotations:
        return params

    samples = _annotations_to_samples(annotations, (app.height, app.width), pixel_size_mm)
    _validate_samples(samples, params)
    params = _suggest_parameters(samples, params)
    _apply_search_radius_from_samples(samples, params)
    _print_sample_summary(samples)

    roi_suggestion = _suggest_roi_from_annotations(annotations, app.width, app.height)
    if roi_suggestion:
        center = roi_suggestion.get("center")
        radius = roi_suggestion.get("radius")
        bbox = roi_suggestion.get("bbox")
        click.echo(
            "\nSuggested ROI from annotations:"
        )
        if center and isinstance(center, (tuple, list)) and radius:
            cx, cy = center
            click.echo(
                f"  Circle center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}px"
            )
        if bbox and isinstance(bbox, (tuple, list)) and len(bbox) == 4:
            bx, by, bw, bh = bbox
            click.echo(
                f"  Bounding box x={bx:.1f}, y={by:.1f}, w={bw:.1f}, h={bh:.1f}"
            )
        if click.confirm("Apply this ROI to tracking parameters?", default=True):
            params.apply_roi_config(roi_suggestion)
        else:
            click.echo("Keeping existing ROI configuration.")

    params.update_object_counts()
    return params
