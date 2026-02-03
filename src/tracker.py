import json
import numpy as np
import json
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field, asdict


@dataclass
class TrackingParameters:
    """
    Parameters used for tracking, with serialization support.

    This dataclass holds all configurable tracking parameters, allowing them
    to be saved with tracking results and reused for visualization.
    
    Supports multi-object detection with configurable mouth, gonads, and bulbs.
    """

    # Background subtraction (frames sampled per episode)
    background_buffer_size: int = 10
    threshold: int = 10
    use_auto_threshold: bool = True

    # Legacy mouth detection parameters (for backward compatibility)
    mouth_min_area: int = 35
    mouth_max_area: int = 160
    mouth_max_disappeared: int = 15
    mouth_max_distance: int = 50
    mouth_search_radius: int | None = None

    # Legacy bulb detection parameters (for backward compatibility)
    bulb_min_area: int = 5
    bulb_max_area: int = 35
    bulb_max_disappeared: int = 10
    bulb_max_distance: int = 30
    bulb_search_radius: int | None = None

    # Object permanence controls
    mouth_exclusion_radius: int = 30
    temporal_smoothing_window: int = 3
    object_priority: list[str] = field(
        default_factory=lambda: ["mouth", "gonad", "tentacle_bulb"]
    )

    # Multi-object configuration (new extensible system)
    # Object counts (user-configurable)
    num_mouths: int = 1
    num_gonads: int = 0  # 0-4 gonads per animal
    num_tentacle_bulbs: int | None = None  # None = auto-detect unlimited
    mouth_pinned: bool = False
    pinned_mouth_point: tuple[float, float] | None = None

    # ROI configuration (interactive selection / bounding box)
    roi_mode: str = "auto"  # "auto", "circle", "polygon", "bounding_box"
    roi_center: tuple[float, float] | None = None  # For circle ROI
    roi_radius: float | None = None  # For circle ROI
    roi_points: list[tuple[float, float]] = field(default_factory=list)  # For polygon ROI
    roi_bbox: tuple[float, float, float, float] | None = None  # For bounding box ROI (x, y, w, h)

    # Object type configurations (extensible dictionary)
    object_types: dict[str, dict] = field(default_factory=lambda: {
        "mouth": {
            "enabled": True,
            "count": 1,  # Expected number
            "min_area": 35,
            "max_area": 160,
             "max_disappeared": 15,
             "max_distance": 50,
             "search_radius": None,
             "track_search_radius": None,
             "reference_object": None,
             "min_distance_from_reference": None,
             "max_jump_px": 60,
             "ownership_radius": 50,
             "overlap_grace_frames": 2,
             "overlap_penalty_weight": 0.5,
             "score_margin": 0.1,
             "exclude_objects": {},
              # Shape parameters (None = no constraint)
              "aspect_ratio_min": None,
            "aspect_ratio_max": None,
            "eccentricity_min": None,
            "eccentricity_max": None,
            "solidity_min": None,
            "solidity_max": None,
        },
        "gonad": {
            "enabled": False,  # Controlled by num_gonads > 0
            "count": 4,  # Max number to detect
            "min_area": 20,
            "max_area": 80,
             "max_disappeared": 15,
             "max_distance": 40,
             "search_radius": 75,
             "track_search_radius": 60,
             "reference_object": "mouth",
             "min_distance_from_reference": 25,
             "max_jump_px": 45,
             "ownership_radius": 40,
             "overlap_grace_frames": 2,
             "overlap_penalty_weight": 0.5,
             "score_margin": 0.15,
             "exclude_objects": {"tentacle_bulb": 20},
              # Gonads are oblong - use aspect ratio filtering
            "aspect_ratio_min": 1.5,  # Oblong shape
            "aspect_ratio_max": 3.0,
            "eccentricity_min": 0.7,  # Elongated
            "eccentricity_max": None,
            "solidity_min": None,
            "solidity_max": None,
        },
        "tentacle_bulb": {
            "enabled": False,  # Controlled by num_tentacle_bulbs > 0
            "count": None,  # User-specified
            "min_area": 5,
            "max_area": 35,
             "max_disappeared": 10,
             "max_distance": 30,
             "search_radius": 60,
             "track_search_radius": 45,
             "reference_object": "mouth",
             "min_distance_from_reference": 10,
             "max_jump_px": 35,
             "ownership_radius": 35,
             "overlap_grace_frames": 2,
             "overlap_penalty_weight": 0.5,
             "score_margin": 0.15,
             "exclude_objects": {"gonad": 20},
            # Tentacle bulbs are typically round
            "aspect_ratio_min": None,
            "aspect_ratio_max": 1.5,
            "eccentricity_min": None,
            "eccentricity_max": 0.7,
            "solidity_min": 0.8,
            "solidity_max": None,
        }
    })

    # Video mode and adaptive background settings
    video_type: str = "non_rotating"
    # Adaptive background (unchanged)
    adaptive_background: bool = False
    rotation_start_threshold_deg: float = 0.01
    rotation_stop_threshold_deg: float = 0.005
    rotation_start_frames: int = 3
    rotation_confidence_threshold: float = 0.3
    min_episode_rotation_deg: float = 5.0
    rotation_center: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        """Convert parameters to a JSON-serializable dictionary."""
        d = asdict(self)
        # rotation_center is already a tuple or None, which is JSON serializable
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrackingParameters":
        """Create TrackingParameters from a dictionary."""
        # Handle rotation_center: convert list back to tuple if present
        if "rotation_center" in d and d["rotation_center"] is not None:
            d = d.copy()
            d["rotation_center"] = tuple(d["rotation_center"])
        if "pinned_mouth_point" in d and d["pinned_mouth_point"] is not None:
            d = d.copy()
            d["pinned_mouth_point"] = tuple(d["pinned_mouth_point"])
        return cls(**d)

    def to_json(self) -> str:
        """Serialize parameters to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "TrackingParameters":
        """Deserialize parameters from JSON string."""
        return cls.from_dict(json.loads(json.loads(json_str)))

    def update_object_counts(self):
        """Update object type enabled status based on count parameters."""
        # Ensure mouth configuration reflects requested count
        if "mouth" in self.object_types:
            mouth_count = max(self.num_mouths, 0)
            mouth_enabled = mouth_count > 0
            self.object_types["mouth"]["enabled"] = mouth_enabled
            self.object_types["mouth"]["count"] = mouth_count if mouth_enabled else None

        # Enable gonads if count > 0
        if "gonad" in self.object_types:
            self.object_types["gonad"]["enabled"] = self.num_gonads > 0
            self.object_types["gonad"]["count"] = self.num_gonads

        # Enable tentacle bulbs if count > 0
        if "tentacle_bulb" in self.object_types:
            if self.num_tentacle_bulbs is None:
                # Auto-detect mode: enabled with no hard cap
                self.object_types["tentacle_bulb"]["enabled"] = True
                self.object_types["tentacle_bulb"]["count"] = None
            else:
                bulb_count = max(self.num_tentacle_bulbs, 0)
                self.object_types["tentacle_bulb"]["enabled"] = bulb_count > 0
                self.object_types["tentacle_bulb"]["count"] = bulb_count if bulb_count > 0 else None

    def get_enabled_object_types(self) -> list[str]:
        """Get list of enabled object types for tracking."""
        self.update_object_counts()
        return [obj_type for obj_type, config in self.object_types.items() 
                if config.get("enabled", False)]

    def get_object_config(self, obj_type: str) -> dict | None:
        """Get configuration for a specific object type."""
        return self.object_types.get(obj_type)

    def set_object_config(self, obj_type: str, config: dict):
        """Set configuration for a specific object type."""
        if obj_type not in self.object_types:
            self.object_types[obj_type] = {}
        self.object_types[obj_type].update(config)

    def update_from_legacy_params(self):
        """Update object type configs from legacy parameters for backward compatibility."""
        # Update mouth config from legacy parameters
        if "mouth" in self.object_types:
            mouth_config = self.object_types["mouth"]
            mouth_config.update({
                "min_area": self.mouth_min_area,
                "max_area": self.mouth_max_area,
                "max_disappeared": self.mouth_max_disappeared,
                "max_distance": self.mouth_max_distance,
                "search_radius": self.mouth_search_radius,
            })
        
        # Update bulb config from legacy parameters
        if "tentacle_bulb" in self.object_types:
            bulb_config = self.object_types["tentacle_bulb"]
            bulb_config.update({
                "min_area": self.bulb_min_area,
                "max_area": self.bulb_max_area,
                "max_disappeared": self.bulb_max_disappeared,
                "max_distance": self.bulb_max_distance,
                "search_radius": self.bulb_search_radius,
            })

    def get_roi_params(self) -> dict:
        """Get ROI configuration parameters."""
        return {
            "mode": self.roi_mode,
            "center": self.roi_center,
            "radius": self.roi_radius,
            "points": self.roi_points,
            "bbox": self.roi_bbox,
        }

    def clear_roi(self) -> None:
        """Reset ROI configuration to automatic mode."""
        self.roi_mode = "auto"
        self.roi_center = None
        self.roi_radius = None
        self.roi_points = []
        self.roi_bbox = None

    def apply_roi_config(self, roi_config: dict | None) -> None:
        """Apply an ROI configuration dictionary returned from selectors."""

        if not roi_config:
            self.clear_roi()
            return

        mode = (roi_config.get("mode") or "auto").lower()
        self.roi_mode = mode

        if mode == "circle":
            self.roi_center = tuple(roi_config.get("center", ())) or None
            self.roi_radius = roi_config.get("radius")
            self.roi_points = []
            self.roi_bbox = None
        elif mode == "polygon":
            self.roi_points = list(roi_config.get("points", []))
            self.roi_center = None
            self.roi_radius = None
            self.roi_bbox = None
        elif mode == "bounding_box":
            self.roi_bbox = roi_config.get("bbox")
            self.roi_center = None
            self.roi_radius = None
            self.roi_points = []
        else:
            self.clear_roi()

    def has_custom_roi(self) -> bool:
        """Return True when ROI mode is something other than auto."""
        return self.roi_mode in {"circle", "polygon", "bounding_box"}


@dataclass
class TrackingData:
    """
    Data-oriented storage for tracking results.
    All arrays have shape (n_tracks, n_frames) with NaN padding for missing data.
    """

    n_tracks: int
    n_frames: int

    # Track metadata
    track_ids: np.ndarray  # (n_tracks,) - unique ID for each track

    # Per-frame detection data: (n_tracks, n_frames)
    x: np.ndarray
    y: np.ndarray
    area: np.ndarray
    major_axis_length_mm: np.ndarray
    bbox_min_row: np.ndarray
    bbox_min_col: np.ndarray
    bbox_max_row: np.ndarray
    bbox_max_col: np.ndarray

    # Frame indices for each detection (to know which frame each column corresponds to)
    frame: np.ndarray  # (n_tracks, n_frames) - frame number for each detection

    @classmethod
    def from_history(cls, history: dict, n_frames: int) -> "TrackingData":
        """
        Convert tracker history dict to data-oriented arrays.

        Args:
            history: {objectID: [detection_or_None for each frame from 1 to n_frames]}
            n_frames: Total number of frames processed
        """
        track_ids = np.array(sorted(history.keys()), dtype=np.int32)
        n_tracks = len(track_ids)

        # Initialize arrays with NaN
        x = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        y = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        area = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        major_axis_length_mm = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_min_row = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_min_col = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_max_row = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_max_col = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        frame = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)

        # Fill in data from history
        for track_idx, obj_id in enumerate(track_ids):
            track_history = history[obj_id]
            for frame_idx, detection in enumerate(track_history):
                if detection is not None:
                    x[track_idx, frame_idx] = detection["centroid"][0]
                    y[track_idx, frame_idx] = detection["centroid"][1]
                    area[track_idx, frame_idx] = detection["area"]
                    major_axis_length_mm[track_idx, frame_idx] = detection[
                        "major_axis_length_mm"
                    ]
                    bbox_min_row[track_idx, frame_idx] = detection["bounding_box"][0]
                    bbox_min_col[track_idx, frame_idx] = detection["bounding_box"][1]
                    bbox_max_row[track_idx, frame_idx] = detection["bounding_box"][2]
                    bbox_max_col[track_idx, frame_idx] = detection["bounding_box"][3]
                    frame[track_idx, frame_idx] = (
                        frame_idx + 1
                    )  # 1-indexed frame number

        return cls(
            n_tracks=n_tracks,
            n_frames=n_frames,
            track_ids=track_ids,
            x=x,
            y=y,
            area=area,
            major_axis_length_mm=major_axis_length_mm,
            bbox_min_row=bbox_min_row,
            bbox_min_col=bbox_min_col,
            bbox_max_row=bbox_max_row,
            bbox_max_col=bbox_max_col,
            frame=frame,
        )


class RobustTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.nextObjectID = 0
        self.objects = {}  # Stores current detection: {ID: detection_dict}
        self.disappeared = {}  # Stores lost frame counts: {ID: count}
        self.history = {}  # Stores full path: {ID: [detection_or_None, ...]}

        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.current_frame = 0

    def register(self, detection):
        """Register a new object with the next available ID."""
        self.objects[self.nextObjectID] = detection
        self.disappeared[self.nextObjectID] = 0
        # Prefill history with None for all frames before this detection
        self.history[self.nextObjectID] = [None] * (self.current_frame - 1) + [
            detection
        ]
        self.nextObjectID += 1

    def deregister(self, objectID):
        """Remove an object from active tracking (history is preserved)."""
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections: list of detection dictionaries with 'centroid' key
        """
        self.current_frame += 1

        # If no tracks exist, register all detections as new objects
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
            return self.objects

        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                self.history[objectID].append(None)
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        # --- Core Matching Logic ---
        objectIDs = list(self.objects.keys())
        objectCentroids = [obj["centroid"] for obj in self.objects.values()]
        detectionCentroids = [d["centroid"] for d in detections]

        # Calculate Distance Matrix
        D = np.zeros((len(objectCentroids), len(detectionCentroids)))
        for t, track_cnt in enumerate(objectCentroids):
            for d, det_cnt in enumerate(detectionCentroids):
                D[t, d] = np.linalg.norm(np.array(track_cnt) - np.array(det_cnt))

        # Hungarian Algorithm
        rows, cols = linear_sum_assignment(D)

        usedRows = set(rows)
        usedCols = set(cols)

        # Update assigned tracks
        for row, col in zip(rows, cols):
            if D[row, col] > self.max_distance:
                usedRows.remove(row)
                usedCols.remove(col)
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = detections[col]
            self.history[objectID].append(detections[col])
            self.disappeared[objectID] = 0

        # Handle lost tracks
        unusedRows = set(range(D.shape[0])).difference(usedRows)
        for row in unusedRows:
            objectID = objectIDs[row]
            self.disappeared[objectID] += 1
            self.history[objectID].append(None)
            if self.disappeared[objectID] > self.max_disappeared:
                self.deregister(objectID)

        # Handle new detections
        unusedCols = set(range(D.shape[1])).difference(usedCols)
        for col in unusedCols:
            self.register(detections[col])

        return self.objects

    def get_tracking_data(self) -> TrackingData:
        """Convert tracker history to data-oriented TrackingData object."""
        return TrackingData.from_history(self.history, self.current_frame)
