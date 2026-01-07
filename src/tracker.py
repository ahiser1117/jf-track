import numpy as np
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass


SKELETON_POINTS = 5


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
    eccentricity: np.ndarray
    major_axis_length_mm: np.ndarray
    bbox_min_row: np.ndarray
    bbox_min_col: np.ndarray
    bbox_max_row: np.ndarray
    bbox_max_col: np.ndarray
    skeleton_x: np.ndarray  # (n_tracks, n_frames, n_points)
    skeleton_y: np.ndarray  # (n_tracks, n_frames, n_points)
    
    # Frame indices for each detection (to know which frame each column corresponds to)
    frame: np.ndarray  # (n_tracks, n_frames) - frame number for each detection
    
    @classmethod
    def from_history(cls, history: dict, n_frames: int) -> 'TrackingData':
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
        eccentricity = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        major_axis_length_mm = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_min_row = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_min_col = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_max_row = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        bbox_max_col = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        skeleton_x = np.full((n_tracks, n_frames, SKELETON_POINTS), np.nan, dtype=np.float64)
        skeleton_y = np.full((n_tracks, n_frames, SKELETON_POINTS), np.nan, dtype=np.float64)
        frame = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        
        # Fill in data from history
        for track_idx, obj_id in enumerate(track_ids):
            track_history = history[obj_id]
            for frame_idx, detection in enumerate(track_history):
                if detection is not None:
                    x[track_idx, frame_idx] = detection['centroid'][0]
                    y[track_idx, frame_idx] = detection['centroid'][1]
                    area[track_idx, frame_idx] = detection['area']
                    eccentricity[track_idx, frame_idx] = detection['eccentricity']
                    major_axis_length_mm[track_idx, frame_idx] = detection['major_axis_length_mm']
                    bbox_min_row[track_idx, frame_idx] = detection['bounding_box'][0]
                    bbox_min_col[track_idx, frame_idx] = detection['bounding_box'][1]
                    bbox_max_row[track_idx, frame_idx] = detection['bounding_box'][2]
                    bbox_max_col[track_idx, frame_idx] = detection['bounding_box'][3]
                    if 'skeleton' in detection and detection['skeleton'] is not None:
                        skel = detection['skeleton']
                        # Ensure we have exactly SKELETON_POINTS samples
                        for p_idx in range(min(len(skel), SKELETON_POINTS)):
                            skeleton_x[track_idx, frame_idx, p_idx] = skel[p_idx][0]
                            skeleton_y[track_idx, frame_idx, p_idx] = skel[p_idx][1]
                    frame[track_idx, frame_idx] = frame_idx + 1  # 1-indexed frame number
        
        return cls(
            n_tracks=n_tracks,
            n_frames=n_frames,
            track_ids=track_ids,
            x=x,
            y=y,
            area=area,
            eccentricity=eccentricity,
            major_axis_length_mm=major_axis_length_mm,
            bbox_min_row=bbox_min_row,
            bbox_min_col=bbox_min_col,
            bbox_max_row=bbox_max_row,
            bbox_max_col=bbox_max_col,
            skeleton_x=skeleton_x,
            skeleton_y=skeleton_y,
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
        self.history[self.nextObjectID] = [None] * (self.current_frame - 1) + [detection]
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
        objectCentroids = [obj['centroid'] for obj in self.objects.values()]
        detectionCentroids = [d['centroid'] for d in detections]

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
