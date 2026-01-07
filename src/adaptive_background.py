"""
Adaptive background subtraction with rotation compensation.

This module provides classes for handling videos where the background rotates
around a fixed center point. It automatically detects rotation episodes,
estimates rotation parameters, and provides frame-by-frame compensated
backgrounds for accurate segmentation.
"""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from collections import deque


class RotationState(Enum):
    """State machine states for background management."""
    STATIC = "static"           # No rotation, use static background
    ROTATING = "rotating"       # Active rotation, compensate background
    TRANSITION = "transition"   # Post-rotation, collecting frames for new background


@dataclass
class RotationEstimate:
    """Result from frame-to-frame rotation estimation."""
    angle_deg: float              # Rotation angle in degrees (positive = clockwise)
    center: tuple[float, float]   # (cx, cy) estimated rotation center
    confidence: float             # 0-1, quality of estimate based on inlier ratio
    n_inliers: int                # Number of matched feature inliers


@dataclass
class RotationEpisode:
    """Records a single rotation episode."""
    start_frame: int
    end_frame: int | None = None  # None if ongoing
    rotation_center: tuple[float, float] = (0.0, 0.0)
    angular_velocity_deg: float = 0.0  # Average degrees per frame
    total_rotation_deg: float = 0.0    # Cumulative rotation


class RotationEstimator:
    """
    Estimates rotation between frames using ORB feature matching.

    Uses cv2.estimateAffinePartial2D to compute rotation + translation,
    then extracts rotation angle and estimates rotation center.
    """

    def __init__(
        self,
        n_features: int = 500,
        match_ratio_threshold: float = 0.75,
        ransac_reproj_threshold: float = 3.0,
        min_inliers: int = 20,
    ):
        """
        Args:
            n_features: Number of ORB features to detect per frame
            match_ratio_threshold: Lowe's ratio test threshold
            ransac_reproj_threshold: RANSAC reprojection error threshold
            min_inliers: Minimum inliers for valid estimate
        """
        self.n_features = n_features
        self.match_ratio_threshold = match_ratio_threshold
        self.ransac_reproj_threshold = ransac_reproj_threshold
        self.min_inliers = min_inliers

        # Create ORB detector
        self.orb = cv2.ORB_create(nfeatures=n_features)
        # Create brute-force matcher with Hamming distance for ORB
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def estimate(
        self,
        frame_prev: np.ndarray,
        frame_curr: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> RotationEstimate | None:
        """
        Estimate rotation from frame_prev to frame_curr.

        Args:
            frame_prev: Previous frame (grayscale uint8)
            frame_curr: Current frame (grayscale uint8)
            mask: Optional mask to exclude foreground objects (255 = include)

        Returns:
            RotationEstimate if successful, None if insufficient features
        """
        # Detect keypoints and compute descriptors
        kp1, desc1 = self.orb.detectAndCompute(frame_prev, mask)
        kp2, desc2 = self.orb.detectAndCompute(frame_curr, mask)

        if desc1 is None or desc2 is None or len(kp1) < 10 or len(kp2) < 10:
            return None

        # Match features using kNN
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.match_ratio_threshold * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.min_inliers:
            return None

        # Extract matched point coordinates
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        # Estimate affine partial (rotation + translation + uniform scale)
        # This constrains the transform to rotation, translation, and uniform scaling
        M, inliers = cv2.estimateAffinePartial2D(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_reproj_threshold,
        )

        if M is None or inliers is None:
            return None

        n_inliers = int(np.sum(inliers))
        if n_inliers < self.min_inliers:
            return None

        # Extract rotation angle from transform matrix
        # M = [[cos(t)*s, -sin(t)*s, tx],
        #      [sin(t)*s,  cos(t)*s, ty]]
        angle_rad = np.arctan2(M[1, 0], M[0, 0])
        angle_deg = np.degrees(angle_rad)

        # Extract rotation center
        center = self._extract_rotation_center(M, frame_prev.shape[:2])

        # Compute confidence based on inlier ratio
        confidence = n_inliers / len(good_matches)

        return RotationEstimate(
            angle_deg=angle_deg,
            center=center,
            confidence=confidence,
            n_inliers=n_inliers,
        )

    def _extract_rotation_center(
        self,
        M: np.ndarray,
        frame_shape: tuple[int, int],
    ) -> tuple[float, float]:
        """
        Extract rotation center from 2x3 affine partial transform.

        For rotation around point (cx, cy):
            p' = R * (p - c) + c = R*p + (I - R)*c

        The affine transform gives us: p' = R*p + t
        Therefore: t = (I - R)*c, so c = (I - R)^(-1) * t

        If matrix is singular (near-zero rotation), returns frame center.
        """
        # Extract rotation matrix and translation
        R = M[:2, :2]
        t = M[:, 2]

        # Compute (I - R)
        I_minus_R = np.eye(2) - R

        # Check if invertible (rotation angle not ~0)
        det = np.linalg.det(I_minus_R)
        if abs(det) < 1e-6:
            # Near-zero rotation, center undefined - return frame center
            h, w = frame_shape
            return (w / 2.0, h / 2.0)

        # Solve for center
        try:
            center = np.linalg.solve(I_minus_R, t)
            return (float(center[0]), float(center[1]))
        except np.linalg.LinAlgError:
            h, w = frame_shape
            return (w / 2.0, h / 2.0)


class AdaptiveBackgroundManager:
    """
    Manages adaptive background for rotation-compensated tracking.

    State Machine:
    - STATIC: Uses precomputed static background
    - ROTATING: Rotates reference background to match current orientation
    - TRANSITION: Collecting frames for new static background

    Usage:
        manager = AdaptiveBackgroundManager(video_path)
        manager.initialize()

        for frame_idx, frame in enumerate(video_frames):
            background = manager.get_background(frame_idx, frame)
            # Use background for subtraction...
    """

    def __init__(
        self,
        video_path: str,
        # Background computation
        initial_bg_samples: int = 10,
        transition_bg_frames: int = 30,
        # Rotation detection thresholds
        rotation_start_threshold_deg: float = 0.5,
        rotation_stop_threshold_deg: float = 0.1,
        rotation_stop_frames: int = 10,
        # Rotation estimation
        n_features: int = 500,
        min_inliers: int = 20,
        # Smoothing
        angle_smoothing_window: int = 5,
    ):
        """
        Args:
            video_path: Path to video file
            initial_bg_samples: Frames to sample for initial background
            transition_bg_frames: Frames to collect after rotation stops
            rotation_start_threshold_deg: Min angle/frame to trigger rotation
            rotation_stop_threshold_deg: Max angle/frame to consider stopped
            rotation_stop_frames: Consecutive low-rotation frames to exit
            n_features: ORB features for rotation estimation
            min_inliers: Minimum feature matches for valid rotation
            angle_smoothing_window: Frames to smooth rotation estimates
        """
        self.video_path = video_path
        self.initial_bg_samples = initial_bg_samples
        self.transition_bg_frames = transition_bg_frames
        self.rotation_start_threshold_deg = rotation_start_threshold_deg
        self.rotation_stop_threshold_deg = rotation_stop_threshold_deg
        self.rotation_stop_frames = rotation_stop_frames
        self.angle_smoothing_window = angle_smoothing_window

        # Create rotation estimator
        self.estimator = RotationEstimator(
            n_features=n_features,
            min_inliers=min_inliers,
        )

        # State
        self._state = RotationState.STATIC
        self._static_background: np.ndarray | None = None
        self._reference_background: np.ndarray | None = None  # For rotation
        self._rotation_center: tuple[float, float] | None = None
        self._cumulative_rotation_deg: float = 0.0
        self._prev_frame: np.ndarray | None = None

        # Rotation detection history
        self._angle_history: deque[float] = deque(maxlen=angle_smoothing_window)
        self._low_rotation_count: int = 0

        # Transition state
        self._transition_frames: list[np.ndarray] = []

        # Episode tracking
        self._rotation_episodes: list[RotationEpisode] = []
        self._current_episode: RotationEpisode | None = None

        # Improved center estimation
        self._rotation_start_frame: np.ndarray | None = None  # Frame when rotation started
        self._center_estimates: list[tuple[float, float, float]] = []  # (cx, cy, weight)
        self._min_angle_for_center: float = 3.0  # Minimum cumulative angle for reliable center estimate
        self._center_update_interval: int = 10  # Frames between center updates
        self._frames_since_center_update: int = 0

    def initialize(
        self,
        max_frames: int | None = None,
        initial_center_estimate: tuple[float, float] | None = None,
    ) -> None:
        """
        Initialize the manager: compute initial static background.

        Args:
            max_frames: Limit frames for background sampling
            initial_center_estimate: Optional hint for rotation center
        """
        self._rotation_center = initial_center_estimate
        self._static_background = self._compute_initial_background(max_frames)
        self._reference_background = self._static_background.copy()

    def _compute_initial_background(
        self,
        max_frames: int | None = None,
    ) -> np.ndarray:
        """Compute initial background from video start."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            cap.release()
            raise ValueError("Video has no frames")

        frames_to_use = min(total_frames, max_frames) if max_frames else total_frames
        indices = np.linspace(0, min(frames_to_use - 1, 100), self.initial_bg_samples).astype(int)

        samples = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                samples.append(gray.astype(np.float32))

        cap.release()

        if not samples:
            raise ValueError("No frames available to compute background")

        background = np.median(samples, axis=0).astype(np.uint8)
        return background

    def get_background(
        self,
        frame_idx: int,
        frame: np.ndarray,
    ) -> np.ndarray:
        """
        Get the appropriate background for the given frame.

        Args:
            frame_idx: 0-based frame index
            frame: Current frame (grayscale uint8)

        Returns:
            Background image (grayscale uint8)
        """
        # Ensure frame is grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # First frame initialization
        if self._prev_frame is None:
            self._prev_frame = gray.copy()
            return self._static_background

        # Estimate rotation
        rotation_estimate = self.estimator.estimate(self._prev_frame, gray)

        # Update state machine
        self._update_state(frame_idx, gray, rotation_estimate)

        # Store current frame for next iteration
        self._prev_frame = gray.copy()

        # Return appropriate background based on state
        if self._state == RotationState.STATIC:
            return self._static_background

        elif self._state == RotationState.ROTATING:
            # Rotate reference background by cumulative angle
            return self._rotate_background(self._cumulative_rotation_deg)

        else:  # TRANSITION
            # Use last rotated background while collecting frames
            return self._rotate_background(self._cumulative_rotation_deg)

    def _update_state(
        self,
        frame_idx: int,
        frame: np.ndarray,
        rotation_estimate: RotationEstimate | None,
    ) -> None:
        """Update state machine based on rotation estimate."""

        # Extract angle (0 if no estimate)
        if rotation_estimate is not None:
            angle = rotation_estimate.angle_deg
            self._angle_history.append(angle)
        else:
            angle = 0.0
            self._angle_history.append(0.0)

        # Compute smoothed angle
        smoothed_angle = np.median(list(self._angle_history)) if self._angle_history else 0.0

        if self._state == RotationState.STATIC:
            # Check for rotation onset
            if abs(smoothed_angle) > self.rotation_start_threshold_deg:
                self._enter_rotating_state(frame_idx)

        elif self._state == RotationState.ROTATING:
            # Accumulate rotation
            self._cumulative_rotation_deg += smoothed_angle

            # Update current episode
            if self._current_episode is not None:
                self._current_episode.total_rotation_deg = self._cumulative_rotation_deg

            # Periodically update center estimate using start frame comparison
            self._frames_since_center_update += 1
            if (self._frames_since_center_update >= self._center_update_interval and
                abs(self._cumulative_rotation_deg) >= self._min_angle_for_center):
                self._update_center_from_start_frame(frame)
                self._frames_since_center_update = 0

            # Check for rotation stop
            if abs(smoothed_angle) < self.rotation_stop_threshold_deg:
                self._low_rotation_count += 1
                if self._low_rotation_count >= self.rotation_stop_frames:
                    self._enter_transition_state(frame_idx)
            else:
                self._low_rotation_count = 0

        elif self._state == RotationState.TRANSITION:
            # Collect frames for new background
            self._transition_frames.append(frame.copy())

            if len(self._transition_frames) >= self.transition_bg_frames:
                self._complete_transition()

    def _enter_rotating_state(self, frame_idx: int) -> None:
        """Transition from STATIC to ROTATING."""
        self._state = RotationState.ROTATING
        self._reference_background = self._static_background.copy()
        self._cumulative_rotation_deg = 0.0
        self._low_rotation_count = 0
        self._center_estimates = []
        self._frames_since_center_update = 0

        # Save the start frame for center estimation
        if self._prev_frame is not None:
            self._rotation_start_frame = self._prev_frame.copy()

        # Start new episode
        self._current_episode = RotationEpisode(start_frame=frame_idx)

        print(f"[AdaptiveBG] Rotation detected at frame {frame_idx}")

    def _update_center_from_start_frame(self, current_frame: np.ndarray) -> None:
        """
        Estimate rotation center by comparing start frame to current frame.

        This uses the larger cumulative rotation angle for more stable estimation.
        The center is weighted by the rotation angle magnitude.
        """
        if self._rotation_start_frame is None:
            return

        # Estimate rotation from start frame to current frame
        estimate = self.estimator.estimate(self._rotation_start_frame, current_frame)

        if estimate is None:
            return

        # Only use estimates where the angle is significant (more reliable)
        if abs(estimate.angle_deg) < self._min_angle_for_center:
            return

        # Weight by angle magnitude - larger angles give more reliable centers
        weight = abs(estimate.angle_deg) * estimate.confidence

        # Store weighted estimate
        self._center_estimates.append((estimate.center[0], estimate.center[1], weight))

        # Compute weighted average and update rotation center
        self._rotation_center = self._compute_weighted_center()

    def _compute_weighted_center(self) -> tuple[float, float] | None:
        """Compute weighted average of center estimates."""
        if not self._center_estimates:
            return None

        total_weight = sum(w for _, _, w in self._center_estimates)
        if total_weight < 1e-6:
            return None

        cx = sum(x * w for x, _, w in self._center_estimates) / total_weight
        cy = sum(y * w for _, y, w in self._center_estimates) / total_weight

        return (cx, cy)

    def _enter_transition_state(self, frame_idx: int) -> None:
        """Transition from ROTATING to TRANSITION."""
        self._state = RotationState.TRANSITION
        self._transition_frames = []

        # Finalize rotation center estimate using weighted average
        if self._center_estimates:
            self._rotation_center = self._compute_weighted_center()

            # Log center estimate quality
            n_estimates = len(self._center_estimates)
            total_weight = sum(w for _, _, w in self._center_estimates)
            print(f"[AdaptiveBG] Center estimated from {n_estimates} samples, total weight: {total_weight:.1f}")

        # Finalize current episode
        if self._current_episode is not None:
            self._current_episode.end_frame = frame_idx
            self._current_episode.rotation_center = self._rotation_center or (0, 0)
            n_frames = frame_idx - self._current_episode.start_frame
            if n_frames > 0:
                self._current_episode.angular_velocity_deg = (
                    self._current_episode.total_rotation_deg / n_frames
                )
            self._rotation_episodes.append(self._current_episode)
            self._current_episode = None

        print(f"[AdaptiveBG] Rotation stopped at frame {frame_idx}, "
              f"total rotation: {self._cumulative_rotation_deg:.1f} deg, "
              f"center: {self._rotation_center}")

    def _complete_transition(self) -> None:
        """Complete transition: compute new static background."""
        # Compute new background from collected frames
        samples = [f.astype(np.float32) for f in self._transition_frames]
        self._static_background = np.median(samples, axis=0).astype(np.uint8)
        self._reference_background = self._static_background.copy()

        # Reset state
        self._state = RotationState.STATIC
        self._transition_frames = []
        self._cumulative_rotation_deg = 0.0
        self._angle_history.clear()

        print("[AdaptiveBG] New static background computed, returning to STATIC state")

    def _rotate_background(self, angle_deg: float) -> np.ndarray:
        """
        Rotate reference background by specified angle.

        Args:
            angle_deg: Rotation angle in degrees (positive = clockwise)

        Returns:
            Rotated background image
        """
        if self._reference_background is None:
            raise ValueError("Reference background not initialized")

        h, w = self._reference_background.shape[:2]

        # Use estimated center or frame center
        if self._rotation_center is not None:
            center = self._rotation_center
        else:
            center = (w / 2.0, h / 2.0)

        # OpenCV uses counter-clockwise rotation, so negate angle
        M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

        rotated = cv2.warpAffine(
            self._reference_background,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    def get_state(self) -> RotationState:
        """Get current state."""
        return self._state

    def get_rotation_episodes(self) -> list[RotationEpisode]:
        """Get all recorded rotation episodes."""
        return self._rotation_episodes.copy()

    def get_cumulative_rotation(self) -> float:
        """Get total rotation from reference in degrees."""
        return self._cumulative_rotation_deg

    def get_rotation_center(self) -> tuple[float, float] | None:
        """Get estimated rotation center."""
        return self._rotation_center

    def get_last_smoothed_angle(self) -> float:
        """Get the last smoothed rotation angle."""
        if self._angle_history:
            return float(np.median(list(self._angle_history)))
        return 0.0

    def get_angle_history(self) -> list[float]:
        """Get the recent angle history."""
        return list(self._angle_history)

    def get_low_rotation_count(self) -> int:
        """Get count of consecutive low-rotation frames."""
        return self._low_rotation_count

    def get_transition_progress(self) -> tuple[int, int]:
        """Get transition progress (current frames, required frames)."""
        return len(self._transition_frames), self.transition_bg_frames

    def get_center_estimate_count(self) -> int:
        """Get number of center estimates collected."""
        return len(self._center_estimates)

    def get_center_estimate_weight(self) -> float:
        """Get total weight of center estimates."""
        return sum(w for _, _, w in self._center_estimates)


def visualize_adaptive_background(
    video_path: str,
    output_path: str,
    max_frames: int | None = None,
    # Adaptive background parameters
    rotation_start_threshold_deg: float = 0.5,
    rotation_stop_threshold_deg: float = 0.1,
    rotation_center: tuple[float, float] | None = None,
    # Visualization options
    diff_threshold: int = 10,
    show_features: bool = False,
):
    """
    Create a diagnostic visualization video showing the adaptive background process.

    This visualization shows:
    - Original frame vs current background vs difference image
    - State machine status (STATIC/ROTATING/TRANSITION)
    - Per-frame rotation angle (raw and smoothed)
    - Cumulative rotation
    - Estimated rotation center
    - Feature matches (optional)

    Args:
        video_path: Path to input video
        output_path: Path for output diagnostic video
        max_frames: Maximum frames to process (None for all)
        rotation_start_threshold_deg: Threshold to trigger rotation detection
        rotation_stop_threshold_deg: Threshold to consider rotation stopped
        rotation_center: Fixed rotation center, or None for auto-detect
        diff_threshold: Threshold for binary mask visualization
        show_features: Show ORB feature matches (slower but more informative)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is None:
        max_frames = total_frames

    # Create adaptive background manager
    print("Initializing adaptive background manager for visualization...")
    manager = AdaptiveBackgroundManager(
        video_path,
        rotation_start_threshold_deg=rotation_start_threshold_deg,
        rotation_stop_threshold_deg=rotation_stop_threshold_deg,
    )
    manager.initialize(
        max_frames=max_frames,
        initial_center_estimate=rotation_center,
    )

    # Output video layout:
    # Top row: Original | Background | Difference
    # Bottom: Info panel with state, angles, etc.
    panel_width = width
    panel_height = height
    info_height = 120

    out_width = panel_width * 3
    out_height = panel_height + info_height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Creating adaptive background diagnostic video: {output_path}")
    print(f"  Output size: {out_width}x{out_height}")
    print(f"  Rotation thresholds: start={rotation_start_threshold_deg}°, stop={rotation_stop_threshold_deg}°")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # For feature visualization
    prev_gray = None
    estimator = manager.estimator if show_features else None

    # Colors for states
    STATE_COLORS = {
        RotationState.STATIC: (0, 255, 0),      # Green
        RotationState.ROTATING: (0, 165, 255),  # Orange
        RotationState.TRANSITION: (255, 0, 255), # Magenta
    }

    # History for plotting
    angle_plot_history = []
    max_plot_points = 200

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get background from manager (this updates the state machine)
        background = manager.get_background(frame_idx, gray)

        # Compute difference
        diff = cv2.absdiff(gray, background)
        _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)

        # Get current state info
        state = manager.get_state()
        cumulative_rotation = manager.get_cumulative_rotation()
        rotation_center_est = manager.get_rotation_center()
        smoothed_angle = manager.get_last_smoothed_angle()
        low_rot_count = manager.get_low_rotation_count()
        transition_progress = manager.get_transition_progress()

        # Track angle history for plotting
        angle_plot_history.append(smoothed_angle)
        if len(angle_plot_history) > max_plot_points:
            angle_plot_history.pop(0)

        # Create output panels
        # Panel 1: Original frame with annotations
        panel1 = frame.copy()
        cv2.putText(panel1, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw rotation center if available
        if rotation_center_est is not None:
            cx, cy = int(rotation_center_est[0]), int(rotation_center_est[1])
            cv2.drawMarker(panel1, (cx, cy), (0, 0, 255),
                          cv2.MARKER_CROSS, 20, 2)
            cv2.circle(panel1, (cx, cy), 5, (0, 0, 255), -1)

        # Panel 2: Current background
        panel2 = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2, "Background", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Draw rotation center on background too
        if rotation_center_est is not None:
            cx, cy = int(rotation_center_est[0]), int(rotation_center_est[1])
            cv2.drawMarker(panel2, (cx, cy), (0, 0, 255),
                          cv2.MARKER_CROSS, 20, 2)

        # Panel 3: Difference/mask
        # Show diff with mask overlay
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        panel3 = cv2.cvtColor(diff_normalized, cv2.COLOR_GRAY2BGR)
        # Highlight thresholded regions in red
        panel3[mask > 0] = [0, 0, 255]
        cv2.putText(panel3, f"Diff (thresh={diff_threshold})", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Combine top row
        top_row = np.hstack([panel1, panel2, panel3])

        # Create info panel
        info_panel = np.zeros((info_height, out_width, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)

        # State indicator
        state_color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.rectangle(info_panel, (10, 10), (200, 50), state_color, -1)
        cv2.rectangle(info_panel, (10, 10), (200, 50), (255, 255, 255), 2)
        cv2.putText(info_panel, state.value.upper(), (25, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Frame info
        cv2.putText(info_panel, f"Frame: {frame_idx}", (220, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(info_panel, f"/{max_frames}", (220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Rotation info
        cv2.putText(info_panel, f"Angle/frame: {smoothed_angle:+.2f} deg", (350, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Cumulative: {cumulative_rotation:+.1f} deg", (350, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Center estimate
        if rotation_center_est is not None:
            cv2.putText(info_panel, f"Center: ({rotation_center_est[0]:.0f}, {rotation_center_est[1]:.0f})",
                        (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # State-specific info
        if state == RotationState.ROTATING:
            cv2.putText(info_panel, f"Low-rot frames: {low_rot_count}/{manager.rotation_stop_frames}",
                        (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            # Show center estimation progress
            center_count = manager.get_center_estimate_count()
            center_weight = manager.get_center_estimate_weight()
            cv2.putText(info_panel, f"Center samples: {center_count} (wt: {center_weight:.1f})",
                        (600, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        elif state == RotationState.TRANSITION:
            curr, total = transition_progress
            cv2.putText(info_panel, f"Transition: {curr}/{total} frames",
                        (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # Thresholds info
        cv2.putText(info_panel, f"Start thresh: {rotation_start_threshold_deg} deg",
                    (600, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        cv2.putText(info_panel, f"Stop thresh: {rotation_stop_threshold_deg} deg",
                    (600, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

        # Mini angle plot
        plot_x_start = 800
        plot_width = out_width - plot_x_start - 20
        plot_height = 80
        plot_y_center = 60

        if len(angle_plot_history) > 1 and plot_width > 0:
            # Draw plot background
            cv2.rectangle(info_panel,
                         (plot_x_start, plot_y_center - plot_height // 2),
                         (plot_x_start + plot_width, plot_y_center + plot_height // 2),
                         (60, 60, 60), -1)

            # Draw zero line
            cv2.line(info_panel,
                    (plot_x_start, plot_y_center),
                    (plot_x_start + plot_width, plot_y_center),
                    (100, 100, 100), 1)

            # Draw threshold lines
            max_angle_display = 2.0  # degrees
            thresh_y_start = int(plot_y_center - (rotation_start_threshold_deg / max_angle_display) * (plot_height // 2))
            thresh_y_stop = int(plot_y_center - (rotation_stop_threshold_deg / max_angle_display) * (plot_height // 2))
            cv2.line(info_panel, (plot_x_start, thresh_y_start), (plot_x_start + plot_width, thresh_y_start), (0, 100, 0), 1)
            cv2.line(info_panel, (plot_x_start, thresh_y_stop), (plot_x_start + plot_width, thresh_y_stop), (0, 50, 0), 1)

            # Draw angle history
            points = []
            for i, angle in enumerate(angle_plot_history):
                x = plot_x_start + int(i * plot_width / max_plot_points)
                # Clamp angle for display
                clamped_angle = max(-max_angle_display, min(max_angle_display, angle))
                y = int(plot_y_center - (clamped_angle / max_angle_display) * (plot_height // 2))
                points.append((x, y))

            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(info_panel, points[i], points[i + 1], (0, 255, 255), 1)

            cv2.putText(info_panel, "Angle", (plot_x_start, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Combine panels
        output_frame = np.vstack([top_row, info_panel])
        out.write(output_frame)

        if (frame_idx + 1) % 100 == 0:
            print(f"  Processed frame {frame_idx + 1}/{max_frames}")

        prev_gray = gray.copy()

    cap.release()
    out.release()

    # Print summary
    episodes = manager.get_rotation_episodes()
    print(f"\nDiagnostic video saved to: {output_path}")
    print(f"Detected {len(episodes)} rotation episode(s):")
    for i, ep in enumerate(episodes):
        print(f"  Episode {i + 1}: frames {ep.start_frame}-{ep.end_frame}, "
              f"rotation: {ep.total_rotation_deg:.1f} deg, "
              f"center: ({ep.rotation_center[0]:.0f}, {ep.rotation_center[1]:.0f})")
