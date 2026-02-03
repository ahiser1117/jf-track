"""
Adaptive background subtraction with rotation compensation.

This module provides classes for handling videos where the background rotates
around a fixed center point. It uses evenly spaced frames from each stationary
episode, estimates rotation for alignment, and computes the background as a
median of aligned frames.
"""

import cv2
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import NamedTuple

from skimage.filters import threshold_otsu

from src.mask_utils import clean_binary_mask


class RotationState(Enum):
    """State machine states for background management."""

    STATIC = "static"  # No rotation, use static background
    ROTATING = "rotating"  # Active rotation, compensate background
    TRANSITION = "transition"  # Post-rotation, collecting frames for new background


@dataclass
class RotationEstimate:
    """Result from frame-to-frame rotation estimation."""

    angle_deg: float  # Rotation angle in degrees (positive = clockwise)
    center: tuple[float, float]  # (cx, cy) estimated rotation center
    confidence: float  # 0-1, quality of estimate based on inlier ratio
    n_inliers: int  # Number of matched feature inliers


@dataclass
class RotationEpisode:
    """Records a single rotation episode."""

    start_frame: int
    end_frame: int | None = None  # None if ongoing
    rotation_center: tuple[float, float] = (0.0, 0.0)
    angular_velocity_deg: float = 0.0  # Average degrees per frame
    total_rotation_deg: float = 0.0  # Cumulative rotation


class BufferedFrame(NamedTuple):
    """A frame stored in the rolling buffer with its cumulative rotation."""

    frame: np.ndarray  # Grayscale frame
    frame_idx: int  # Original frame index
    cumulative_rotation: float  # Degrees rotated since this frame was added


class RollingBackgroundManager:
    """
    Manages a background computed once per stationary episode.

    The background is computed as a median of evenly spaced frames
    across the episode and held constant afterward.
    """

    def __init__(self, buffer_size: int = 10):
        """
        Args:
            buffer_size: Number of frames to sample for background computation
        """
        self.buffer_size = buffer_size
        self._episode_frames: list[np.ndarray] = []
        self._cached_background: np.ndarray | None = None
        self._background_ready: bool = False
        self._cache_valid: bool = False

    def update(self, frame: np.ndarray) -> None:
        """
        Add a new frame to the sampled episode.

        Args:
            frame: Grayscale frame to add
        """
        if self._background_ready:
            return

        self._episode_frames.append(frame.copy())
        self._cache_valid = False
        self._compute_episode_background()

    def get_background(self) -> np.ndarray:
        """
        Get the current background computed from sampled frames.

        Returns:
            Background image as grayscale uint8
        """
        if not self._episode_frames:
            raise ValueError("No frames in buffer")

        if self._cache_valid and self._cached_background is not None:
            return self._cached_background

        if not self._background_ready:
            self._compute_episode_background()

        if self._cached_background is None:
            raise ValueError("Background not available")

        return self._cached_background

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for a valid background."""
        return self._background_ready

    def buffer_fill(self) -> tuple[int, int]:
        """Get current buffer fill (current, max)."""
        return len(self._episode_frames), self.buffer_size

    def _compute_episode_background(self) -> None:
        """Compute background from evenly spaced episode frames."""
        if self._background_ready:
            return

        total_frames = len(self._episode_frames)
        if total_frames < min(3, self.buffer_size):
            return

        n_samples = min(self.buffer_size, total_frames)
        indices = np.linspace(0, total_frames - 1, n_samples).astype(int)
        frames = np.array(
            [self._episode_frames[idx].astype(np.float32) for idx in indices]
        )
        self._cached_background = np.median(frames, axis=0).astype(np.uint8)
        self._background_ready = True
        self._cache_valid = True
        self._episode_frames = []


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
            pts1,
            pts2,
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
    Manages adaptive background for rotation-compensated tracking using
    a rolling buffer of aligned frames.

    State Machine:
    - STATIC: Uses precomputed median of the stationary episode
    - ROTATING: Aligns buffered frames to current orientation before averaging
    - TRANSITION: Collecting frames for the next static episode

    The key improvement is using sampled frames from stationary episodes rather
    than a single reference. During rotation, each buffered frame is rotated to
    align with the current frame orientation before computing the median background.

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
        buffer_size: int = 10,
        transition_bg_frames: int = 30,
        # Rotation detection thresholds
        rotation_start_threshold_deg: float = 0.5,
        rotation_stop_threshold_deg: float = 0.1,
        rotation_stop_frames: int = 10,
        rotation_start_frames: int = 3,
        rotation_confidence_threshold: float = 0.3,
        # Rotation estimation
        n_features: int = 500,
        min_inliers: int = 20,
        # Smoothing
        angle_smoothing_window: int = 5,
        # Episode filtering
        min_episode_rotation_deg: float = 5.0,
    ):
        """
        Args:
            video_path: Path to video file
            buffer_size: Number of recent frames to use for rotation-aligned background
            transition_bg_frames: Frames to collect after rotation stops
            rotation_start_threshold_deg: Min angle/frame to trigger rotation
            rotation_stop_threshold_deg: Max angle/frame to consider stopped
            rotation_stop_frames: Consecutive low-rotation frames to exit
            rotation_start_frames: Consecutive high-rotation frames required to enter rotation
            rotation_confidence_threshold: Minimum confidence needed to accept a rotation estimate
            n_features: ORB features for rotation estimation
            min_inliers: Minimum feature matches for valid rotation
            angle_smoothing_window: Frames to smooth rotation estimates
            min_episode_rotation_deg: Minimum absolute rotation required to keep an episode
        """
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.transition_bg_frames = transition_bg_frames
        self.rotation_start_threshold_deg = rotation_start_threshold_deg
        self.rotation_stop_threshold_deg = rotation_stop_threshold_deg
        self.rotation_stop_frames = rotation_stop_frames
        self.rotation_start_frames = max(1, int(rotation_start_frames))
        self.rotation_confidence_threshold = rotation_confidence_threshold
        self.min_episode_rotation_deg = min_episode_rotation_deg
        self.angle_smoothing_window = angle_smoothing_window

        # Create rotation estimator
        self.estimator = RotationEstimator(
            n_features=n_features,
            min_inliers=min_inliers,
        )

        # State
        self._state = RotationState.STATIC
        self._rotation_center: tuple[float, float] | None = None
        self._cumulative_rotation_deg: float = 0.0
        self._prev_frame: np.ndarray | None = None
        self._frame_shape: tuple[int, int] | None = None

        # Rolling frame buffer: stores (frame, cumulative_rotation_when_added)
        self._frame_buffer: deque[BufferedFrame] = deque(maxlen=buffer_size)

        # Static episode tracking
        self._static_episode_frames: list[np.ndarray] = []
        self._static_background_ready: bool = False

        # Rotation detection history
        self._angle_history: deque[float] = deque(maxlen=angle_smoothing_window)
        self._low_rotation_count: int = 0
        self._high_rotation_count: int = 0
        self._last_confidence: float = 0.0

        # Transition state
        self._transition_frames: list[np.ndarray] = []

        # Episode tracking
        self._rotation_episodes: list[RotationEpisode] = []
        self._current_episode: RotationEpisode | None = None
        self._discarded_episodes: int = 0

        # Center estimation during rotation
        self._rotation_start_frame: np.ndarray | None = None
        self._center_estimates: list[
            tuple[float, float, float]
        ] = []  # (cx, cy, weight)
        self._min_angle_for_center: float = 3.0
        self._center_update_interval: int = 10
        self._frames_since_center_update: int = 0

        # Background cache
        self._cached_background: np.ndarray | None = None
        self._cache_valid: bool = False

    def initialize(
        self,
        max_frames: int | None = None,
        initial_center_estimate: tuple[float, float] | None = None,
    ) -> None:
        """
        Initialize the manager.

        Args:
            max_frames: Unused (kept for API compatibility)
            initial_center_estimate: Optional hint for rotation center
        """
        self._rotation_center = initial_center_estimate
        # Buffer will be filled as frames are processed

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

        if self._frame_shape is None:
            self._frame_shape = gray.shape[:2]

        # First frame: add to buffer and return frame itself
        if self._prev_frame is None:
            self._prev_frame = gray.copy()
            self._add_frame_to_buffer(gray, frame_idx)
            self._add_static_episode_frame(gray)
            return gray  # No background yet

        # Estimate rotation between consecutive frames
        rotation_estimate = self.estimator.estimate(self._prev_frame, gray)

        # Update state machine
        self._update_state(frame_idx, gray, rotation_estimate)

        if self._state == RotationState.STATIC:
            self._add_static_episode_frame(gray)

        # Add current frame to buffer
        self._add_frame_to_buffer(gray, frame_idx)

        # Store current frame for next iteration
        self._prev_frame = gray.copy()

        if self._state == RotationState.STATIC:
            self._maybe_compute_static_background()
            if not self._static_background_ready:
                return gray

        # Compute and return background
        return self._compute_background()

    def update_state(self, frame_idx: int, frame: np.ndarray) -> None:
        """
        Update rotation state and buffers without computing background.

        Args:
            frame_idx: 0-based frame index
            frame: Current frame (grayscale uint8)
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self._frame_shape is None:
            self._frame_shape = gray.shape[:2]

        if self._prev_frame is None:
            self._prev_frame = gray.copy()
            self._add_frame_to_buffer(gray, frame_idx)
            return

        rotation_estimate = self.estimator.estimate(self._prev_frame, gray)
        self._update_state(frame_idx, gray, rotation_estimate)
        self._add_frame_to_buffer(gray, frame_idx)
        self._prev_frame = gray.copy()

    def _add_frame_to_buffer(self, frame: np.ndarray, frame_idx: int) -> None:
        """Add frame to rolling buffer with current cumulative rotation."""
        buffered = BufferedFrame(
            frame=frame.copy(),
            frame_idx=frame_idx,
            cumulative_rotation=self._cumulative_rotation_deg,
        )
        self._frame_buffer.append(buffered)
        self._cache_valid = False

    def _compute_background(self) -> np.ndarray:
        """
        Compute background from buffered frames.

        In STATIC mode: use cached background for the stationary episode.
        In ROTATING mode: align each buffered frame to current orientation, then median.
        """
        if not self._frame_buffer:
            raise ValueError("No frames in buffer")

        if self._cache_valid and self._cached_background is not None:
            return self._cached_background

        if self._state == RotationState.STATIC:
            if not self._static_background_ready or self._cached_background is None:
                raise ValueError("Static background not ready")
            return self._cached_background

        else:  # ROTATING or TRANSITION
            # Align each buffered frame to current orientation
            aligned_frames = []
            current_rotation = self._cumulative_rotation_deg

            for bf in self._frame_buffer:
                # How much to rotate this frame to align with current
                rotation_offset = current_rotation - bf.cumulative_rotation

                if abs(rotation_offset) < 0.01:
                    # No rotation needed
                    aligned_frames.append(bf.frame.astype(np.float32))
                else:
                    # Rotate frame to align
                    aligned = self._rotate_frame(bf.frame, rotation_offset)
                    aligned_frames.append(aligned.astype(np.float32))

            frames = np.array(aligned_frames)
            self._cached_background = np.median(frames, axis=0).astype(np.uint8)

        self._cache_valid = True
        return self._cached_background

    def _rotate_frame(self, frame: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate a frame by specified angle.

        Args:
            frame: Grayscale frame to rotate
            angle_deg: Rotation angle in degrees (positive = clockwise)

        Returns:
            Rotated frame
        """
        h, w = frame.shape[:2]

        # Use estimated center or frame center
        if self._rotation_center is not None:
            center = self._rotation_center
        else:
            center = (w / 2.0, h / 2.0)

        # OpenCV uses counter-clockwise rotation, so negate angle
        M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

        rotated = cv2.warpAffine(
            frame,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

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
            confidence = rotation_estimate.confidence
        else:
            angle = 0.0
            confidence = 0.0

        self._last_confidence = confidence
        self._angle_history.append(angle)

        # Compute smoothed angle
        smoothed_angle = (
            np.median(list(self._angle_history)) if self._angle_history else 0.0
        )

        if self._state == RotationState.STATIC:
            # Check for rotation onset
            if (
                abs(smoothed_angle) > self.rotation_start_threshold_deg
                and confidence >= self.rotation_confidence_threshold
            ):
                self._high_rotation_count += 1
            else:
                self._high_rotation_count = 0

            if self._high_rotation_count >= self.rotation_start_frames:
                self._enter_rotating_state(frame_idx)
                self._cumulative_rotation_deg += smoothed_angle
                self._cache_valid = False
                if self._current_episode is not None:
                    self._current_episode.total_rotation_deg = (
                        self._cumulative_rotation_deg
                    )
                self._high_rotation_count = 0

        elif self._state == RotationState.ROTATING:
            self._high_rotation_count = 0
            # Accumulate rotation
            self._cumulative_rotation_deg += smoothed_angle
            self._cache_valid = False  # Alignment changes each frame

            # Update current episode
            if self._current_episode is not None:
                self._current_episode.total_rotation_deg = self._cumulative_rotation_deg

            # Periodically update center estimate
            self._frames_since_center_update += 1
            if (
                self._frames_since_center_update >= self._center_update_interval
                and abs(self._cumulative_rotation_deg) >= self._min_angle_for_center
            ):
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
            self._high_rotation_count = 0
            # Continue accumulating rotation during transition
            self._cumulative_rotation_deg += smoothed_angle
            self._cache_valid = False

            # Collect frames for transition
            self._transition_frames.append(frame.copy())

            if len(self._transition_frames) >= self.transition_bg_frames:
                self._complete_transition()

    def _enter_rotating_state(self, frame_idx: int) -> None:
        """Transition from STATIC to ROTATING."""
        self._state = RotationState.ROTATING
        self._cumulative_rotation_deg = 0.0
        self._low_rotation_count = 0
        self._center_estimates = []
        self._frames_since_center_update = 0
        self._cache_valid = False
        self._reset_static_episode_frames()

        # Reset cumulative rotation for all buffered frames
        # (they're all at the same orientation when rotation starts)
        new_buffer = deque(maxlen=self.buffer_size)
        for bf in self._frame_buffer:
            new_buffer.append(
                BufferedFrame(
                    frame=bf.frame,
                    frame_idx=bf.frame_idx,
                    cumulative_rotation=0.0,
                )
            )
        self._frame_buffer = new_buffer

        # Save start frame for center estimation
        if self._prev_frame is not None:
            self._rotation_start_frame = self._prev_frame.copy()

        # Start new episode
        self._current_episode = RotationEpisode(start_frame=frame_idx)

        print(f"[AdaptiveBG] Rotation detected at frame {frame_idx}")

    def _update_center_from_start_frame(self, current_frame: np.ndarray) -> None:
        """Estimate rotation center by comparing start frame to current."""
        if self._rotation_start_frame is None:
            return

        estimate = self.estimator.estimate(self._rotation_start_frame, current_frame)

        if estimate is None:
            return

        if abs(estimate.angle_deg) < self._min_angle_for_center:
            return

        weight = abs(estimate.angle_deg) * estimate.confidence
        self._center_estimates.append((estimate.center[0], estimate.center[1], weight))
        self._rotation_center = self._compute_weighted_center()
        self._cache_valid = False

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

        # Finalize center estimate
        if self._center_estimates:
            self._rotation_center = self._compute_weighted_center()
            n_estimates = len(self._center_estimates)
            total_weight = sum(w for _, _, w in self._center_estimates)
            print(
                f"[AdaptiveBG] Center estimated from {n_estimates} samples, weight: {total_weight:.1f}"
            )

        # Finalize episode
        if self._current_episode is not None:
            self._current_episode.total_rotation_deg = self._cumulative_rotation_deg
            self._current_episode.end_frame = frame_idx
            self._current_episode.rotation_center = self._rotation_center or (0, 0)
            n_frames = frame_idx - self._current_episode.start_frame
            if n_frames > 0:
                self._current_episode.angular_velocity_deg = (
                    self._current_episode.total_rotation_deg / n_frames
                )
            total_rotation = abs(self._current_episode.total_rotation_deg)
            if total_rotation < self.min_episode_rotation_deg:
                self._discarded_episodes += 1
                print(
                    f"[AdaptiveBG] Ignoring spurious rotation episode (total {total_rotation:.2f} deg < {self.min_episode_rotation_deg} deg)"
                )
            else:
                self._rotation_episodes.append(self._current_episode)
                print(
                    f"[AdaptiveBG] Rotation stopped at frame {frame_idx}, "
                    f"total: {self._cumulative_rotation_deg:.1f} deg, "
                    f"center: {self._rotation_center}"
                )
            self._current_episode = None

    def _complete_transition(self) -> None:
        """Complete transition: reset buffer with transition frames."""
        # Clear buffer and add transition frames
        self._frame_buffer.clear()
        start_idx = max(0, len(self._transition_frames) - self.buffer_size)
        for i, frame in enumerate(self._transition_frames[start_idx:]):
            self._frame_buffer.append(
                BufferedFrame(
                    frame=frame,
                    frame_idx=-1,  # Unknown original index
                    cumulative_rotation=0.0,
                )
            )

        self._reset_static_episode_frames(self._transition_frames)

        # Reset state
        self._state = RotationState.STATIC
        self._transition_frames = []
        self._cumulative_rotation_deg = 0.0
        self._angle_history.clear()
        self._cache_valid = False

        print("[AdaptiveBG] Transition complete, returning to STATIC state")

    def _add_static_episode_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the stationary episode buffer."""
        if self._static_background_ready:
            return

        self._static_episode_frames.append(frame.copy())
        self._cache_valid = False

    def _reset_static_episode_frames(
        self, frames: list[np.ndarray] | None = None
    ) -> None:
        """Reset static episode frames and optionally seed from frames."""
        self._static_episode_frames = []
        self._static_background_ready = False

        if frames:
            for frame in frames:
                self._static_episode_frames.append(frame.copy())
        self._cache_valid = False

    def _maybe_compute_static_background(self) -> None:
        """Compute static background once per episode if ready."""
        if self._static_background_ready:
            return

        total_frames = len(self._static_episode_frames)
        if total_frames < min(3, self.buffer_size):
            return

        n_samples = min(self.buffer_size, total_frames)
        indices = np.linspace(0, total_frames - 1, n_samples).astype(int)
        frames = np.array(
            [self._static_episode_frames[idx].astype(np.float32) for idx in indices]
        )
        self._cached_background = np.median(frames, axis=0).astype(np.uint8)
        self._static_background_ready = True
        self._static_episode_frames = []
        self._cache_valid = True

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

    def get_buffer_fill(self) -> tuple[int, int]:
        """Get buffer fill (current, max)."""
        return len(self._frame_buffer), self.buffer_size

    def is_ready(self) -> bool:
        """Check if background is ready for use."""
        if self._state == RotationState.STATIC:
            return self._static_background_ready
        return len(self._frame_buffer) >= min(3, self.buffer_size)


class BackgroundProcessor:
    """
    Centralized background processing for tracking and visualization.

    This class unifies the background subtraction logic used in both
    tracking and visualization, ensuring consistent behavior.

    It handles:
    - Rolling or adaptive background management
    - Background subtraction and thresholding
    - ROI mask application
    - Search position rotation during rotation episodes
    """

    def __init__(
        self,
        video_path: str,
        width: int,
        height: int,
        background_buffer_size: int = 10,
        threshold: int = 10,
        adaptive_background: bool = False,
        rotation_start_threshold_deg: float = 0.01,
        rotation_stop_threshold_deg: float = 0.005,
        rotation_start_frames: int = 3,
        rotation_confidence_threshold: float = 0.3,
        min_episode_rotation_deg: float = 5.0,
        rotation_center: tuple[float, float] | None = None,
        max_frames: int | None = None,
        roi_mask: np.ndarray | None = None,
        use_auto_threshold: bool = False,
    ):
        """
        Initialize the background processor.

        Args:
            video_path: Path to video file (required for AdaptiveBackgroundManager)
            width: Frame width in pixels
            height: Frame height in pixels
            background_buffer_size: Number of frames sampled per episode for background
            threshold: Binary threshold for background subtraction
            adaptive_background: Enable rotation-compensated background
            rotation_start_threshold_deg: Degrees/frame to trigger rotation detection
            rotation_stop_threshold_deg: Degrees/frame to consider rotation stopped
            rotation_start_frames: Consecutive high-rotation frames required to trigger rotation
            rotation_confidence_threshold: Minimum rotation confidence to accept rotation
            min_episode_rotation_deg: Minimum absolute rotation to keep an episode
            rotation_center: Fixed rotation center (cx, cy), or None for auto-detection
            max_frames: Maximum frames to process (for initialization)
        """
        self.threshold = threshold
        self.adaptive_background = adaptive_background
        self._width = width
        self._height = height
        self._last_background: np.ndarray | None = None

        # Create circular ROI mask (can be overridden by explicit ROI application)
        if roi_mask is not None:
            self._roi_mask = roi_mask
        else:
            self._roi_mask = self._create_circular_roi_mask(height, width)
        self._roi_center = (width // 2, height // 2)
        self._roi_radius = min(width, height) // 2

        # Initialize background manager (separate typed attributes to avoid union type issues)
        self._adaptive_manager: AdaptiveBackgroundManager | None = None
        self._rolling_manager: RollingBackgroundManager | None = None
        self._global_background: np.ndarray | None = None

        if adaptive_background:
            self._adaptive_manager = AdaptiveBackgroundManager(
                video_path,
                buffer_size=background_buffer_size,
                rotation_start_threshold_deg=rotation_start_threshold_deg,
                rotation_stop_threshold_deg=rotation_stop_threshold_deg,
                rotation_start_frames=rotation_start_frames,
                rotation_confidence_threshold=rotation_confidence_threshold,
                min_episode_rotation_deg=min_episode_rotation_deg,
            )
            self._adaptive_manager.initialize(
                max_frames=max_frames,
                initial_center_estimate=rotation_center,
            )
        else:
            background, auto_threshold = self._compute_sampled_background(
                video_path,
                max_frames=max_frames,
                roi_mask=self._roi_mask,
                num_samples=60,
            )
            self._global_background = background
            if use_auto_threshold and auto_threshold is not None:
                self.threshold = max(int(round(auto_threshold)), 5)

    @staticmethod
    def _create_circular_roi_mask(height: int, width: int) -> np.ndarray:
        """Create a circular region-of-interest mask."""
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = min(width, height) // 2
        cv2.circle(mask, center, radius, (255,), -1)
        return mask

    @staticmethod
    def _compute_sampled_background(
        video_path: str,
        max_frames: int | None,
        roi_mask: np.ndarray | None,
        num_samples: int = 60,
    ) -> tuple[np.ndarray, float | None]:
        """Compute a sampled median background to limit memory usage."""

        if num_samples <= 0:
            raise ValueError("num_samples must be positive")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames == 0:
            cap.release()
            raise ValueError("Video contains no readable frames")

        if max_frames is not None and max_frames > 0:
            total_frames = min(total_frames, max_frames)

        num_samples = min(num_samples, total_frames)
        indices = np.linspace(0, total_frames - 1, num_samples).astype(int)

        samples: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if roi_mask is not None:
                gray = cv2.bitwise_and(gray, roi_mask)
            samples.append(gray.astype(np.float32))

        cap.release()

        if not samples:
            raise ValueError("Unable to compute sampled background")

        stack = np.stack(samples, axis=0)
        background = np.median(stack, axis=0).astype(np.uint8)

        diff_values = np.abs(stack - background.astype(np.float32))
        valid = diff_values[diff_values > 2.0]

        auto_threshold: float | None
        if valid.size:
            try:
                auto_threshold = float(threshold_otsu(valid))
            except ValueError:
                auto_threshold = None
        else:
            auto_threshold = None

        return background, auto_threshold


    def process_frame(
        self,
        frame_idx: int,
        gray: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """
        Process a frame through background subtraction.

        Args:
            frame_idx: 0-based frame index
            gray: Grayscale frame (uint8)

        Returns:
            diff: Background subtraction difference image
            mask: Binary mask after thresholding and ROI application
            is_ready: Whether background is ready for use
        """
        if self._adaptive_manager is not None:
            gray_bg = self._adaptive_manager.get_background(frame_idx, gray)
            state = self._adaptive_manager.get_state()
            is_ready = self._adaptive_manager.is_ready() and state == RotationState.STATIC
            if not is_ready:
                return (
                    np.zeros_like(gray),
                    np.zeros_like(gray),
                    False,
                )
        elif self._global_background is not None:
            gray_bg = self._global_background
            is_ready = True
        else:
            raise RuntimeError("No background manager initialized")

        # Background subtraction
        self._last_background = gray_bg

        diff = cv2.absdiff(gray, gray_bg)
        _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_and(mask, self._roi_mask)
        diff = cv2.bitwise_and(diff, self._roi_mask)

        mask = clean_binary_mask(mask, min_area=5)

        return diff, mask, is_ready

    def get_last_background(self) -> np.ndarray | None:
        """Return the most recent background image used for subtraction."""

        return self._last_background

    def rotate_point_if_needed(
        self,
        point: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        """
        Rotate a point if adaptive background detects rotation.

        Use this to rotate search positions during rotation episodes so they
        follow the rotating background.

        Args:
            point: (x, y) coordinates to potentially rotate, or None

        Returns:
            Rotated (x, y) if rotation is active and point is not None, else original point
        """
        if point is None:
            return None

        if self._adaptive_manager is None:
            return point

        state = self._adaptive_manager.get_state()
        if state not in (RotationState.ROTATING, RotationState.TRANSITION):
            return point

        rotation_angle = self._adaptive_manager.get_last_smoothed_angle()
        if abs(rotation_angle) < 0.001:
            return point

        rotation_center = self._adaptive_manager.get_rotation_center()

        # Use frame center as fallback when rotation center not yet estimated
        # (matches the fallback behavior in AdaptiveBackgroundManager._rotate_frame)
        if rotation_center is None:
            rotation_center = (self._width / 2.0, self._height / 2.0)

        # Rotate point
        angle_rad = np.radians(rotation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        px, py = point[0] - rotation_center[0], point[1] - rotation_center[1]
        rx = px * cos_a - py * sin_a
        ry = px * sin_a + py * cos_a

        return (rx + rotation_center[0], ry + rotation_center[1])

    def get_state(self) -> RotationState:
        """Get current rotation state (STATIC if not using adaptive background)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_state()
        return RotationState.STATIC

    def get_rotation_center(self) -> tuple[float, float] | None:
        """Get estimated rotation center (None if not using adaptive background)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_rotation_center()
        return None

    def get_cumulative_rotation(self) -> float:
        """Get cumulative rotation in degrees (0 if not using adaptive background)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_cumulative_rotation()
        return 0.0

    def get_rotation_episodes(self) -> list[RotationEpisode]:
        """Get rotation episodes (empty list if not using adaptive background)."""
        if self._adaptive_manager is not None:
            return self._adaptive_manager.get_rotation_episodes()
        return []

    def get_roi_params(self) -> tuple[tuple[int, int], int]:
        """Get ROI circle parameters (center, radius)."""
        return self._roi_center, self._roi_radius

    def get_roi_mask(self) -> np.ndarray:
        """Get the ROI mask array."""
        return self._roi_mask

    @property
    def is_adaptive(self) -> bool:
        """Check if using adaptive background."""
        return self.adaptive_background


def visualize_adaptive_background(
    video_path: str,
    output_path: str,
    max_frames: int | None = None,
    # Buffer parameters
    buffer_size: int = 10,
    # Adaptive background parameters
    rotation_start_threshold_deg: float = 0.01,
    rotation_stop_threshold_deg: float = 0.005,
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
    - Buffer fill status
    - Feature matches (optional)

    Args:
        video_path: Path to input video
        output_path: Path for output diagnostic video
        max_frames: Maximum frames to process (None for all)
        buffer_size: Number of recent frames to use for background (default 10)
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
    print(
        f"Initializing adaptive background manager for visualization (buffer_size={buffer_size})..."
    )
    manager = AdaptiveBackgroundManager(
        video_path,
        buffer_size=buffer_size,
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

    print(f"Creating adaptive background diagnostic video: {output_path}")
    print(f"  Output size: {out_width}x{out_height}")
    print(
        f"  Rotation thresholds: start={rotation_start_threshold_deg}°, stop={rotation_stop_threshold_deg}°"
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # For feature visualization
    prev_gray = None
    estimator = manager.estimator if show_features else None

    # Colors for states
    STATE_COLORS = {
        RotationState.STATIC: (0, 255, 0),  # Green
        RotationState.ROTATING: (0, 165, 255),  # Orange
        RotationState.TRANSITION: (255, 0, 255),  # Magenta
    }
    ROI_COLOR = (128, 128, 128)  # Gray

    # ROI circle parameters
    roi_center = (width // 2, height // 2)
    roi_radius = min(width, height) // 2

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
        cv2.putText(
            panel1,
            "Original",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw rotation center if available
        if rotation_center_est is not None:
            cx, cy = int(rotation_center_est[0]), int(rotation_center_est[1])
            cv2.drawMarker(panel1, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
            cv2.circle(panel1, (cx, cy), 5, (0, 0, 255), -1)

        # Draw ROI circle on panel 1
        cv2.circle(panel1, roi_center, roi_radius, ROI_COLOR, 1)

        # Panel 2: Current background
        panel2 = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            panel2,
            "Background",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw rotation center on background too
        if rotation_center_est is not None:
            cx, cy = int(rotation_center_est[0]), int(rotation_center_est[1])
            cv2.drawMarker(panel2, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Draw ROI circle on panel 2
        cv2.circle(panel2, roi_center, roi_radius, ROI_COLOR, 1)

        # Panel 3: Difference/mask
        # Show diff with mask overlay
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        panel3 = cv2.cvtColor(diff_normalized, cv2.COLOR_GRAY2BGR)
        # Highlight thresholded regions in red
        panel3[mask > 0] = [0, 0, 255]
        cv2.putText(
            panel3,
            f"Diff (thresh={diff_threshold})",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Draw ROI circle on panel 3
        cv2.circle(panel3, roi_center, roi_radius, ROI_COLOR, 1)

        # Combine top row
        top_row = np.hstack([panel1, panel2, panel3])

        # Create info panel
        info_panel = np.zeros((info_height, out_width, 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)

        # State indicator
        state_color = STATE_COLORS.get(state, (255, 255, 255))
        cv2.rectangle(info_panel, (10, 10), (200, 50), state_color, -1)
        cv2.rectangle(info_panel, (10, 10), (200, 50), (255, 255, 255), 2)
        cv2.putText(
            info_panel,
            state.value.upper(),
            (25, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
        )

        # Frame info
        cv2.putText(
            info_panel,
            f"Frame: {frame_idx}",
            (220, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            info_panel,
            f"/{max_frames}",
            (220, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        # Rotation info
        cv2.putText(
            info_panel,
            f"Angle/frame: {smoothed_angle:+.2f} deg",
            (350, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            info_panel,
            f"Cumulative: {cumulative_rotation:+.1f} deg",
            (350, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )

        # Center estimate
        if rotation_center_est is not None:
            cv2.putText(
                info_panel,
                f"Center: ({rotation_center_est[0]:.0f}, {rotation_center_est[1]:.0f})",
                (350, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )

        # Buffer fill info
        buf_curr, buf_max = manager.get_buffer_fill()
        cv2.putText(
            info_panel,
            f"Buffer: {buf_curr}/{buf_max}",
            (220, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1,
        )

        # State-specific info
        if state == RotationState.ROTATING:
            cv2.putText(
                info_panel,
                f"Low-rot frames: {low_rot_count}/{manager.rotation_stop_frames}",
                (600, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
            # Show center estimation progress
            center_count = manager.get_center_estimate_count()
            center_weight = manager.get_center_estimate_weight()
            cv2.putText(
                info_panel,
                f"Center samples: {center_count} (wt: {center_weight:.1f})",
                (600, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
        elif state == RotationState.TRANSITION:
            curr, total = transition_progress
            cv2.putText(
                info_panel,
                f"Transition: {curr}/{total} frames",
                (600, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
            )

        # Thresholds info
        cv2.putText(
            info_panel,
            f"Start thresh: {rotation_start_threshold_deg} deg",
            (600, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
        )
        cv2.putText(
            info_panel,
            f"Stop thresh: {rotation_stop_threshold_deg} deg",
            (600, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (100, 100, 100),
            1,
        )

        # Mini angle plot
        plot_x_start = 800
        plot_width = out_width - plot_x_start - 20
        plot_height = 80
        plot_y_center = 60

        if len(angle_plot_history) > 1 and plot_width > 0:
            # Draw plot background
            cv2.rectangle(
                info_panel,
                (plot_x_start, plot_y_center - plot_height // 2),
                (plot_x_start + plot_width, plot_y_center + plot_height // 2),
                (60, 60, 60),
                -1,
            )

            # Draw zero line
            cv2.line(
                info_panel,
                (plot_x_start, plot_y_center),
                (plot_x_start + plot_width, plot_y_center),
                (100, 100, 100),
                1,
            )

            # Draw threshold lines
            max_angle_display = 2.0  # degrees
            thresh_y_start = int(
                plot_y_center
                - (rotation_start_threshold_deg / max_angle_display)
                * (plot_height // 2)
            )
            thresh_y_stop = int(
                plot_y_center
                - (rotation_stop_threshold_deg / max_angle_display) * (plot_height // 2)
            )
            cv2.line(
                info_panel,
                (plot_x_start, thresh_y_start),
                (plot_x_start + plot_width, thresh_y_start),
                (0, 100, 0),
                1,
            )
            cv2.line(
                info_panel,
                (plot_x_start, thresh_y_stop),
                (plot_x_start + plot_width, thresh_y_stop),
                (0, 50, 0),
                1,
            )

            # Draw angle history
            points = []
            for i, angle in enumerate(angle_plot_history):
                x = plot_x_start + int(i * plot_width / max_plot_points)
                # Clamp angle for display
                clamped_angle = max(-max_angle_display, min(max_angle_display, angle))
                y = int(
                    plot_y_center
                    - (clamped_angle / max_angle_display) * (plot_height // 2)
                )
                points.append((x, y))

            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(info_panel, points[i], points[i + 1], (0, 255, 255), 1)

            cv2.putText(
                info_panel,
                "Angle",
                (plot_x_start, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (150, 150, 150),
                1,
            )

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
        print(
            f"  Episode {i + 1}: frames {ep.start_frame}-{ep.end_frame}, "
            f"rotation: {ep.total_rotation_deg:.1f} deg, "
            f"center: ({ep.rotation_center[0]:.0f}, {ep.rotation_center[1]:.0f})"
        )
