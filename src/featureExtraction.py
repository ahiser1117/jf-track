import numpy as np
from scipy.ndimage import convolve1d
from dataclasses import dataclass


@dataclass
class FeatureData:
    """
    Data-oriented storage for extracted features.
    All arrays have shape (n_tracks, n_frames) with NaN padding.
    """
    # Smoothed positions
    smooth_x: np.ndarray
    smooth_y: np.ndarray
    
    # Kinematic features
    direction: np.ndarray
    speed: np.ndarray
    ang_speed: np.ndarray
    
    # Behavioral states
    is_reversal: np.ndarray  # bool - True during any reversal event
    is_omega: np.ndarray  # bool - True during omega turn (subset of reversal)
    is_upsilon: np.ndarray  # bool - True during upsilon turn (subset of reversal)
    reversal_category: np.ndarray  # string dtype: 'sRev', 'lRev', 'sRevOmega', etc.


class WormFeatureExtractor:
    def __init__(self, pixel_size_mm: float = 0.1, frame_rate: float = 3.0, body_length_mm: float = 1.0):
        self.pixel_size = pixel_size_mm
        self.frame_rate = frame_rate
        self.mean_body_length_mm = body_length_mm
        
        self.prefs = {
            'SmoothWinSize': 1.0,          # seconds
            'StepSize': 1.0,               # seconds
            'AngSpeedThreshold': 60,       # deg/sec (for Omegas)
            'RevAngSpeedThreshold': 75,    # deg/sec (for Reversals)
            'OmegaEccThresh': 0.875,       # (for Omegas)
            'UpsilonEccThresh': 0.905,     # (for Turns)
            'MinDeltaHeadingUpsilon': 40,  # deg (Critical angle)
            'SmallReversalThreshold': 0.092,
            'LargeReversalThreshold': 0.5,
            'RevOmegaMaxGap': 1.5,         # seconds
        }

    def _smooth_1d(self, data: np.ndarray, win_size_frames: int) -> np.ndarray:
        """Apply boxcar smoothing to 1D array, handling NaNs."""
        if win_size_frames < 1:
            return data.copy()
        
        result = np.full_like(data, np.nan)
        valid = ~np.isnan(data)
        if np.sum(valid) < 2:
            return result
        
        # Get contiguous valid segments and smooth each
        valid_data = data[valid]
        kernel = np.ones(int(win_size_frames)) / int(win_size_frames)
        smoothed = convolve1d(valid_data, kernel, mode='nearest')
        result[valid] = smoothed
        return result

    def _calc_dif_1d(self, data: np.ndarray, step_size_frames: int) -> np.ndarray:
        """Central difference derivative for 1D array, handling NaNs."""
        n = len(data)
        dif = np.full(n, np.nan)
        valid = ~np.isnan(data)
        
        if np.sum(valid) <= 1:
            return dif
        
        step = int(step_size_frames)
        half_step = int(np.ceil(step / 2))
        
        # Work only on valid contiguous data
        valid_indices = np.where(valid)[0]
        valid_data = data[valid]
        n_valid = len(valid_data)
        
        if n_valid <= 1:
            return dif
        
        valid_dif = np.zeros(n_valid)
        if n_valid > step:
            # Central difference; align slice lengths for even/odd step sizes
            start = half_step
            end = start + (n_valid - step)
            valid_dif[start:end] = (valid_data[step:] - valid_data[:-step]) / step
        # Fill edges with nearest forward/backward difference
        valid_dif[:half_step] = valid_data[1] - valid_data[0]
        valid_dif[n_valid - half_step:] = valid_data[-1] - valid_data[-2]
        
        dif[valid_indices] = valid_dif
        return dif

    def _calc_angle_dif_1d(self, angles: np.ndarray, step_size_frames: int) -> np.ndarray:
        """Angular difference for 1D array, handling NaNs."""
        n = len(angles)
        dif = np.full(n, np.nan)
        valid = ~np.isnan(angles)
        
        if np.sum(valid) <= 1:
            return dif
        
        step = int(step_size_frames)
        half_step = int(np.ceil(step / 2))
        
        valid_indices = np.where(valid)[0]
        valid_data = angles[valid]
        n_valid = len(valid_data)
        
        if n_valid <= 1:
            return dif
        
        def get_angle_diff(a1, a2):
            d = a2 - a1
            d = (d + 180) % 360 - 180
            return d
        
        valid_dif = np.zeros(n_valid)
        if n_valid > step:
            delta = get_angle_diff(valid_data[:-step], valid_data[step:])
            start = half_step
            end = start + (n_valid - step)
            valid_dif[start:end] = delta / step
        
        dif[valid_indices] = valid_dif
        return dif

    def _get_events_1d(self, mask: np.ndarray):
        """Find start/end indices of contiguous True regions."""
        padded = np.concatenate(([False], mask, [False]))
        diff = np.diff(padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        return list(zip(starts, ends))

    def extract_features(self, tracking_data) -> FeatureData:
        """
        Extract features from TrackingData object.
        
        Args:
            tracking_data: TrackingData object with (n_tracks, n_frames) arrays
            
        Returns:
            FeatureData object with extracted features
        """
        n_tracks, n_frames = tracking_data.n_tracks, tracking_data.n_frames
        
        # Initialize output arrays
        smooth_x = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        smooth_y = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        direction = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        speed = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        ang_speed = np.full((n_tracks, n_frames), np.nan, dtype=np.float64)
        is_reversal = np.full((n_tracks, n_frames), False, dtype=bool)
        is_omega = np.full((n_tracks, n_frames), False, dtype=bool)
        is_upsilon = np.full((n_tracks, n_frames), False, dtype=bool)
        reversal_category = np.full((n_tracks, n_frames), '', dtype='U20')
        
        win_frames = int(self.prefs['SmoothWinSize'] * self.frame_rate)
        step_frames = int(self.prefs['StepSize'] * self.frame_rate)
        
        print("Extracting features...")
        for track_idx in range(n_tracks):
            # Get this track's data
            x = tracking_data.x[track_idx]
            y = tracking_data.y[track_idx]
            ecc = tracking_data.eccentricity[track_idx]
            
            # Skip if no valid data
            if np.all(np.isnan(x)):
                continue
            
            # Smoothing
            sx = self._smooth_1d(x, win_frames)
            sy = self._smooth_1d(y, win_frames)
            smooth_x[track_idx] = sx
            smooth_y[track_idx] = sy
            
            # Direction & Speed
            xdif = self._calc_dif_1d(sx, step_frames) * self.frame_rate
            ydif = -self._calc_dif_1d(sy, step_frames) * self.frame_rate
            
            direction_rad = np.arctan2(xdif, ydif)
            direction[track_idx] = np.degrees(direction_rad) % 360
            speed[track_idx] = np.sqrt(xdif**2 + ydif**2) * self.pixel_size
            
            # Angular Speed
            ang_speed[track_idx] = self._calc_angle_dif_1d(direction[track_idx], step_frames) * self.frame_rate
            
            valid = ~np.isnan(ang_speed[track_idx])
            
            # Step 1: Detect reversals based on high angular speed
            # A reversal is any high angular speed event
            reversal_mask = valid & (np.abs(ang_speed[track_idx]) > self.prefs['RevAngSpeedThreshold'])
            is_reversal[track_idx] = reversal_mask
            
            # Step 2: Within reversals, detect omega and upsilon turns as subcategories
            # Omega: High AngSpeed + Low Eccentricity (curled body shape)
            omega_mask = reversal_mask & (ecc <= self.prefs['OmegaEccThresh'])
            is_omega[track_idx] = omega_mask
            
            # Upsilon: Reversal with moderate eccentricity (not as curled as omega)
            upsilon_mask = reversal_mask & ~omega_mask & (ecc <= self.prefs['UpsilonEccThresh'])
            is_upsilon[track_idx] = upsilon_mask
            
            # Step 3: Categorize reversal events by size and turn type
            rev_events = self._get_events_1d(reversal_mask)
            
            for r_start, r_end in rev_events:
                if np.isnan(x[r_start]) or np.isnan(x[r_end]):
                    continue
                    
                # Calculate reversal length
                p1 = np.array([x[r_start], y[r_start]])
                p2 = np.array([x[r_end], y[r_end]])
                dist_mm = np.linalg.norm(p2 - p1) * self.pixel_size
                norm_len = dist_mm / self.mean_body_length_mm
                
                # Determine size category
                if norm_len < self.prefs['SmallReversalThreshold']:
                    rev_type = 'pause'  # Too small to be a real reversal
                elif norm_len < self.prefs['LargeReversalThreshold']:
                    rev_type = 'sRev'
                else:
                    rev_type = 'lRev'
                
                if rev_type == 'pause':
                    reversal_category[track_idx, r_start:r_end+1] = 'pause'
                    continue
                
                # Check if this reversal contains omega or upsilon turns
                has_omega = np.any(omega_mask[r_start:r_end+1])
                has_upsilon = np.any(upsilon_mask[r_start:r_end+1])
                
                if has_omega:
                    final_cat = f"{rev_type}_omega"
                elif has_upsilon:
                    final_cat = f"{rev_type}_upsilon"
                else:
                    final_cat = f"{rev_type}_pure"
                
                reversal_category[track_idx, r_start:r_end+1] = final_cat
        
        return FeatureData(
            smooth_x=smooth_x,
            smooth_y=smooth_y,
            direction=direction,
            speed=speed,
            ang_speed=ang_speed,
            is_reversal=is_reversal,
            is_omega=is_omega,
            is_upsilon=is_upsilon,
            reversal_category=reversal_category,
        )
