import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from skimage.measure import label, regionprops
from .tracker import TrackingParameters


@dataclass
class SampledFeature:
    """Data class for sampled feature properties."""
    object_type: str
    centroid: Tuple[float, float]
    area: float
    aspect_ratio: float
    eccentricity: float
    solidity: float
    major_axis_length_mm: float
    minor_axis_length_mm: float


class InteractiveFeatureSampler:
    """OpenCV-based interactive feature sampling for size/shape calibration."""
    
    def __init__(self, video_path: str, pixel_size_mm: float = 0.01):
        self.video_path = video_path
        self.pixel_size_mm = pixel_size_mm
        self.sampled_features: List[SampledFeature] = []
        self.current_frame = None
        self.binary_mask = None
        self.roi_mask = None
        
        # UI state
        self.instructions = [
            "Interactive Feature Sampling for Size/Shape Calibration",
            "",
            "Instructions:",
            "1. Click on representative objects of each type",
            "2. Press 'm' for next mouth sample, 'g' for gonad, 't' for tentacle bulb",
            "3. Press 'u' to undo last sample, 'c' to clear all samples",
            "4. Press 'b' to cycle background subtraction threshold",
            "5. Press 'n' for next frame, 'p' for previous frame",
            "6. Press 'q' to finish sampling",
            "",
            "Current threshold: 10"
        ]
        
    def load_frame(self, frame_idx: int = 0) -> bool:
        """Load a specific frame from the video."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.current_frame = frame
            self.frame_idx = frame_idx
            return True
        return False
    
    def apply_background_subtraction(self, threshold: int = 10) -> np.ndarray:
        """Apply simple background subtraction using first frame as background."""
        if self.current_frame is None:
            return np.zeros((1, 1), dtype=np.uint8)
            
        # For simplicity, use a blurred version as "background"
        # In a real implementation, this could be more sophisticated
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (51, 51), 0)
        
        diff = cv2.absdiff(gray, blurred)
        _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def analyze_clicked_region(self, x: int, y: int) -> Optional[Dict]:
        """Analyze the region around a clicked point to extract object properties."""
        if self.binary_mask is None:
            return None
            
        # Find the object at the clicked location
        label_img = label(self.binary_mask)
        
        # Get the label at the clicked point
        if (0 <= y < label_img.shape[0] and 0 <= x < label_img.shape[1]):
            obj_label = label_img[y, x]
            
            if obj_label == 0:  # Background
                return None
            
            # Find regionprops for this label
            props = regionprops(label_img)
            for prop in props:
                if prop.label == obj_label:
                    # Extract comprehensive properties
                    y_centroid, x_centroid = prop.centroid
                    major_axis = prop.axis_major_length
                    minor_axis = prop.minor_axis_length if prop.minor_axis_length > 0 else 1.0
                    
                    return {
                        "centroid": (x_centroid, y_centroid),
                        "area": prop.area,
                        "major_axis_length_mm": major_axis * self.pixel_size_mm,
                        "minor_axis_length_mm": minor_axis * self.pixel_size_mm,
                        "aspect_ratio": major_axis / minor_axis,
                        "eccentricity": prop.eccentricity,
                        "solidity": prop.solidity,
                        "orientation_deg": np.degrees(prop.orientation),
                        "bounding_box": prop.bbox,
                    }
        
        return None
    
    def add_sample(self, object_type: str, properties: Dict):
        """Add a sampled feature to the collection."""
        feature = SampledFeature(
            object_type=object_type,
            centroid=properties["centroid"],
            area=properties["area"],
            aspect_ratio=properties["aspect_ratio"],
            eccentricity=properties["eccentricity"],
            solidity=properties["solidity"],
            major_axis_length_mm=properties["major_axis_length_mm"],
            minor_axis_length_mm=properties["minor_axis_length_mm"],
        )
        self.sampled_features.append(feature)
        print(f"Added {object_type} sample: area={properties['area']:.0f}, "
              f"aspect_ratio={properties['aspect_ratio']:.2f}")
    
    def suggest_parameters(self) -> Dict[str, Dict]:
        """Suggest tracking parameters from sampled features."""
        if not self.sampled_features:
            return {}
        
        # Group samples by object type
        by_type = {}
        for feature in self.sampled_features:
            if feature.object_type not in by_type:
                by_type[feature.object_type] = []
            by_type[feature.object_type].append(feature)
        
        suggestions = {}
        
        for obj_type, features in by_type.items():
            if len(features) == 0:
                continue
            
            # Calculate statistics
            areas = [f.area for f in features]
            aspect_ratios = [f.aspect_ratio for f in features]
            eccentricities = [f.eccentricity for f in features]
            solidities = [f.solidity for f in features]
            
            # Suggest parameter ranges with 20% tolerance
            area_mean = np.mean(areas)
            area_std = np.std(areas)
            area_min = max(5, area_mean - 0.5 * area_std)
            area_max = area_mean + 0.5 * area_std
            
            aspect_ratio_mean = np.mean(aspect_ratios)
            aspect_ratio_std = np.std(aspect_ratios)
            
            eccentricity_mean = np.mean(eccentricities)
            eccentricity_std = np.std(eccentricities)
            
            solidity_mean = np.mean(solidities)
            solidity_std = np.std(solidities)
            
            # Create configuration
            config = {
                "min_area": int(area_min),
                "max_area": int(area_max),
                "aspect_ratio_min": max(1.0, aspect_ratio_mean - 0.3 * aspect_ratio_std),
                "aspect_ratio_max": aspect_ratio_mean + 0.3 * aspect_ratio_std,
                "eccentricity_min": max(0.0, eccentricity_mean - 0.3 * eccentricity_std),
                "eccentricity_max": min(1.0, eccentricity_mean + 0.3 * eccentricity_std),
                "solidity_min": max(0.0, solidity_mean - 0.3 * solidity_std),
                "solidity_max": min(1.0, solidity_mean + 0.3 * solidity_std),
            }
            
            # Special adjustments based on object type
            if obj_type == "gonad":
                config["aspect_ratio_min"] = max(1.5, config["aspect_ratio_min"])  # Oblong
                config["eccentricity_min"] = max(0.7, config["eccentricity_min"])   # Elongated
            elif obj_type == "tentacle_bulb":
                config["aspect_ratio_max"] = min(1.5, config["aspect_ratio_max"])  # Round
                config["eccentricity_max"] = min(0.7, config["eccentricity_max"])   # Circular
            elif obj_type == "mouth":
                # Mouth can be variable, so use wider ranges
                config["min_area"] = int(area_min * 0.8)
                config["max_area"] = int(area_max * 1.2)
            
            suggestions[obj_type] = config
        
        return suggestions
    
    def update_tracking_parameters(self, params: TrackingParameters) -> TrackingParameters:
        """Update tracking parameters with suggested values from sampled features."""
        suggestions = self.suggest_parameters()
        
        for obj_type, config in suggestions.items():
            if obj_type in params.object_types:
                params.object_types[obj_type].update(config)
        
        return params
    
    def draw_overlay(self, frame: np.ndarray, threshold: int = 10) -> np.ndarray:
        """Draw UI overlay on the frame."""
        overlay = frame.copy()
        
        # Draw binary mask as overlay
        if self.binary_mask is not None:
            # Convert binary mask to 3-channel
            mask_colored = cv2.cvtColor(self.binary_mask, cv2.COLOR_GRAY2BGR)
            mask_colored[np.where((mask_colored == [255, 255, 255]))] = [0, 255, 0]  # Green for objects
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        # Draw sampled points
        for feature in self.sampled_features:
            x, y = int(feature.centroid[0]), int(feature.centroid[1])
            color = {"mouth": (0, 0, 255), "gonad": (0, 255, 255), "tentacle_bulb": (255, 0, 0)}.get(feature.object_type, (128, 128, 128))
            cv2.circle(overlay, (x, y), 5, color, -1)
            cv2.circle(overlay, (x, y), 7, color, 2)
            
            # Label
            cv2.putText(overlay, feature.object_type[0].upper(), (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw instructions
        y_offset = 30
        for line in self.instructions:
            if "threshold" in line:
                line = f"Current threshold: {threshold}"
            cv2.putText(overlay, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Draw sample count
        count_text = f"Samples: {len(self.sampled_features)} total"
        by_type = {}
        for feature in self.sampled_features:
            by_type[feature.object_type] = by_type.get(feature.object_type, 0) + 1
        
        for obj_type, count in by_type.items():
            count_text += f", {obj_type}: {count}"
        
        cv2.putText(overlay, count_text, (10, overlay.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        return overlay
    
    def run_interactive_sampling(self) -> TrackingParameters:
        """Run the interactive sampling interface."""
        print("Starting interactive feature sampling...")
        
        # Load first frame
        if not self.load_frame(0):
            raise ValueError(f"Could not load video: {self.video_path}")
        
        cv2.namedWindow("Feature Sampling", cv2.WINDOW_NORMAL)
        
        threshold = 10
        current_object_type = "mouth"
        
        while True:
            # Update binary mask
            self.binary_mask = self.apply_background_subtraction(threshold)
            
            # Draw overlay
            display_frame = self.draw_overlay(self.current_frame, threshold)
            
            cv2.imshow("Feature Sampling", display_frame)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('m'):  # Mouth sampling mode
                current_object_type = "mouth"
                print(f"Switched to mouth sampling mode")
            elif key == ord('g'):  # Gonad sampling mode
                current_object_type = "gonad"
                print(f"Switched to gonad sampling mode")
            elif key == ord('t'):  # Tentacle bulb sampling mode
                current_object_type = "tentacle_bulb"
                print(f"Switched to tentacle bulb sampling mode")
            elif key == ord('u'):  # Undo last sample
                if self.sampled_features:
                    removed = self.sampled_features.pop()
                    print(f"Removed last {removed.object_type} sample")
            elif key == ord('c'):  # Clear all samples
                count = len(self.sampled_features)
                self.sampled_features.clear()
                print(f"Cleared {count} samples")
            elif key == ord('b'):  # Cycle threshold
                thresholds = [5, 10, 15, 20, 25, 30]
                current_idx = thresholds.index(threshold) if threshold in thresholds else 0
                threshold = thresholds[(current_idx + 1) % len(thresholds)]
                print(f"Threshold changed to: {threshold}")
            elif key == ord('n'):  # Next frame
                self.load_frame(self.frame_idx + 1)
                print(f"Frame {self.frame_idx + 1}")
            elif key == ord('p'):  # Previous frame
                if self.frame_idx > 0:
                    self.load_frame(self.frame_idx - 1)
                    print(f"Frame {self.frame_idx - 1}")
            elif key == ord(' ') or key == 13:  # Space or Enter - sample at mouse position
                # Get mouse position (approximate by waiting for mouse click)
                # For simplicity, we'll use the next click as sample location
                print(f"Click on a {current_object_type} to sample...")
                
                # Wait for mouse click
                while True:
                    display_frame = self.draw_overlay(self.current_frame, threshold)
                    cv2.imshow("Feature Sampling", display_frame)
                    mouse_key = cv2.waitKey(1) & 0xFF
                    
                    if mouse_key == ord('q'):  # Quit
                        break
                    elif mouse_key == 27:  # ESC - cancel sampling
                        print("Sampling cancelled")
                        break
                    
                    # Check for mouse events (need to use callback in real implementation)
                    # For now, we'll skip mouse detection complexity
                    # In a full implementation, you'd set up a mouse callback
                    
                    break
                
                # For demonstration, skip actual mouse detection
                # In real implementation, you'd get mouse position and call analyze_clicked_region()
        
        cv2.destroyAllWindows()
        
        # Create and update parameters
        params = TrackingParameters()
        
        # Update parameters based on samples
        if self.sampled_features:
            params = self.update_tracking_parameters(params)
            
            # Print suggestions
            suggestions = self.suggest_parameters()
            print("\nSuggested Parameters:")
            for obj_type, config in suggestions.items():
                print(f"\n{obj_type.upper()}:")
                print(f"  Area range: {config['min_area']}-{config['max_area']}")
                if config['aspect_ratio_min'] is not None:
                    print(f"  Aspect ratio: {config['aspect_ratio_min']:.2f}-{config['aspect_ratio_max']:.2f}")
                if config['eccentricity_min'] is not None:
                    print(f"  Eccentricity: {config['eccentricity_min']:.2f}-{config['eccentricity_max']:.2f}")
        
        return params


def run_interactive_feature_sampling(video_path: str, pixel_size_mm: float = 0.01) -> TrackingParameters:
    """Convenience function to run interactive feature sampling."""
    sampler = InteractiveFeatureSampler(video_path, pixel_size_mm)
    return sampler.run_interactive_sampling()