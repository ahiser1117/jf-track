import cv2
import numpy as np
from typing import Tuple, List, Optional, Union


def create_circular_roi_mask(height: int, width: int, center: Optional[Tuple[float, float]] = None, 
                           radius: Optional[float] = None) -> np.ndarray:
    """
    Create a circular ROI mask.
    
    Args:
        height: Frame height in pixels
        width: Frame width in pixels
        center: (x, y) center coordinates (default: frame center)
        radius: Radius in pixels (default: min(width, height)/2)
    
    Returns:
        Binary mask with shape (height, width)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(width, height) // 2
    
    cv2.circle(mask, center, int(radius), 255, -1)
    return mask


def create_polygon_roi_mask(height: int, width: int, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Create a polygon ROI mask from vertex points.
    
    Args:
        height: Frame height in pixels
        width: Frame width in pixels
        points: List of (x, y) polygon vertices
    
    Returns:
        Binary mask with shape (height, width)
    """
    if len(points) < 3:
        raise ValueError("Polygon must have at least 3 points")
    
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Convert points to numpy array and reshape for fillPoly
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    
    cv2.fillPoly(mask, [pts], 255)
    return mask


def create_auto_roi_mask(height: int, width: int) -> np.ndarray:
    """
    Create automatic circular ROI mask (centered, max possible radius).
    
    Args:
        height: Frame height in pixels
        width: Frame width in pixels
    
    Returns:
        Binary mask with shape (height, width)
    """
    return create_circular_roi_mask(height, width)


def create_roi_mask(height: int, width: int, roi_mode: str = "auto",
                  center: Optional[Tuple[float, float]] = None,
                  radius: Optional[float] = None,
                  points: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    Factory function to create ROI masks based on mode and parameters.
    
    Args:
        height: Frame height in pixels
        width: Frame width in pixels
        roi_mode: "auto", "circle", or "polygon"
        center: Circle center (for circle mode)
        radius: Circle radius (for circle mode)
        points: Polygon vertices (for polygon mode)
    
    Returns:
        Binary mask with shape (height, width)
    
    Raises:
        ValueError: If invalid mode or insufficient parameters provided
    """
    if roi_mode == "auto":
        return create_auto_roi_mask(height, width)
    elif roi_mode == "circle":
        if center is None or radius is None:
            raise ValueError("Circle mode requires center and radius parameters")
        return create_circular_roi_mask(height, width, center, radius)
    elif roi_mode == "polygon":
        if points is None or len(points) < 3:
            raise ValueError("Polygon mode requires at least 3 points")
        return create_polygon_roi_mask(height, width, points)
    else:
        raise ValueError(f"Invalid ROI mode: {roi_mode}")


class ROISelector:
    """Interactive ROI selection using OpenCV."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get frame dimensions
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Read first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video")
        self.cap.release()
        
        # Selection state
        self.roi_mode = "auto"
        self.circle_center = None
        self.circle_radius = None
        self.polygon_points = []
        self.drawing = False
        self.selection_complete = False
        
        # Window setup
        cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("ROI Selection", self.mouse_callback)
        
        # Instructions
        self.instructions = [
            "ROI Selection Mode: Auto (press 'c' for circle, 'p' for polygon)",
            "Circle: Click center, drag for radius, release to finish",
            "Polygon: Click vertices, press 'f' to finish polygon",
            "Press 'a' for auto ROI, 'r' to reset, 'q' to finish selection"
        ]
    
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for interactive selection."""
        if self.roi_mode == "circle":
            self._circle_mouse_callback(event, x, y, flags)
        elif self.roi_mode == "polygon":
            self._polygon_mouse_callback(event, x, y)
    
    def _circle_mouse_callback(self, event, x, y, flags):
        """Handle mouse events for circle selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.circle_center = (x, y)
            self.circle_radius = 0
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.circle_radius = np.sqrt((x - self.circle_center[0])**2 + (y - self.circle_center[1])**2)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.selection_complete = True
    
    def _polygon_mouse_callback(self, event, x, y):
        """Handle mouse events for polygon selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.polygon_points.append((x, y))
    
    def draw_overlay(self) -> np.ndarray:
        """Draw selection overlay on frame."""
        overlay = self.frame.copy()
        
        # Draw ROI preview
        if self.roi_mode == "circle":
            if self.circle_center is not None:
                cv2.circle(overlay, self.circle_center, int(self.circle_radius or 0), (0, 255, 0), 2)
                if self.circle_center:
                    cv2.circle(overlay, self.circle_center, 3, (0, 0, 255), -1)
        
        elif self.roi_mode == "polygon":
            # Draw polygon edges
            if len(self.polygon_points) >= 2:
                pts = np.array(self.polygon_points, dtype=np.int32)
                cv2.polylines(overlay, [pts], False, (0, 255, 0), 2)
            
            # Draw vertices
            for point in self.polygon_points:
                cv2.circle(overlay, point, 3, (0, 0, 255), -1)
        
        # Draw instructions
        y_offset = 30
        mode_text = f"Current mode: {self.roi_mode.upper()}"
        cv2.putText(overlay, mode_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        for instruction in self.instructions[1:]:
            cv2.putText(overlay, instruction, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        return overlay
    
    def run_selection(self) -> dict:
        """Run interactive ROI selection interface."""
        print("Starting interactive ROI selection...")
        
        while True:
            display_frame = self.draw_overlay()
            cv2.imshow("ROI Selection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('a'):  # Auto mode
                self.roi_mode = "auto"
                self.selection_complete = True
                print("Selected automatic ROI mode")
            elif key == ord('c'):  # Circle mode
                self.roi_mode = "circle"
                self.circle_center = None
                self.circle_radius = None
                self.polygon_points = []
                self.selection_complete = False
                print("Switched to circle selection mode")
            elif key == ord('p'):  # Polygon mode
                self.roi_mode = "polygon"
                self.circle_center = None
                self.circle_radius = None
                self.polygon_points = []
                self.selection_complete = False
                print("Switched to polygon selection mode")
            elif key == ord('r'):  # Reset
                self.circle_center = None
                self.circle_radius = None
                self.polygon_points = []
                self.selection_complete = False
                print("Reset selection")
            elif key == ord('f'):  # Finish polygon
                if self.roi_mode == "polygon" and len(self.polygon_points) >= 3:
                    self.selection_complete = True
                    print(f"Polygon completed with {len(self.polygon_points)} vertices")
        
        cv2.destroyAllWindows()
        
        # Return ROI configuration
        roi_config = {
            "mode": self.roi_mode,
            "height": self.height,
            "width": self.width
        }
        
        if self.roi_mode == "circle" and self.circle_center and self.circle_radius:
            roi_config.update({
                "center": self.circle_center,
                "radius": self.circle_radius
            })
        elif self.roi_mode == "polygon" and len(self.polygon_points) >= 3:
            roi_config.update({
                "points": self.polygon_points
            })
        
        return roi_config


def run_interactive_roi_selection(video_path: str) -> dict:
    """Convenience function to run interactive ROI selection."""
    selector = ROISelector(video_path)
    return selector.run_selection()


def create_mask_from_config(roi_config: dict) -> np.ndarray:
    """Create ROI mask from ROI configuration dictionary."""
    height = roi_config["height"]
    width = roi_config["width"]
    mode = roi_config["mode"]
    
    if mode == "auto":
        return create_auto_roi_mask(height, width)
    elif mode == "circle":
        return create_circular_roi_mask(height, width, roi_config["center"], roi_config["radius"])
    elif mode == "polygon":
        return create_polygon_roi_mask(height, width, roi_config["points"])
    else:
        raise ValueError(f"Unknown ROI mode: {mode}")


# Utility functions for validation
def validate_roi_config(roi_config: dict) -> bool:
    """Validate ROI configuration dictionary."""
    required_keys = ["mode", "height", "width"]
    if not all(key in roi_config for key in required_keys):
        return False
    
    mode = roi_config["mode"]
    if mode == "circle":
        return "center" in roi_config and "radius" in roi_config
    elif mode == "polygon":
        return "points" in roi_config and len(roi_config["points"]) >= 3
    elif mode == "auto":
        return True
    else:
        return False


def get_roi_statistics(roi_mask: np.ndarray) -> dict:
    """Calculate statistics for an ROI mask."""
    total_pixels = roi_mask.size
    roi_pixels = np.count_nonzero(roi_mask)
    roi_area_ratio = roi_pixels / total_pixels
    
    # Find ROI bounding box
    rows, cols = np.where(roi_mask > 0)
    if len(rows) > 0 and len(cols) > 0:
        bbox = {
            "min_row": int(rows.min()),
            "max_row": int(rows.max()),
            "min_col": int(cols.min()),
            "max_col": int(cols.max())
        }
        bbox_width = bbox["max_col"] - bbox["min_col"] + 1
        bbox_height = bbox["max_row"] - bbox["min_row"] + 1
        bbox_area = bbox_width * bbox_height
    else:
        bbox = None
        bbox_area = 0
    
    return {
        "total_pixels": int(total_pixels),
        "roi_pixels": int(roi_pixels),
        "roi_area_ratio": float(roi_area_ratio),
        "bbox": bbox,
        "bbox_area": int(bbox_area)
    }