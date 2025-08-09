#!/usr/bin/env python3
"""
Live dimension extraction using iPhone camera stream.
Uses iPhone's Continuity Camera or HTTP streaming.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
import queue
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

class LiveDimensionExtractor:
    """Real-time furniture dimension extraction from iPhone camera."""
    
    def __init__(self):
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.results_queue = queue.Queue(maxsize=5)
        
        # Processing parameters - focused on main objects
        self.min_area_ratio = 0.03  # Focus on larger objects
        self.aspect_ratio_range = (0.4, 4.0)  # Typical furniture ratios
        self.min_score_threshold = 0.05  # Higher quality threshold
        
        # Enhanced temporal filtering
        self.stable_detection = None
        self.stable_count = 0
        self.stability_threshold = 8
        self.detection_history = []  # Track recent detections
        self.history_size = 15
        self.current_depth_map = None  # Store loaded depth map
        
    def start_camera_stream(self) -> bool:
        """Start camera stream from iPhone."""
        # Skip source 0 (Mac camera), try iPhone sources 1-9
        sources = list(range(1, 10))  # Try sources 1-9 (skip Mac camera)
        
        logging.info("Scanning for iPhone camera sources (skipping Mac camera)...")
        
        for source in sources:
            try:
                logging.info(f"Trying camera source {source}...")
                self.cap = cv2.VideoCapture(source)
                
                if self.cap.isOpened():
                    # Test frame capture first
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        logging.info(f"✅ iPhone camera found on source {source}: {frame.shape}")
                        
                        # Try to set higher resolution (optional)
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                        
                        # Disable auto-rotation
                        self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
                        
                        # Get actual resolution
                        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        logging.info(f"iPhone camera resolution: {actual_w}x{actual_h}")
                        
                        return True
                    else:
                        logging.debug(f"Source {source} opened but no frame")
                else:
                    logging.debug(f"Source {source} failed to open")
                    
                if self.cap:
                    self.cap.release()
                    self.cap = None
                    
            except Exception as e:
                logging.debug(f"Exception on source {source}: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
                
        logging.error("❌ No iPhone camera sources available")
        logging.error("Try:")
        logging.error("1. Connect iPhone via USB and trust computer")
        logging.error("2. Enable Continuity Camera in Settings")
        logging.error("3. Install EpocCam or Camo app")
        return False
    
    def detect_furniture_live(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect furniture in live frame with enhanced stability."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple, effective preprocessing
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive thresholds based on image statistics
        mean_val = np.mean(blurred)
        low_thresh = max(50, int(mean_val * 0.6))
        high_thresh = min(180, int(mean_val * 1.5))
        
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Light morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_final = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours with hierarchy for better filtering
        contours, hierarchy = cv2.findContours(edges_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = int(w * h * self.min_area_ratio)
        
        # Focus on center region to reduce environmental noise
        center_x, center_y = w // 2, h // 2
        center_weight_radius = min(w, h) // 3
        
        best_contour = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            
            # Get contour center
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Distance from image center (favor central objects)
            center_dist = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
            center_weight = max(0.3, 1.0 - (center_dist / center_weight_radius))
            
            # Basic shape analysis
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            area_ratio = area / (w * h)
            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect) if min(w_rect, h_rect) > 0 else 0
            
            # Simple solidity check
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Furniture-like filtering
            if (area_ratio >= self.min_area_ratio and 
                self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1] and
                solidity > 0.5):  # Only very solid shapes
                
                # Simple scoring: size + center position + solidity
                score = area_ratio * center_weight * solidity
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        if best_contour is None:
            return None
            
        # Create result with improved rectangle fitting
        area = cv2.contourArea(best_contour)
        
        # Try to fit a proper rectangle to the contour
        fitted_corners = self._fit_rectangle_to_contour(best_contour)
        
        if fitted_corners is not None:
            box = fitted_corners.astype(np.int32)
            # Calculate dimensions from fitted corners
            rect_w = max(
                np.linalg.norm(fitted_corners[1] - fitted_corners[0]),
                np.linalg.norm(fitted_corners[2] - fitted_corners[3])
            )
            rect_h = max(
                np.linalg.norm(fitted_corners[3] - fitted_corners[0]),
                np.linalg.norm(fitted_corners[2] - fitted_corners[1])
            )
        else:
            # Fallback to minimum area rectangle
            min_rect = cv2.minAreaRect(best_contour)
            box = cv2.boxPoints(min_rect)
            box = np.array(box, dtype=np.int32)
            rect_w, rect_h = min_rect[1]
        
        if rect_w < rect_h:
            rect_w, rect_h = rect_h, rect_w
            
        return {
            'contour': best_contour,
            'box': box,
            'area': area,
            'area_ratio': area / (w * h),
            'pixel_dimensions': (rect_w, rect_h),
            'score': best_score,
            'edges': edges_final
        }
    
    def estimate_dimensions_from_depth(self, detection: Dict[str, Any], depth_map: np.ndarray) -> Optional[Dict[str, float]]:
        """Estimate real-world dimensions using LiDAR depth data."""
        try:
            # Get furniture mask from detection
            h, w = depth_map.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Scale contour to depth map size if needed
            scale_x = w / detection['contour'][:, 0, 0].max() if detection['contour'][:, 0, 0].max() > 0 else 1
            scale_y = h / detection['contour'][:, 0, 1].max() if detection['contour'][:, 0, 1].max() > 0 else 1
            scaled_contour = detection['contour'].copy()
            scaled_contour[:, 0, 0] = (scaled_contour[:, 0, 0] * scale_x).astype(int)
            scaled_contour[:, 0, 1] = (scaled_contour[:, 0, 1] * scale_y).astype(int)
            
            cv2.fillPoly(mask, [scaled_contour], 255)
            
            # Get depth values for furniture region
            furniture_depths = depth_map[mask > 0]
            if len(furniture_depths) == 0:
                return None
            
            # Scale box corners to depth map size
            box = detection['box'].copy().astype(float)
            box[:, 0] *= scale_x
            box[:, 1] *= scale_y
            box = box.astype(int)
            
            # iPhone camera parameters
            focal_length_mm = 26.0
            sensor_width_mm = 7.0
            focal_length_pixels = (focal_length_mm * w) / sensor_width_mm
            
            # Calculate real-world distances between adjacent corners using depth
            real_distances = []
            corner_depths = []
            
            for i in range(4):
                p1, p2 = box[i], box[(i+1)%4]
                
                # Get depth at corners (with bounds checking)
                x1 = int(np.clip(p1[0], 0, w-1))
                y1 = int(np.clip(p1[1], 0, h-1))
                x2 = int(np.clip(p2[0], 0, w-1))
                y2 = int(np.clip(p2[1], 0, h-1))
                
                d1, d2 = depth_map[y1, x1], depth_map[y2, x2]
                corner_depths.extend([d1, d2])
                
                # Use average depth for this edge
                avg_edge_depth = (d1 + d2) / 2
                
                # Convert pixel distance to real-world distance
                pixel_dist = np.linalg.norm(p1 - p2)
                real_dist = (pixel_dist * avg_edge_depth) / focal_length_pixels
                real_distances.append(real_dist)
            
            # Opposite sides should be similar - average them
            width_m = (real_distances[0] + real_distances[2]) / 2
            height_m = (real_distances[1] + real_distances[3]) / 2
            
            # Average distance to furniture
            avg_distance = np.median(furniture_depths)
            
            return {
                'width_m': round(float(width_m), 3),
                'height_m': round(float(height_m), 3),
                'distance_m': round(float(avg_distance), 3),
                'confidence': detection['score']
            }
            
        except Exception as e:
            logging.error(f"Error estimating dimensions from depth: {e}")
            return None
    
    def extract_depth_from_live_photo(self, image_path: str) -> Optional[np.ndarray]:
        """Extract depth map from iPhone Live Photo."""
        try:
            with Image.open(image_path) as img:
                # Handle MPO format (Live Photos)
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    for i in range(img.n_frames):
                        img.seek(i)
                        
                        # Frame 1 is typically the depth map in Live Photos
                        if i == 1:
                            depth_array = np.array(img.convert('L'), dtype=np.float32)
                            
                            # Check if this is actually depth data
                            if np.std(depth_array) < 10:
                                continue
                            
                            # Process depth data (invert and scale)
                            inverted = 255 - depth_array
                            normalized = inverted / 255.0
                            depth_map = 0.2 + normalized * 2.8  # Map to 0.2-3.0m range
                            
                            return depth_map.astype(np.float32)
                            
        except Exception as e:
            logging.error(f"Error extracting depth from {image_path}: {e}")
            
        return None
    
    def _fit_rectangle_to_contour(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """Fit a rectangle to contour using multiple methods for better perspective handling."""
        try:
            # Method 1: Douglas-Peucker approximation for 4-sided shapes
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                return self._order_corners(approx.reshape(4, 2))
            
            # Method 2: Try with more aggressive approximation
            epsilon = 0.03 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx) == 4:
                return self._order_corners(approx.reshape(4, 2))
            
            # Method 3: Convex hull approximation
            hull = cv2.convexHull(contour)
            epsilon = 0.02 * cv2.arcLength(hull, True)
            hull_approx = cv2.approxPolyDP(hull, epsilon, True)
            
            if len(hull_approx) == 4:
                return self._order_corners(hull_approx.reshape(4, 2))
            
            # Method 4: Find corner points using contour analysis
            corners = self._find_corner_points(contour)
            if corners is not None and len(corners) == 4:
                return self._order_corners(corners)
                
        except Exception:
            pass
        
        return None
    
    def _find_corner_points(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """Find corner points by analyzing contour curvature."""
        try:
            # Get contour points
            points = contour.reshape(-1, 2)
            if len(points) < 8:
                return None
            
            # Calculate curvature at each point
            curvatures = []
            window = 5  # Look at 5 points on each side
            
            for i in range(len(points)):
                # Get neighboring points (circular)
                prev_idx = (i - window) % len(points)
                next_idx = (i + window) % len(points)
                
                # Calculate vectors
                v1 = points[prev_idx] - points[i]
                v2 = points[next_idx] - points[i]
                
                # Calculate angle between vectors
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    curvature = np.pi - angle  # Higher for sharper corners
                else:
                    curvature = 0
                
                curvatures.append((curvature, i, points[i]))
            
            # Sort by curvature and take top 4
            curvatures.sort(reverse=True)
            
            # Get the 4 points with highest curvature
            corner_candidates = curvatures[:8]  # Take more candidates
            
            # Filter corners that are too close to each other
            min_distance = 30  # Minimum distance between corners
            selected_corners = []
            
            for curvature, idx, point in corner_candidates:
                if curvature < 0.5:  # Minimum curvature threshold
                    continue
                    
                # Check distance to already selected corners
                too_close = False
                for _, _, selected_point in selected_corners:
                    if np.linalg.norm(point - selected_point) < min_distance:
                        too_close = True
                        break
                
                if not too_close:
                    selected_corners.append((curvature, idx, point))
                    
                if len(selected_corners) == 4:
                    break
            
            if len(selected_corners) == 4:
                return np.array([point for _, _, point in selected_corners])
                
        except Exception:
            pass
        
        return None
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order corners as top-left, top-right, bottom-right, bottom-left."""
        # Sort by y-coordinate
        pts = pts[np.argsort(pts[:, 1])]
        
        # Split into top and bottom pairs
        top = pts[:2]
        bottom = pts[2:]
        
        # Sort top pair by x-coordinate (left to right)
        top = top[np.argsort(top[:, 0])]
        
        # Sort bottom pair by x-coordinate (left to right)
        bottom = bottom[np.argsort(bottom[:, 0])]
        
        # Return in order: top-left, top-right, bottom-right, bottom-left
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def apply_temporal_filter(self, detection: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Enhanced temporal filtering with detection history."""
        # Add to history
        self.detection_history.append(detection)
        if len(self.detection_history) > self.history_size:
            self.detection_history.pop(0)
        
        # If no current detection, try to recover from recent history
        if detection is None:
            # Look for recent valid detections
            recent_valid = [d for d in self.detection_history[-8:] if d is not None and d['score'] > self.min_score_threshold]
            if recent_valid:
                # Use the most recent valid detection with reduced confidence
                last_valid = recent_valid[-1].copy()
                last_valid['score'] *= 0.6  # Reduce confidence for recovered detection
                self.stable_count = max(0, self.stable_count - 1)
                return last_valid
            else:
                self.stable_count = 0
                self.stable_detection = None
                return None
        
        # Current detection exists
        current_dims = detection['pixel_dimensions']
        
        if self.stable_detection is not None:
            stable_dims = self.stable_detection['pixel_dimensions']
            
            # More generous similarity check (25% tolerance)
            w_diff = abs(current_dims[0] - stable_dims[0]) / stable_dims[0]
            h_diff = abs(current_dims[1] - stable_dims[1]) / stable_dims[1]
            
            if w_diff < 0.25 and h_diff < 0.25:
                self.stable_count += 1
                # Blend current with stable detection for smoother transitions
                if self.stable_count >= 3:
                    alpha = 0.8  # Weight towards current detection
                    blended_dims = (
                        alpha * current_dims[0] + (1-alpha) * stable_dims[0],
                        alpha * current_dims[1] + (1-alpha) * stable_dims[1]
                    )
                    detection['pixel_dimensions'] = blended_dims
            else:
                self.stable_count = max(1, self.stable_count - 1)  # Gradual decay
                if self.stable_count <= 2:
                    self.stable_detection = detection
        else:
            self.stable_detection = detection
            self.stable_count = 1
        
        return detection
    
    def draw_overlay(self, frame: np.ndarray, detection: Optional[Dict[str, Any]], 
                    dimensions: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Draw detection overlay on frame."""
        overlay = frame.copy()
        
        if detection is None:
            # Show "searching" indicator
            cv2.putText(overlay, "Searching for furniture...", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            return overlay
        
        # Draw contour
        cv2.drawContours(overlay, [detection['contour']], -1, (0, 255, 0), 3)
        
        # Draw fitted rectangle
        cv2.drawContours(overlay, [detection['box']], -1, (255, 0, 0), 3)
        
        # Add corner markers with better visibility
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Different colors for each corner
        for i, corner in enumerate(detection['box']):
            cv2.circle(overlay, tuple(corner), 10, corner_colors[i], -1)
            cv2.circle(overlay, tuple(corner), 12, (255, 255, 255), 2)  # White border
            cv2.putText(overlay, str(i+1), tuple(corner + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Add detection info
        info_y = 30
        info_texts = [
            f"Area: {detection['area_ratio']*100:.1f}% of image",
            f"Score: {detection['score']:.3f}",
            f"Pixels: {detection['pixel_dimensions'][0]:.0f} x {detection['pixel_dimensions'][1]:.0f}",
            f"Stable: {self.stable_count}/{self.stability_threshold}",
            f"History: {len([d for d in self.detection_history if d is not None])}/{len(self.detection_history)}"
        ]
        
        if dimensions:
            info_texts.extend([
                f"Width: {dimensions['width_m']:.3f}m (LiDAR)",
                f"Height: {dimensions['height_m']:.3f}m (LiDAR)",
                f"Distance: {dimensions['distance_m']:.2f}m (LiDAR)",
                f"Confidence: {dimensions['confidence']:.2f}"
            ])
        else:
            info_texts.append("Dimensions: Need Live Photo depth data")
        
        for text in info_texts:
            cv2.putText(overlay, text, (20, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            info_y += 30
        
        # Stability indicator
        stability_color = (0, 255, 0) if self.stable_count >= self.stability_threshold else (0, 255, 255)
        cv2.circle(overlay, (frame.shape[1] - 30, 30), 15, stability_color, -1)
        
        return overlay
    
    def run_live_detection(self):
        """Main live detection loop with depth-based measurements."""
        if not self.start_camera_stream():
            return
            
        logging.info("Starting live furniture detection with LiDAR depth...")
        logging.info("Press 'q' to quit, 's' to save current detection, 'c' to capture Live Photo for depth")
        
        self.running = True
        current_depth_map = None
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Fix orientation - rotate if needed
                h, w = frame.shape[:2]
                if h > w:  # Portrait mode, rotate to landscape
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                # Detect furniture
                detection = self.detect_furniture_live(frame)
                self.apply_temporal_filter(detection)
                
                # Estimate dimensions using depth if available
                dimensions = None
                if detection and detection['score'] > self.min_score_threshold:
                    if current_depth_map is not None:
                        # Resize depth map to match frame if needed
                        if current_depth_map.shape[:2] != frame.shape[:2]:
                            current_depth_map = cv2.resize(current_depth_map, (frame.shape[1], frame.shape[0]))
                        dimensions = self.estimate_dimensions_from_depth(detection, current_depth_map)
                
                # Draw overlay
                display_frame = self.draw_overlay(frame, detection, dimensions)
                
                # Add depth status
                depth_status = "Depth: Available" if current_depth_map is not None else "Depth: Capture Live Photo (press 'c')"
                cv2.putText(display_frame, depth_status, (20, display_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Live Furniture Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and dimensions:
                    # Save current detection with depth-based measurements
                    timestamp = int(time.time())
                    cv2.imwrite(f'live_detection_{timestamp}.jpg', display_frame)
                    logging.info(f"Saved detection with depth: {dimensions}")
                elif key == ord('c'):
                    # Prompt user to capture Live Photo
                    logging.info("Please capture a Live Photo of the furniture and provide the file path...")
                    
        except KeyboardInterrupt:
            logging.info("Interrupted by user")
        finally:
            self.running = False
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
    
    def load_depth_from_file(self, heic_path: str) -> bool:
        """Load depth map from HEIC file for live measurements."""
        depth_map = self.extract_depth_from_live_photo(heic_path)
        if depth_map is not None:
            self.current_depth_map = depth_map
            logging.info(f"Loaded depth map from {heic_path}")
            return True
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live furniture dimension extraction with LiDAR depth')
    parser.add_argument('--depth-file', type=str, help='HEIC file with LiDAR depth data')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    
    extractor = LiveDimensionExtractor()
    
    # Load depth map if provided
    if args.depth_file:
        if extractor.load_depth_from_file(args.depth_file):
            logging.info("Depth data loaded successfully")
        else:
            logging.warning("Failed to load depth data, using live detection only")
    
    extractor.run_live_detection()

if __name__ == '__main__':
    main()
