import os
import cv2
import json
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
import struct

from .config import Config

register_heif_opener()

class DimensionExtractor:
    """
    Extracts furniture dimensions from iPhone 15 Pro HEIC images with LiDAR depth data.
    
    This module leverages the depth information embedded in HEIC files from iPhone 15 Pro
    to calculate real-world dimensions of rectangular furniture pieces.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.depth_dir = os.path.join(config.processing_dir, 'depth_maps')
        self.dimensions_dir = os.path.join(config.results_dir, 'dimensions')
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.dimensions_dir, exist_ok=True)
        
    def extract_dimensions(self, heic_files: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Extract dimensions from HEIC files with LiDAR depth data.
        
        Args:
            heic_files: List of HEIC file paths
            
        Returns:
            Dictionary mapping filename to dimensions (width, height, depth in meters)
        """
        results = {}
        
        for heic_file in heic_files:
            logging.info(f"Processing {heic_file} for dimension extraction...")
            
            # Extract depth map from HEIC
            depth_map = self._extract_depth_from_heic(heic_file)
            if depth_map is None:
                logging.warning(f"No depth data found in {heic_file}")
                continue
                
            # Load RGB image
            rgb_image = self._load_rgb_from_heic(heic_file)
            if rgb_image is None:
                continue
                
            # Detect rectangular furniture using depth data
            furniture_mask = self._detect_furniture_rectangle(rgb_image, depth_map)
            if furniture_mask is None:
                logging.warning(f"No rectangular furniture detected in {heic_file}")
                continue
                
            # Calculate dimensions using depth data
            dimensions = self._calculate_dimensions(depth_map, furniture_mask)
            if dimensions:
                results[heic_file] = dimensions
                self._save_results(heic_file, rgb_image, depth_map, furniture_mask, dimensions)
                
        return results
    
    def _extract_depth_from_heic(self, heic_path: str) -> Optional[np.ndarray]:
        """Extract depth map from iPhone Live Photo (MPO/HEIC format)."""
        try:
            with Image.open(heic_path) as img:
                logging.info(f"Image format: {img.format}, frames: {getattr(img, 'n_frames', 1)}")
                
                # Handle MPO format (Live Photos)
                if hasattr(img, 'n_frames') and img.n_frames > 1:
                    for i in range(img.n_frames):
                        img.seek(i)
                        logging.debug(f"Frame {i}: mode={img.mode}, size={img.size}")
                        
                        # Frame 1 is typically the depth map in Live Photos
                        if i == 1:
                            depth_array = np.array(img.convert('L'), dtype=np.float32)
                            
                            # Save raw depth frame for debugging
                            base_name = os.path.splitext(os.path.basename(heic_path))[0]
                            cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_raw_frame1.jpg'), depth_array.astype(np.uint8))
                            logging.info(f"Saved raw depth frame: {base_name}_raw_frame1.jpg")
                            
                            # Check if this is actually depth data
                            if np.std(depth_array) < 10:  # Very low variance = not depth
                                logging.warning(f"Frame 1 appears to be uniform (std={np.std(depth_array):.2f}), not depth data")
                                continue
                                
                            depth_map = self._process_depth_data(depth_array)
                            
                            # Save processed depth for debugging
                            depth_vis = cv2.applyColorMap((depth_map / depth_map.max() * 255).astype(np.uint8), cv2.COLORMAP_JET)
                            cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_processed_depth.jpg'), depth_vis)
                            logging.info(f"Processed depth range: {depth_map.min():.3f} - {depth_map.max():.3f}m")
                            
                            return depth_map
                            
        except Exception as e:
            logging.error(f"Error extracting depth from {heic_path}: {e}")
            
        return None
    
    def _process_depth_data(self, depth_array: np.ndarray) -> np.ndarray:
        """Process depth data - save raw first, then process."""
        logging.info(f"Raw depth stats: min={depth_array.min()}, max={depth_array.max()}, std={np.std(depth_array):.2f}")
        
        # INVERT the depth values - darker = closer in iPhone depth maps
        inverted = 255 - depth_array
        
        # Map inverted values to reasonable depth range (0.2m to 3.0m)
        normalized = inverted / 255.0
        depth_map = 0.2 + normalized * 2.8  # Map to 0.2-3.0m range
        
        logging.info(f"Processed depth stats: min={depth_map.min():.3f}m, max={depth_map.max():.3f}m")
        return depth_map.astype(np.float32)
    
    def _load_rgb_from_heic(self, heic_path: str) -> Optional[np.ndarray]:
        """Load RGB image from HEIC file."""
        try:
            with Image.open(heic_path) as img:
                rgb_array = np.array(img.convert('RGB'))
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logging.error(f"Error loading RGB from {heic_path}: {e}")
            return None
    
    def _detect_furniture_rectangle(self, image: np.ndarray, depth_map: np.ndarray = None) -> Optional[np.ndarray]:
        """Detect furniture using robust RGB-based rectangle detection."""
        base_name = "debug_furniture_detection"
        h, w = image.shape[:2]
        
        # RGB-based rectangle detection (more robust than depth for shadows)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Adaptive preprocessing
        mean_val = np.mean(gray)
        blur_size = 5 if np.std(gray) > 30 else 3
        blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
        
        # Adaptive Canny thresholds
        low_thresh = max(30, int(mean_val * 0.5))
        high_thresh = min(150, int(mean_val * 1.2))
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges_final = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = int(w * h * 0.01)  # Lowered to 1% of image
        
        # Save debug images
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_edges.jpg'), edges_final)
        
        # Debug: save all contours
        all_contours_vis = image.copy()
        cv2.drawContours(all_contours_vis, contours, -1, (0, 255, 0), 2)
        cv2.putText(all_contours_vis, f"Total contours: {len(contours)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_all_contours.jpg'), all_contours_vis)
        
        # Find best rectangular contour
        best_contour = None
        best_score = 0
        candidates = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            # Always try minimum area rectangle (more robust)
            min_rect = cv2.minAreaRect(contour)
            rect_w, rect_h = min_rect[1]
            
            if rect_w < rect_h:
                rect_w, rect_h = rect_h, rect_w
            
            # Score based on area and aspect ratio
            area_ratio = area / (w * h)
            aspect_ratio = rect_w / rect_h if rect_h > 0 else 0
            
            # Much more flexible criteria
            if area_ratio >= 0.01 and 0.2 <= aspect_ratio <= 10.0:  # Very flexible
                score = area_ratio  # Larger objects get higher scores
                candidates.append({
                    'contour': contour,
                    'area': area,
                    'area_ratio': area_ratio,
                    'aspect_ratio': aspect_ratio,
                    'score': score
                })
                
                if score > best_score:
                    best_score = score
                    best_contour = contour
        
        # Log candidates for debugging
        logging.info(f"Found {len(candidates)} candidates from {len(contours)} contours")
        for i, cand in enumerate(candidates[:5]):
            logging.info(f"  Candidate {i}: area={cand['area']:.0f} ({cand['area_ratio']*100:.1f}%), aspect={cand['aspect_ratio']:.2f}, score={cand['score']:.3f}")
        
        if best_contour is None:
            logging.warning("No suitable rectangular furniture found")
            # If no candidates, just use the largest contour
            if contours:
                largest = max(contours, key=cv2.contourArea)
                largest_area = cv2.contourArea(largest)
                if largest_area > min_area:
                    logging.info(f"Using largest contour as fallback: area={largest_area:.0f}")
                    best_contour = largest
                    best_score = largest_area / (w * h)
            
            if best_contour is None:
                # Save debug visualization
                debug_img = image.copy()
                cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
                cv2.putText(debug_img, f"Found {len(contours)} contours, none suitable", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_failed.jpg'), debug_img)
                return None
        
        # Create mask from best contour
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(final_mask, [best_contour], 255)
        
        # Save debug visualizations
        contour_vis = image.copy()
        cv2.drawContours(contour_vis, [best_contour], -1, (0, 255, 0), 3)
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_contour.jpg'), contour_vis)
        
        overlay = image.copy()
        overlay[final_mask > 0] = [0, 255, 0]
        blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_overlay.jpg'), blended)
        
        area = cv2.contourArea(best_contour)
        logging.info(f"RGB-based detection: area={area:.0f}px ({area/(h*w)*100:.1f}%), score={best_score:.3f}")
        
        return final_mask
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order corners as top-left, top-right, bottom-right, bottom-left"""
        pts = pts[np.argsort(pts[:, 1])]
        top = pts[:2][np.argsort(pts[:2, 0])]
        bottom = pts[2:][np.argsort(pts[2:, 0])]
        return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)
    
    def _calculate_dimensions(self, depth_map: np.ndarray, furniture_mask: np.ndarray) -> Optional[Dict[str, float]]:
        """Calculate furniture dimensions with perspective correction."""
        # Resize mask to match depth map resolution
        mask_resized = cv2.resize(furniture_mask, (depth_map.shape[1], depth_map.shape[0]))
        
        furniture_depths = depth_map[mask_resized > 0]
        if len(furniture_depths) == 0:
            return None
        
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        main_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(main_contour)
        box = cv2.boxPoints(rect)
        
        # Get depth at each corner for perspective correction
        corner_depths = []
        for pt in box:
            x, y = int(np.clip(pt[0], 0, depth_map.shape[1]-1)), int(np.clip(pt[1], 0, depth_map.shape[0]-1))
            corner_depths.append(depth_map[y, x])
        
        # iPhone camera parameters
        focal_length_mm = 26.0
        sensor_width_mm = 7.0
        focal_length_pixels = (focal_length_mm * depth_map.shape[1]) / sensor_width_mm
        
        # Calculate real-world distances between adjacent corners
        real_distances = []
        for i in range(4):
            p1, p2 = box[i], box[(i+1)%4]
            d1, d2 = corner_depths[i], corner_depths[(i+1)%4]
            
            # Use average depth for this edge
            avg_edge_depth = (d1 + d2) / 2
            
            # Convert pixel distance to real-world distance
            pixel_dist = np.linalg.norm(p1 - p2)
            real_dist = (pixel_dist * avg_edge_depth) / focal_length_pixels
            real_distances.append(real_dist)
        
        # Opposite sides should be similar - average them
        width_meters = (real_distances[0] + real_distances[2]) / 2
        height_meters = (real_distances[1] + real_distances[3]) / 2
        
        # Estimate thickness from depth variance
        avg_depth = np.median(furniture_depths)
        depth_std = np.std(furniture_depths)
        thickness_meters = max(0.05, depth_std * 2.0)  # More realistic thickness estimation
        
        return {
            'width_m': round(float(width_meters), 3),
            'height_m': round(float(height_meters), 3), 
            'depth_m': round(float(thickness_meters), 3),
            'distance_m': round(float(avg_depth), 3),
            'corner_depths': [round(float(d), 3) for d in corner_depths],
            'edge_distances': [round(float(d), 3) for d in real_distances]
        }
    
    def _save_results(self, heic_file: str, rgb_image: np.ndarray, depth_map: np.ndarray, 
                     furniture_mask: np.ndarray, dimensions: Dict[str, float]):
        """Save comprehensive visualization and results."""
        base_name = os.path.splitext(os.path.basename(heic_file))[0]
        
        # Save depth map visualization
        depth_vis = cv2.applyColorMap(
            (depth_map / depth_map.max() * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_depth.jpg'), depth_vis)
        
        # Save furniture mask
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_mask.jpg'), furniture_mask)
        
        # Create mask overlay on RGB
        mask_overlay = rgb_image.copy()
        mask_overlay[furniture_mask > 0] = [0, 255, 0]  # Green overlay
        blended = cv2.addWeighted(rgb_image, 0.7, mask_overlay, 0.3, 0)
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_mask_overlay.jpg'), blended)
        
        # Create depth map with mask overlay
        depth_masked = depth_vis.copy()
        mask_resized = cv2.resize(furniture_mask, (depth_map.shape[1], depth_map.shape[0]))
        depth_masked[mask_resized > 0] = [0, 255, 0]  # Green overlay on depth
        depth_blended = cv2.addWeighted(depth_vis, 0.7, depth_masked, 0.3, 0)
        cv2.imwrite(os.path.join(self.depth_dir, f'{base_name}_depth_masked.jpg'), depth_blended)
        
        # Create detailed result visualization
        result_vis = rgb_image.copy()
        contours, _ = cv2.findContours(furniture_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            main_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(result_vis, [main_contour], -1, (0, 255, 0), 3)
            
            # Draw bounding rectangle
            rect = cv2.minAreaRect(main_contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(result_vis, [box], -1, (255, 0, 0), 3)
            
            # Add corner labels
            for i, corner in enumerate(box):
                cv2.circle(result_vis, tuple(corner), 10, (0, 0, 255), -1)
                cv2.putText(result_vis, str(i), tuple(corner + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Calculate and display measurements
            area = cv2.contourArea(main_contour)
            width_pixels = np.linalg.norm(box[0] - box[1])
            height_pixels = np.linalg.norm(box[1] - box[2])
            
            # Get depth statistics
            mask_resized = cv2.resize(furniture_mask, (depth_map.shape[1], depth_map.shape[0]))
            furniture_depths = depth_map[mask_resized > 0]
            avg_depth = np.median(furniture_depths)
            depth_std = np.std(furniture_depths)
            
            debug_text = [
                f"RGB: {rgb_image.shape[:2]}, Depth: {depth_map.shape}",
                f"Contour area: {area:.0f} pixels ({area/(rgb_image.shape[0]*rgb_image.shape[1])*100:.1f}%)",
                f"Bounding box: {width_pixels:.0f} x {height_pixels:.0f} pixels",
                f"Avg depth: {avg_depth:.2f}m, Std: {depth_std:.3f}m",
                f"Focal length: {(26.0 * depth_map.shape[1]) / 7.0:.0f} pixels",
                f"Real dimensions: {dimensions['width_m']:.3f}m x {dimensions['height_m']:.3f}m"
            ]
            
            y_offset = 30
            for text in debug_text:
                cv2.putText(result_vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 0), 2)
                y_offset += 35
        
        cv2.imwrite(os.path.join(self.dimensions_dir, f'{base_name}_analysis.jpg'), result_vis)
        
        # Save dimensions JSON with additional debug info
        debug_info = {
            **dimensions,
            'debug': {
                'rgb_resolution': rgb_image.shape[:2],
                'depth_resolution': depth_map.shape,
                'contour_area_pixels': float(area),
                'contour_area_percent': float(area/(rgb_image.shape[0]*rgb_image.shape[1])*100),
                'bounding_box_pixels': [float(width_pixels), float(height_pixels)],
                'depth_stats': {
                    'median': float(avg_depth),
                    'std': float(depth_std),
                    'min': float(np.min(furniture_depths)),
                    'max': float(np.max(furniture_depths))
                }
            }
        }
        
        with open(os.path.join(self.dimensions_dir, f'{base_name}_dimensions.json'), 'w') as f:
            json.dump(debug_info, f, indent=2)
            
        logging.info(f"Comprehensive analysis saved to {self.dimensions_dir}")
        logging.info(f"Dimensions: W={dimensions['width_m']}m, H={dimensions['height_m']}m, D={dimensions['depth_m']}m")

