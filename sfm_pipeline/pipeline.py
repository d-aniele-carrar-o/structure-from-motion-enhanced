import os
import shutil
import logging
import cv2
import numpy as np
from pickle import load
from tqdm import tqdm

from .config import Config
from .media_processor import MediaProcessor
from .feature_matcher import FeatureMatcher
from .reconstructor import Reconstructor
from . import utils


class Pipeline:
    """
    Orchestrates the entire Structure from Motion (SfM) pipeline.
    
    This class manages the sequence of operations:
    1. Cleaning previous results (optional).
    2. Processing media files (videos, HEIC images).
    3. Extracting and matching features.
    4. Performing 3D reconstruction.
    """

    def __init__(self, config: Config):
        """
        Initializes the pipeline with a configuration object.

        Args:
            config: A Config object containing all pipeline settings.
        """
        self.config = config
        self.media_processor = MediaProcessor(config)
        self.feature_matcher = FeatureMatcher(config)
        self.reconstructor = Reconstructor(config)

    def run(self):
        """Executes the SfM pipeline stages in sequence."""
        # Suppress matplotlib font debug messages
        logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)
        
        if self.config.clean:
            self._clean_directories()

        # --- Stage 1: Media Processing ---
        if not self.media_processor.is_complete():
            logging.info("Starting media processing stage...")
            self.media_processor.run()
            logging.info("Media processing stage complete.")
        else:
            logging.info("Media processing already complete, skipping.")

        if not self._verify_images_exist():
            return
            
        # --- Stage 2: Feature Matching ---
        if not self.feature_matcher.is_complete():
            logging.info("Starting feature matching stage...")
            self.feature_matcher.run()
            logging.info("Feature matching stage complete.")
        else:
            logging.info("Feature matching already complete, skipping.")

        # --- Stage 3: 3D Reconstruction ---
        logging.info("Starting 3D reconstruction stage...")
        self.reconstructor.run()
        logging.info("3D reconstruction stage complete.")
        
        # --- Stage 4: Panorama Creation ---
        logging.info("Creating panorama visualization...")
        self._create_panorama()
        logging.info("Panorama creation complete.")
        
    def _clean_directories(self):
        """Removes all previously generated files and directories."""
        logging.warning("--- CLEANING PIPELINE FILES ---")
        dirs_to_clean = [
            self.config.images_dir,
            self.config.videos_dir,
            os.path.join(self.config.dataset_dir, 'features'),
            os.path.join(self.config.dataset_dir, 'matches'),
            os.path.join(self.config.dataset_dir, 'matches_vis'),
            os.path.join(self.config.dataset_dir, 'calibrations'),
            self.config.results_dir
        ]
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    logging.info(f"Removed directory: {dir_path}")
                except (OSError, PermissionError, FileNotFoundError) as e:
                    logging.error(f"Error removing directory {dir_path}: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error removing directory {dir_path}: {e}")
        
        # Re-create essential directories after cleaning
        self.config.__post_init__()
        logging.warning("--- CLEANING COMPLETE ---")

    def _verify_images_exist(self) -> bool:
        """Checks if any images are available for processing."""
        image_files = [f for f in os.listdir(self.config.images_dir) if f.lower().endswith(tuple(self.config.ext))]
        if not image_files:
            logging.error(f"No images found in {self.config.images_dir}.")
            logging.error("Cannot proceed with feature matching. Please ensure media files are processed correctly.")
            return False
        logging.info(f"Found {len(image_files)} images to process.")
        return True

    def _create_panorama(self):
        """Creates a panorama from processed images using feature matches."""
        image_names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.config.images_dir) if f.lower().endswith(tuple(self.config.ext))])
        
        if len(image_names) < 2:
            logging.warning("Not enough images to create panorama.")
            return

        # Load all features
        features = {}
        for name in image_names:
            kp_path = os.path.join(self.config.features_dir, f'kp_{name}.pkl')
            desc_path = os.path.join(self.config.features_dir, f'desc_{name}.pkl')
            with open(kp_path, 'rb') as f: kp = load(f)
            with open(desc_path, 'rb') as f: desc = load(f)
            features[name] = (utils.deserialize_keypoints(kp), desc)

        # Calculate cumulative homographies
        H_cumulative = np.identity(3)
        images_to_warp = []
        
        # Add first image as reference
        base_name = image_names[0]
        base_img = cv2.imread(os.path.join(self.config.images_dir, f"{base_name}.{self.config.ext[0]}"))
        images_to_warp.append({'img': base_img, 'name': base_name, 'H': np.identity(3)})

        for i in range(1, len(image_names)):
            prev_name, curr_name = image_names[i-1], image_names[i]
            kp1, _ = features[prev_name]
            kp2, _ = features[curr_name]
            
            match_path = os.path.join(self.config.matches_dir, f'match_{prev_name}_{curr_name}.pkl')
            if not os.path.exists(match_path):
                match_path = os.path.join(self.config.matches_dir, f'match_{curr_name}_{prev_name}.pkl')
                if not os.path.exists(match_path):
                    logging.warning(f"Match file not found for pair: {prev_name}-{curr_name}. Stopping panorama creation.")
                    return

            with open(match_path, 'rb') as f:
                matches = utils.deserialize_matches(load(f))
            
            if len(matches) < self.config.min_matches:
                logging.warning(f"Too few matches between {prev_name} and {curr_name}. Stopping panorama creation.")
                return

            pts1 = np.float32([kp.pt for kp in kp1])[np.array([m.queryIdx for m in matches])]
            pts2 = np.float32([kp.pt for kp in kp2])[np.array([m.trainIdx for m in matches])]
            
            # Use LMEDS for more robust homography estimation against outliers
            H_relative, _ = cv2.findHomography(pts2, pts1, cv2.LMEDS, 5.0)
            if H_relative is None:
                logging.warning(f"Could not find homography for {curr_name}. Stopping panorama creation.")
                return

            H_cumulative = H_cumulative @ H_relative
            curr_img = cv2.imread(os.path.join(self.config.images_dir, f"{curr_name}.{self.config.ext[0]}"))
            images_to_warp.append({'img': curr_img, 'name': curr_name, 'H': H_cumulative})

        # Determine canvas size
        all_corners = []
        for item in images_to_warp:
            h, w = item['img'].shape[:2]
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners, item['H'])
            all_corners.append(warped_corners)

        all_corners = np.concatenate(all_corners, axis=0)
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        output_width, output_height = x_max - x_min, y_max - y_min
        
        # Warp and blend images
        panorama = np.zeros((output_height, output_width, 3), np.float32)
        pixel_counts = np.zeros((output_height, output_width, 3), np.float32)

        for item in images_to_warp:
            H_final = H_translation @ item['H']
            warped_img = cv2.warpPerspective(item['img'], H_final, (output_width, output_height)).astype(np.float32)
            mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) > 0
            panorama += warped_img
            pixel_counts += cv2.merge([mask.astype(np.float32)] * 3)

        pixel_counts[pixel_counts == 0] = 1.0
        panorama = (panorama / pixel_counts).astype(np.uint8)

        # Save panorama
        panorama_path = os.path.join(self.config.results_dir, 'panorama.jpg')
        cv2.imwrite(panorama_path, panorama)
        logging.info(f"Panorama saved to: {panorama_path}")
