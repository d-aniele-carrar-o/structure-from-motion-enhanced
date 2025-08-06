import os
import cv2
import pickle
import logging
import itertools
import numpy as np
from typing import List, Tuple

from .config import Config
from . import utils

ImageData = Tuple[str, List[cv2.KeyPoint], np.ndarray]

class FeatureMatcher:
    """
    Handles feature extraction and matching for the SfM pipeline.
    This class performs two main tasks:
    1. Extracts features (e.g., SIFT) and descriptors from all processed images.
    2. Matches features between all possible pairs of images, applying geometric
       verification with the Fundamental Matrix to find reliable inliers.
    """

    def __init__(self, config: Config):
        self.config = config
        self.image_names = sorted([
            os.path.splitext(f)[0] for f in os.listdir(self.config.images_dir)
            if f.lower().endswith(tuple(self.config.ext))
        ])

    def is_complete(self) -> bool:
        if not self.image_names: return False
        
        # Check if feature files exist for every image
        for name in self.image_names:
            kp_path = os.path.join(self.config.features_dir, f'kp_{name}.pkl')
            desc_path = os.path.join(self.config.features_dir, f'desc_{name}.pkl')
            if not (os.path.exists(kp_path) and os.path.exists(desc_path)):
                logging.debug(f"Feature files for {name} are missing.")
                return False
        
        if not os.listdir(self.config.matches_dir):
            logging.debug("Matches directory is empty.")
            return False
        
        return True

    def run(self):
        """Executes the full feature extraction and matching process."""
        logging.info(f"Processing {len(self.image_names)} images for feature matching.")
        
        # --- Stage 1: Feature Extraction ---
        image_data = self._extract_all_features()

        # --- Stage 2: Feature Matching ---
        self._match_all_pairs(image_data)

    def _extract_all_features(self) -> List[ImageData]:
        """Extracts SIFT features for all images, loading from cache if available."""
        logging.info("Starting feature extraction...")
        
        all_feature_data = []
        detector = cv2.SIFT_create(nfeatures=self.config.max_features)
        for name in self.image_names:
            kp_path = os.path.join(self.config.features_dir, f'kp_{name}.pkl')
            desc_path = os.path.join(self.config.features_dir, f'desc_{name}.pkl')
        
            if os.path.exists(kp_path) and os.path.exists(desc_path):
                logging.debug(f"Loading cached features for {name}.")
                with open(kp_path, 'rb') as f: keypoints = utils.deserialize_keypoints(pickle.load(f))
                with open(desc_path, 'rb') as f: descriptors = pickle.load(f)
        
            else:
                logging.debug(f"Detecting features for {name}.")
                image_path = os.path.join(self.config.images_dir, f"{name}.{self.config.ext[0]}")
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
                if image is None:
                    logging.warning(f"Could not read image {image_path}, skipping.")
                    continue
        
                keypoints, descriptors = detector.detectAndCompute(image, None)
                with open(kp_path, 'wb') as f: pickle.dump(utils.serialize_keypoints(keypoints), f)
                with open(desc_path, 'wb') as f: pickle.dump(descriptors, f)
        
            logging.debug(f"Found {len(keypoints)} features in image {name}.")
            all_feature_data.append((name, keypoints, descriptors))
        
        logging.info("Feature extraction complete.")
        
        return all_feature_data

    def _match_all_pairs(self, image_data: List[ImageData]):
        image_pairs = list(itertools.combinations(image_data, 2))
        logging.info(f"Starting feature matching for {len(image_pairs)} unique image pairs.")
        
        # Create matcher once and reuse for all pairs
        if self.config.matcher_type == 'BFMatcher':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2)
        else:
            matcher = cv2.FlannBasedMatcher_create()

        total_inliers = 0
        for i, (data1, data2) in enumerate(image_pairs):
            name1, kp1, desc1 = data1
            name2, kp2, desc2 = data2
            match_path = os.path.join(self.config.matches_dir, f'match_{name1}_{name2}.pkl')
            if os.path.exists(match_path):
                logging.debug(f"Skipping existing match: {name1} <-> {name2}")
                continue
            
            # --- Step 1: Raw KNN Matching ---
            knn_matches_1_to_2 = matcher.knnMatch(desc1, desc2, k=2)
            knn_matches_2_to_1 = matcher.knnMatch(desc2, desc1, k=2)
            
            # --- Step 2: Lowe's Ratio Test ---
            ratio_matches_1_to_2 = [m for match_pair in knn_matches_1_to_2 if len(match_pair) == 2 for m, n in [match_pair] if m.distance < self.config.ratio_threshold * n.distance]
            ratio_matches_2_to_1 = {m.queryIdx: m for match_pair in knn_matches_2_to_1 if len(match_pair) == 2 for m, n in [match_pair] if m.distance < self.config.ratio_threshold * n.distance}

            # --- Step 3: Symmetry Test (Cross-Check) ---
            symmetric_matches = []
            for m in ratio_matches_1_to_2:
                if m.trainIdx in ratio_matches_2_to_1 and ratio_matches_2_to_1[m.trainIdx].trainIdx == m.queryIdx:
                    symmetric_matches.append(m)

            # --- Step 4: Geometric Verification with Fundamental Matrix ---
            inlier_matches = []
            if len(symmetric_matches) >= 8:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in symmetric_matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in symmetric_matches])
                _, mask = cv2.findFundamentalMat(pts1, pts2, 
                                                 method=self.config.fund_method, 
                                                 ransacReprojThreshold=self.config.outlier_threshold,
                                                 confidence=self.config.fund_prob
                                                 )
                if mask is not None:
                    inlier_matches = np.array(symmetric_matches)[mask.ravel() == 1].tolist()
            
            # --- Step 5: Check against minimum match threshold ---
            if len(inlier_matches) < self.config.min_matches:
                logging.debug(f"Pair {name1}-{name2}: Found {len(inlier_matches)} inliers. Below threshold, skipping.")
                # Save empty list to indicate pair was processed
                with open(match_path, 'wb') as f: pickle.dump(utils.serialize_matches([]), f)
                continue

            logging.debug(f"Pair {name1}-{name2}: Found {len(inlier_matches)} final inliers (from {len(symmetric_matches)} symmetric matches).")
            total_inliers += len(inlier_matches)
            
            # 4. Save matches and optional visualization
            with open(match_path, 'wb') as f: pickle.dump(utils.serialize_matches(inlier_matches), f)
            
            if self.config.save_matches_vis:
                self._save_match_visualization(name1, kp1, name2, kp2, inlier_matches)
            
            if self.config.debug and (i + 1) % 20 == 0:
                logging.debug(f"  ... processed {i+1}/{len(image_pairs)} pairs.")
        
        logging.info(f"Feature matching complete. Found a total of {total_inliers} inlier matches.")

    def _save_match_visualization(self, name1, kp1, name2, kp2, matches):
        """Saves an image showing the matched keypoints between two images."""
        img1 = cv2.imread(os.path.join(self.config.images_dir, f"{name1}.{self.config.ext[0]}"))
        img2 = cv2.imread(os.path.join(self.config.images_dir, f"{name2}.{self.config.ext[0]}"))
        
        drawn_matches = sorted(matches, key=lambda x: x.distance)[:100]
        match_vis = cv2.drawMatches(img1, kp1, img2, kp2, drawn_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        vis_path = os.path.join(self.config.matches_vis_dir, f'match_{name1}_{name2}.jpg')
        cv2.imwrite(vis_path, match_vis)
        
        logging.debug(f"Saved match visualization to {vis_path}")
