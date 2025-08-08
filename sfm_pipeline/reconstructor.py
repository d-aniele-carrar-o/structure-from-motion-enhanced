import os
import cv2
import logging
import json
import numpy as np
from pickle import load
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .config import Config
from . import utils


class Reconstructor:
    """
    Performs the 3D reconstruction from 2D features and matches.

    The process follows these main steps:
    1.  Load camera calibration data.
    2.  Establish an initial two-view baseline reconstruction.
    3.  Incrementally add new views using Perspective-n-Point (PnP).
    4.  Triangulate new 3D points from newly registered views.
    5.  Perform Bundle Adjustment to globally optimize all camera poses and 3D points.
    """

    def __init__(self, config: Config):
        self.config = config

        self.point_cloud = np.zeros((0, 3))
        self.point_cloud_colors = np.zeros((0, 3))

        self.image_names = sorted([os.path.splitext(f)[0] for f in os.listdir(config.images_dir) if f.lower().endswith(tuple(config.ext))])
        self.image_data: Dict[str, Dict[str, Any]] = {}
        self.image_size: Optional[List[int]] = None
        
        self.camera_poses: Dict[str, Dict[str, np.ndarray]] = {}
        self.point_map: Dict[int, Dict[str, int]] = {}
        
        self.K = self._load_camera_calibration()

    def run(self):
        """Executes the full 3D reconstruction process."""
        if self.K is None: return

        # 1. Establish baseline reconstruction from two best views (most matches)
        baseline_pair = self._find_best_baseline_pair()
        if not baseline_pair:
            logging.error("Could not find a suitable baseline pair. Aborting reconstruction.")
            return
        
        name1, name2, baseline_matches = baseline_pair
        logging.info(f"--- Establishing baseline with best pair: '{name1}' and '{name2}' ({len(baseline_matches)} matches) ---")
        self._estimate_baseline_pose(name1, name2, baseline_matches)
        self.baseline_cameras = [name1, name2]

        registered_views = {name1, name2}
        
        # 2. Incrementally add new views
        for i, new_name in enumerate(self.image_names):
            if new_name in registered_views: continue

            logging.debug(f"Processing view {i+1}/{len(self.image_names)}: '{new_name}'")
            if self._add_new_view_to_reconstruction(new_name):
                registered_views.add(new_name)

        logging.info(f"Initial reconstruction complete. Registered {len(self.camera_poses)} cameras and {len(self.point_cloud)} 3D points.")
        self._log_point_cloud_statistics("after initial reconstruction")
        
        # 3. Optimize the entire reconstruction (optional)
        if self.config.enable_bundle_adjustment and len(self.point_cloud) > 0 and len(self.camera_poses) > 2:
            self._run_bundle_adjustment()
            self._log_point_cloud_statistics("after bundle adjustment")
        elif self.config.enable_bundle_adjustment:
            logging.warning("Skipping bundle adjustment (not enough cameras or points).")
        else:
            logging.info("Bundle adjustment disabled (enable_bundle_adjustment=False).")

        # 4. Save the final point cloud
        if len(self.point_cloud) > 0:
            output_path = os.path.join(self.config.custom_sfm_dir, 'point_cloud.ply')
            self._save_point_cloud(output_path)

    def _find_best_baseline_pair(self) -> Optional[Tuple[str, str, List[cv2.DMatch]]]:
        best_pair = (None, None, [])
        max_matches = 0
        match_files = os.listdir(self.config.matches_dir)
        
        for match_file in match_files:
            if not match_file.startswith('match_') or not match_file.endswith('.pkl'):
                continue
                
            with open(os.path.join(self.config.matches_dir, match_file), 'rb') as f:
                matches = utils.deserialize_matches(load(f))
        
            if len(matches) > max_matches:
                max_matches = len(matches)
                # Extract names from match_name1_name2.pkl format
                filename_without_ext = os.path.splitext(match_file)[0]
                # Remove 'match_' prefix and split by '_' to get the two names
                names_part = filename_without_ext[6:]  # Remove 'match_'
                # Find the split point by looking for the last occurrence that creates valid names
                parts = names_part.split('_')
                # Reconstruct the two names by finding the split point
                for i in range(1, len(parts)):
                    name1 = '_'.join(parts[:i])
                    name2 = '_'.join(parts[i:])
                    if name1 in self.image_names and name2 in self.image_names:
                        best_pair = (name1, name2, matches)
                        break
        
        if max_matches > 0:
            return best_pair
        
        return None

    def _estimate_baseline_pose(self, name1: str, name2: str, matches: List[cv2.DMatch]):
        kp1, _ = self._load_features(name1)
        kp2, _ = self._load_features(name2)
        
        if not matches:
            logging.error("Cannot establish baseline pose with zero matches.")
            return
        
        pts1, pts2 = utils.get_aligned_matches(kp1, kp2, matches)
        
        E, E_mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, self.K, mask=E_mask)
        
        self.camera_poses[name1] = {'R': np.eye(3), 't': np.zeros((3, 1))}
        self.camera_poses[name2] = {'R': R, 't': t}
        
        inlier_matches = np.array(matches)[pose_mask.ravel().astype(bool)]
        
        self._triangulate_new_points(name1, name2, inlier_matches, is_swapped=False)

    def _add_new_view_to_reconstruction(self, new_view_name: str) -> bool:
        object_points, image_points = self._find_2d_3d_correspondences(new_view_name)
        if len(object_points) < 8:
            logging.warning(f"Skipping view '{new_view_name}': not enough 2D-3D matches found ({len(object_points)}).")
            return False
        
        success, rvec, tvec, inliers = cv2.solvePnPRansac(np.array(object_points), np.array(image_points), self.K, None, confidence=self.config.pnp_prob, reprojectionError=self.config.reprojection_threshold)
        
        if not success or (inliers is not None and len(inliers) < 8):
            logging.warning(f"PnP failed for '{new_view_name}' (only {len(inliers) if inliers is not None else 0} inliers).")
            return False
        
        R, _ = cv2.Rodrigues(rvec)
        self.camera_poses[new_view_name] = {'R': R, 't': tvec}
        logging.info(f"Successfully registered view '{new_view_name}' with {len(inliers)} inlier matches.")
        
        for registered_name in self.camera_poses:
            if registered_name == new_view_name: continue
            matches, is_swapped = self._load_matches(registered_name, new_view_name)
            if len(matches) > self.config.min_matches:
                self._triangulate_new_points(registered_name, new_view_name, matches, is_swapped)
        
        return True

    def _find_2d_3d_correspondences(self, new_view_name: str) -> Tuple[List, List]:
        kp_new, _ = self._load_features(new_view_name)
        object_points, image_points = [], []
        
        # Build a reverse lookup map from (view_name, kp_idx) -> pt3d_idx
        point_cloud_indices = {}
        for pt3d_idx, observations in self.point_map.items():
            for view_name, kp_idx in observations.items():
                point_cloud_indices[(view_name, kp_idx)] = pt3d_idx
        
        # Iterate through already registered views to find links to the new view
        for registered_name in self.camera_poses:
            if registered_name == new_view_name: continue
            
            matches, is_swapped = self._load_matches(registered_name, new_view_name)
        
            for m in matches:
                # Get keypoint indices for the registered view and the new view
                kp_idx_reg = m.trainIdx if is_swapped else m.queryIdx
                kp_idx_new = m.queryIdx if is_swapped else m.trainIdx
                
                # Check if the keypoint from the registered view corresponds to a 3D point
                pt3d_idx = point_cloud_indices.get((registered_name, kp_idx_reg))
        
                if pt3d_idx is not None:
                    object_points.append(self.point_cloud[pt3d_idx])
                    image_points.append(kp_new[kp_idx_new].pt)
        
        return object_points, image_points

    def _triangulate_new_points(self, name1: str, name2: str, matches: List, is_swapped: bool):
        kp1, _ = self._load_features(name1)
        kp2, _ = self._load_features(name2)
        
        if is_swapped:
            kp1, kp2, name1, name2 = kp2, kp1, name2, name1
        
        pts1, pts2 = utils.get_aligned_matches(kp1, kp2, matches)
        
        R1, t1 = self.camera_poses[name1]['R'], self.camera_poses[name1]['t']
        R2, t2 = self.camera_poses[name2]['R'], self.camera_poses[name2]['t']
        P1, P2 = self.K @ np.hstack((R1, t1)), self.K @ np.hstack((R2, t2))
        
        points4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points3d = cv2.convertPointsFromHomogeneous(points4d_hom.T).reshape(-1, 3)
        
        # --- Reprojection Error Check ---
        rvec1, _ = cv2.Rodrigues(R1)
        rvec2, _ = cv2.Rodrigues(R2)
        
        proj_pts1, _ = cv2.projectPoints(points3d, rvec1, t1, self.K, None)
        proj_pts2, _ = cv2.projectPoints(points3d, rvec2, t2, self.K, None)
        
        error1 = np.linalg.norm(proj_pts1.reshape(-1, 2) - pts1, axis=1)
        error2 = np.linalg.norm(proj_pts2.reshape(-1, 2) - pts2, axis=1)
        
        reprojection_mask = (error1 < self.config.reprojection_threshold) & (error2 < self.config.reprojection_threshold)
        
        # --- Cheirality Check (points in front of camera) ---
        pts_in_cam1, pts_in_cam2 = (R1 @ points3d.T + t1).T, (R2 @ points3d.T + t2).T
        cheirality_mask = (pts_in_cam1[:, 2] > 0) & (pts_in_cam2[:, 2] > 0)
        
        valid_mask = reprojection_mask & cheirality_mask
        
        new_points_count = 0
        start_idx = len(self.point_cloud)
        for i, match in enumerate(np.array(matches)[valid_mask]):
            idx1, idx2 = match.queryIdx, match.trainIdx
            is_new = not any((name1 in o and o[name1] == idx1) or (name2 in o and o[name2] == idx2) for o in self.point_map.values())
        
            if is_new:
                pt3d_idx = start_idx + new_points_count
                self.point_cloud = np.vstack((self.point_cloud, points3d[valid_mask][i]))
        
                if pt3d_idx not in self.point_map: self.point_map[pt3d_idx] = {}
                self.point_map[pt3d_idx][name1], self.point_map[pt3d_idx][name2] = idx1, idx2
                new_points_count += 1
        
        if new_points_count > 0:
            logging.debug(f"Triangulated {new_points_count} new valid 3D points between {name1} and {name2}.")

    def _run_bundle_adjustment(self):
        """
        Performs Bundle Adjustment to globally optimize camera poses and 3D point locations.
        It fixes the first two (baseline) cameras to anchor the scene and prevent drift.
        """
        logging.info("--- Starting Bundle Adjustment ---")
        
        registered_names = list(self.camera_poses.keys())
        if len(registered_names) < 3:
            logging.info("Not enough cameras for bundle adjustment. Skipping.")
            return
            
        # Ensure baseline cameras are first
        baseline_names = self.baseline_cameras[:2]
        other_names = [name for name in registered_names if name not in baseline_names]
        ordered_names = baseline_names + sorted(other_names)
        name_to_idx = {name: i for i, name in enumerate(ordered_names)}
        
        n_cameras_total = len(ordered_names)
        n_cameras_to_optimize = n_cameras_total - 2  # Fix first two
        n_points = len(self.point_cloud)
        
        # Pack optimizable camera parameters (skip first two)
        camera_params = np.empty((n_cameras_to_optimize, 6))
        for i in range(n_cameras_to_optimize):
            cam_name = ordered_names[i + 2]
            R = self.camera_poses[cam_name]['R']
            t = self.camera_poses[cam_name]['t']
            rvec, _ = cv2.Rodrigues(R)
            camera_params[i, :3] = rvec.ravel()
            camera_params[i, 3:] = t.ravel()
        
        # Build observation data
        point_indices, camera_indices, points_2d = [], [], []
        for pt3d_idx, observations in self.point_map.items():
            for img_name, pt2d_idx in observations.items():
                if img_name in name_to_idx:
                    point_indices.append(pt3d_idx)
                    camera_indices.append(name_to_idx[img_name])
                    kp = self._load_features(img_name)[0]
                    points_2d.append(kp[pt2d_idx].pt)
        
        # Pack parameters
        initial_params = np.hstack((camera_params.ravel(), self.point_cloud.ravel()))
        
        # Build sparsity matrix
        n_residuals = len(points_2d) * 2
        sparsity = lil_matrix((n_residuals, len(initial_params)), dtype=int)
        
        for i in range(len(points_2d)):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            
            # Camera parameters (if not fixed)
            if cam_idx >= 2:
                cam_param_idx = cam_idx - 2
                sparsity[2*i:2*i+2, cam_param_idx*6:cam_param_idx*6+6] = 1
            
            # Point parameters
            point_param_start = n_cameras_to_optimize * 6
            sparsity[2*i:2*i+2, point_param_start + pt_idx*3:point_param_start + pt_idx*3+3] = 1
        
        # Fixed camera poses
        fixed_poses = np.empty((2, 6))
        for i in range(2):
            cam_name = ordered_names[i]
            R = self.camera_poses[cam_name]['R']
            t = self.camera_poses[cam_name]['t']
            rvec, _ = cv2.Rodrigues(R)
            fixed_poses[i, :3] = rvec.ravel()
            fixed_poses[i, 3:] = t.ravel()
        
        # Run optimization
        def residuals_func(params):
            return self._bundle_adjustment_residuals(
                params, n_cameras_total, n_cameras_to_optimize, n_points,
                np.array(camera_indices), np.array(point_indices), 
                np.array(points_2d), fixed_poses
            )
        
        logging.info(f"Optimizing {n_cameras_to_optimize} cameras and {n_points} points...")
        result = least_squares(
            residuals_func, initial_params, jac_sparsity=sparsity,
            method='trf', loss='soft_l1', verbose=0
        )
        
        # Unpack results
        optimized_params = result.x
        optimized_cameras = optimized_params[:n_cameras_to_optimize * 6].reshape((n_cameras_to_optimize, 6))
        
        for i in range(n_cameras_to_optimize):
            cam_name = ordered_names[i + 2]
            rvec = optimized_cameras[i, :3]
            tvec = optimized_cameras[i, 3:]
            R, _ = cv2.Rodrigues(rvec)
            self.camera_poses[cam_name]['R'] = R
            self.camera_poses[cam_name]['t'] = tvec
        
        self.point_cloud = optimized_params[n_cameras_to_optimize * 6:].reshape((n_points, 3))
        logging.info("--- Bundle Adjustment Complete ---")
    
    def _bundle_adjustment_residuals(self, params, n_cameras_total, n_cameras_to_optimize, n_points,
                                   camera_indices, point_indices, points_2d, fixed_poses):
        """Compute residuals for bundle adjustment."""
        optimized_cameras = params[:n_cameras_to_optimize * 6].reshape((n_cameras_to_optimize, 6))
        points_3d = params[n_cameras_to_optimize * 6:].reshape((n_points, 3))
        
        # Combine fixed and optimized camera parameters
        all_camera_params = np.empty((n_cameras_total, 6))
        all_camera_params[:2] = fixed_poses
        all_camera_params[2:] = optimized_cameras
        
        # Convert to rotation matrices
        Rs = np.empty((n_cameras_total, 3, 3))
        ts = all_camera_params[:, 3:]
        for i in range(n_cameras_total):
            Rs[i], _ = cv2.Rodrigues(all_camera_params[i, :3])
        
        # Project points
        observed_Rs = Rs[camera_indices]
        observed_ts = ts[camera_indices]
        observed_points = points_3d[point_indices]
        
        # Transform to camera coordinates
        points_cam = np.einsum('...ij,...j->...i', observed_Rs, observed_points) + observed_ts
        
        # Project to image plane
        points_cam[:, 2] = np.maximum(points_cam[:, 2], 1e-8)  # Avoid division by zero
        points_normalized = points_cam[:, :2] / points_cam[:, 2:3]
        
        # Apply camera matrix
        points_homogeneous = np.column_stack((points_normalized, np.ones(len(points_normalized))))
        points_projected = (self.K @ points_homogeneous.T).T
        points_projected = points_projected[:, :2] / points_projected[:, 2:3]
        
        # Compute residuals
        residuals = (points_projected - points_2d).ravel()
        return residuals

    def _get_point_colors(self):
        self.point_cloud_colors = np.zeros_like(self.point_cloud)
        
        for pt3d_idx, observations in self.point_map.items():
            if pt3d_idx >= len(self.point_cloud): continue
        
            first_view_name = list(observations.keys())[0]
            feature_idx = observations[first_view_name]
            image_path = os.path.join(self.config.images_dir, f"{first_view_name}.{self.config.ext[0]}")
            img = cv2.imread(image_path)
        
            if img is None: continue
        
            kp = self._load_features(first_view_name)[0]
            pt = kp[feature_idx].pt
        
            if 0 <= int(pt[1]) < img.shape[0] and 0 <= int(pt[0]) < img.shape[1]:
                self.point_cloud_colors[pt3d_idx] = img[int(pt[1]), int(pt[0])][::-1]

    def _save_point_cloud(self, filepath: str):
        self._get_point_colors()
        utils.points_to_ply(self.point_cloud, self.point_cloud_colors.astype(int), filepath)
        logging.info(f"✅ Final point cloud saved to: {filepath}")

    def _visualize_point_cloud(self):
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2], 
                   c=self.point_cloud_colors / 255.0, s=3, alpha=0.6, marker='.')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title('Final 3D Point Cloud')
        
        plt.show()

    def _load_features(self, name: str) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        if name in self.image_data: return self.image_data[name]['kp'], self.image_data[name]['desc']
        
        with open(os.path.join(self.config.features_dir, f'kp_{name}.pkl'), 'rb') as f: 
            kp = utils.deserialize_keypoints(load(f))
        
        with open(os.path.join(self.config.features_dir, f'desc_{name}.pkl'), 'rb') as f:
            desc = load(f)
        
        self.image_data[name] = {'kp': kp, 'desc': desc}
        
        return kp, desc

    def _load_matches(self, name1: str, name2: str) -> Tuple[List[cv2.DMatch], bool]:
        path1 = os.path.join(self.config.matches_dir, f'match_{name1}_{name2}.pkl')
        path2 = os.path.join(self.config.matches_dir, f'match_{name2}_{name1}.pkl')
        if os.path.exists(path1):
            with open(path1, 'rb') as f: return utils.deserialize_matches(load(f)), False
        if os.path.exists(path2):
            with open(path2, 'rb') as f: return utils.deserialize_matches(load(f)), True
        return [], False

    def _load_camera_calibration(self) -> Optional[np.ndarray]:
        calib_path = self.config.get_calibration_path()
        
        if not os.path.exists(calib_path):
            logging.warning(f"Calibration file not found: {calib_path}. Will be created during media processing.")
            return None
        
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        
        # Store the image size from the calibration file
        self.image_size = calib_data.get('image_size')

        K = np.array(calib_data['camera_matrix'], dtype=np.float64)
        logging.info(f"Successfully loaded camera calibration from {calib_path}")
        k_str = f"[[{K[0,0]:.2f}, 0, {K[0,2]:.2f}], [0, {K[1,1]:.2f}, {K[1,2]:.2f}], [0, 0, 1]]"
        logging.info(f"  K = {k_str}")
        
        return K

    def _log_point_cloud_statistics(self, stage: str):
        """Log detailed point cloud statistics."""
        if len(self.point_cloud) == 0:
            logging.info(f"📊 Point cloud statistics {stage}: No points")
            return
            
        # Basic statistics
        n_points = len(self.point_cloud)
        n_cameras = len(self.camera_poses)
        
        # Spatial extent
        min_coords = np.min(self.point_cloud, axis=0)
        max_coords = np.max(self.point_cloud, axis=0)
        extent = max_coords - min_coords
        centroid = np.mean(self.point_cloud, axis=0)
        
        # Point density (observations per point)
        observations_per_point = [len(obs) for obs in self.point_map.values()]
        avg_observations = np.mean(observations_per_point) if observations_per_point else 0
        
        logging.info(f"📊 Point cloud statistics {stage}:")
        logging.info(f"  • Points: {n_points:,} | Cameras: {n_cameras}")
        logging.info(f"  • Spatial extent: X={extent[0]:.2f}, Y={extent[1]:.2f}, Z={extent[2]:.2f}")
        logging.info(f"  • Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
        logging.info(f"  • Avg observations per point: {avg_observations:.1f}")

    # def export_to_openmvs(self, mvs_scene_path: str):
    #     """
    #     Exports the reconstructed scene to the OpenMVS format.
    #     """
    #     logging.info("Exporting reconstruction to OpenMVS format...")
        
    #     # Ensure the directory for the .mvs file exists
    #     os.makedirs(os.path.dirname(mvs_scene_path), exist_ok=True)
        
    #     with open(mvs_scene_path, 'w') as f:
    #         # Header
    #         f.write("OPENMVS_SCENE\n")
    #         f.write("# Produced by Gemini's Awesome SfM Pipeline\n\n")
            
    #         # Platforms (not strictly needed but good practice)
    #         f.write(f"PLATFORMS 1\n")
    #         f.write("0 0\n\n") # Platform 0 with 0 cameras (we define cameras individually)

    #         # Images
    #         n_images = len(self.camera_poses)
    #         f.write(f"IMAGES {n_images}\n")
            
    #         # Map image names to sorted indices
    #         sorted_names = sorted(list(self.camera_poses.keys()))
    #         name_to_id = {name: i for i, name in enumerate(sorted_names)}

    #         for i, name in enumerate(sorted_names):
    #             # Construct the full, absolute path to the image
    #             image_path = os.path.abspath(os.path.join(self.config.images_dir, f"{name}.{self.config.ext[0]}"))
    #             f.write(f"{i} {image_path}\n")
    #         f.write("\n")

    #         # Poses
    #         f.write(f"POSES {n_images}\n")
    #         for i, name in enumerate(sorted_names):
    #             pose = self.camera_poses[name]
    #             R = pose['R']
    #             # MVS needs the camera center, which is -R.T @ t
    #             t = pose['t']
    #             camera_center = -R.T @ t
                
    #             f.write(f"{i} ")
    #             f.write(" ".join(map(str, camera_center.flatten())) + " ")
    #             f.write(" ".join(map(str, R.flatten())) + "\n")
    #         f.write("\n")

    #         # Intrinsics
    #         f.write("INTRINSICS 1\n")
    #         # Assuming a single camera calibration for all images
    #         fx, fy = self.K[0, 0], self.K[1, 1]
    #         cx, cy = self.K[0, 2], self.K[1, 2]
            
    #         # Load image size from calibration file
    #         calib_path = self.config.get_calibration_path()
    #         if os.path.exists(calib_path):
    #             with open(calib_path, 'r') as calib_f:
    #                 calib_data = json.load(calib_f)
    #                 if 'image_size' in calib_data:
    #                     img_size = calib_data['image_size']
            
    #         f.write(f"0 0 {img_size[0]} {img_size[1]} {fx} {fy} {cx} {cy} 0 0\n\n")

    #         # Link cameras to intrinsics
    #         f.write(f"IMAGES_INTRINSICS {n_images}\n")
    #         for i in range(n_images):
    #             f.write(f"{i} 0\n") # All images use intrinsic profile 0
    #         f.write("\n")

    #         # Sparse Point Cloud (optional but recommended)
    #         f.write(f"POINTS {len(self.point_cloud)}\n")
    #         for point in self.point_cloud:
    #             f.write(" ".join(map(str, point)) + "\n")
        
    #     logging.info(f"✅ Scene saved to: {mvs_scene_path}")

    # In reconstructor.py

    def export_to_openmvs(self, mvs_scene_path: str):
        """
        Exports the full reconstruction to OpenMVS format.
        """
        logging.info("Exporting full reconstruction to OpenMVS format...")
        os.makedirs(os.path.dirname(mvs_scene_path), exist_ok=True)

        if len(self.camera_poses) < 2:
            logging.error("Cannot create scene, not enough registered cameras.")
            return

        cameras_to_export = self.camera_poses
        points_to_write = self.point_cloud
        
        logging.info(f"Creating scene with {len(cameras_to_export)} cameras and {len(points_to_write)} points.")

        with open(mvs_scene_path, 'w') as f:
            f.write("OPENMVS_SCENE\n")
            f.write("# DEBUG: Minimal Two-View Scene (No Platforms)\n\n")

            n_images = len(cameras_to_export)
            f.write(f"IMAGES {n_images}\n")
            sorted_names = sorted(list(cameras_to_export.keys()))
            for i, name in enumerate(sorted_names):
                image_path = os.path.abspath(os.path.join(self.config.images_dir, f"{name}.{self.config.ext[0]}"))
                f.write(f"{i} {image_path}\n")
            f.write("\n")

            f.write(f"POSES {n_images}\n")
            for i, name in enumerate(sorted_names):
                pose = cameras_to_export[name]
                R = pose['R']
                t = pose['t']
                camera_center = -R.T @ t
                f.write(f"{i} {' '.join(map(str, camera_center.flatten()))} {' '.join(map(str, R.flatten()))}\n")
            f.write("\n")

            # --- Main Changes Are Here ---
            # We removed the PLATFORMS section entirely.
            # For INTRINSICS, the second number is the platform ID. We use -1 for no platform.
            f.write("INTRINSICS 1\n")
            fx, fy = self.K[0, 0], self.K[1, 1]
            cx, cy = self.K[0, 2], self.K[1, 2]
            img_size = self.image_size
            f.write(f"0 -1 {img_size[0]} {img_size[1]} {fx} {fy} {cx} {cy} 0 0\n\n")

            f.write(f"IMAGES_INTRINSICS {n_images}\n")
            for i in range(n_images):
                f.write(f"{i} 0\n")
            f.write("\n")

            f.write(f"POINTS {len(points_to_write)}\n")
            for point in points_to_write:
                f.write(" ".join(map(str, point)) + "\n")
        
        logging.info(f"✅ Minimal debug scene saved to: {mvs_scene_path}")
    