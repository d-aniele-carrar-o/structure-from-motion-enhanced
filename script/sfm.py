import numpy as np 
import cv2 
import argparse
import pickle
import os 
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from utils import * 


class Camera(object): 
    def __init__(self, R, t, ref): 
        self.R = R 
        self.t = t 
        self.ref = ref


class Match(object): 
    def __init__(self, matches, img1pts, img2pts, img1idx, img2idx, mask): 
        self.matches = matches
        self.img1pts, self.img2pts = img1pts, img2pts 
        self.img1idx, self.img2idx = img1idx, img2idx
        self.mask = mask


class SFM(object): 
    def __init__(self, opts): 
        self.opts = opts
        self.point_cloud = np.zeros((0,3))
        self.point_cloud_colors = np.zeros((0,3))

        # --- Directory setup ---
        self.images_dir = os.path.join(opts.data_dir, opts.dataset, 'images')
        self.feat_dir = os.path.join(opts.data_dir, opts.dataset, 'features', opts.features)
        self.matches_dir = os.path.join(opts.data_dir, opts.dataset, 'matches', opts.matcher)
        self.out_cloud_dir = os.path.join(opts.out_dir, opts.dataset, 'point-clouds')
        os.makedirs(self.out_cloud_dir, exist_ok=True)

        self.image_names = [x.split('.')[0] for x in sorted(os.listdir(self.images_dir)) if x.split('.')[-1].lower() in opts.ext]
        
        # --- Data storage ---
        self.image_data = {} # Stores keypoints and descriptors for each image
        self.camera_poses = {} # Stores rotation (R) and translation (t) for each camera
        self.point_map = {} # Maps 3D point indices to 2D feature observations across images
        self.baseline_cam_names = []

        # --- Load Camera Calibration ---
        calib_path = os.path.join('script', 'calibrations', f'{opts.dataset}_calibration.json')
        with open(calib_path, 'r') as f:
            calib_data = json.load(f)
        self.K = np.array(calib_data['camera_matrix'], dtype=np.float64)

        print(f"\n=== CAMERA CALIBRATION LOADED: {opts.dataset} ===")
        print(f'  fx: {self.K[0,0]:.1f}, fy: {self.K[1,1]:.1f}, cx: {self.K[0,2]:.1f}, cy: {self.K[1,2]:.1f}')
    
    def _LoadFeatures(self, name): 
        if name in self.image_data:
            return self.image_data[name]['kp'], self.image_data[name]['desc']
        with open(os.path.join(self.feat_dir, f'kp_{name}.pkl'),'rb') as f: 
            kp = DeserializeKeypoints(pickle.load(f))
        with open(os.path.join(self.feat_dir, f'desc_{name}.pkl'),'rb') as f: 
            desc = pickle.load(f)
        self.image_data[name] = {'kp': kp, 'desc': desc}
        return kp, desc 

    def _LoadMatches(self, name1, name2): 
        path = os.path.join(self.matches_dir, f'match_{name1}_{name2}.pkl')
        is_swapped = False
        if not os.path.exists(path):
            path = os.path.join(self.matches_dir, f'match_{name2}_{name1}.pkl')
            is_swapped = True
            if not os.path.exists(path): return [], is_swapped
        
        with open(path, 'rb') as f: 
            matches = DeserializeMatches(pickle.load(f))
        return matches, is_swapped

    def _GetAlignedMatches(self, kp1, kp2, matches, is_swapped=False):
        if is_swapped:
            img1idx = np.array([m.trainIdx for m in matches], dtype=int)
            img2idx = np.array([m.queryIdx for m in matches], dtype=int)
        else:
            img1idx = np.array([m.queryIdx for m in matches], dtype=int)
            img2idx = np.array([m.trainIdx for m in matches], dtype=int)

        kp1_ = (np.array(kp1))[img1idx]
        kp2_ = (np.array(kp2))[img2idx]
        img1pts = np.array([kp.pt for kp in kp1_])
        img2pts = np.array([kp.pt for kp in kp2_])
        return img1pts, img2pts

    def _BaselinePoseEstimation(self, name1, name2):
        kp1, _ = self._LoadFeatures(name1)
        kp2, _ = self._LoadFeatures(name2)  
        matches, _ = self._LoadMatches(name1, name2)
        
        pts1, pts2 = self._GetAlignedMatches(kp1, kp2, matches)

        F, F_mask = cv2.findFundamentalMat(pts1, pts2, method=self.opts.fund_method,
                                        ransacReprojThreshold=self.opts.outlier_thres, confidence=self.opts.fund_prob)
        F_mask = F_mask.flatten().astype(bool)
        
        inlier_matches_F = np.array(matches)[F_mask]
        inlier_pts1_F = pts1[F_mask]
        inlier_pts2_F = pts2[F_mask]

        E = self.K.T @ F @ self.K
        _, R, t, E_mask = cv2.recoverPose(E, inlier_pts1_F, inlier_pts2_F, self.K)
        E_mask = E_mask.flatten().astype(bool)

        self.camera_poses[name1] = {'R': np.eye(3, 3), 't': np.zeros((3, 1))}
        self.camera_poses[name2] = {'R': R, 't': t}

        final_inlier_matches = inlier_matches_F[E_mask]
        self._TriangulateNewPoints(name1, name2, final_inlier_matches, is_swapped=False)

    def _NewViewPoseEstimation(self, new_view_name):
        kp_new, _ = self._LoadFeatures(new_view_name)
        
        object_points = []
        image_points = []
        
        for registered_name in self.camera_poses.keys():
            matches, is_swapped = self._LoadMatches(registered_name, new_view_name)
            if len(matches) < self.opts.min_matches: continue

            kp_reg, _ = self._LoadFeatures(registered_name)
            
            for match in matches:
                if is_swapped:
                    reg_idx, new_idx = match.trainIdx, match.queryIdx
                else:
                    reg_idx, new_idx = match.queryIdx, match.trainIdx

                pt3d_idx = None
                for idx, observations in self.point_map.items():
                    if registered_name in observations and observations[registered_name] == reg_idx:
                        pt3d_idx = idx
                        break
                
                if pt3d_idx is not None:
                    object_points.append(self.point_cloud[pt3d_idx])
                    image_points.append(kp_new[new_idx].pt)
        
        if len(object_points) < 8:
            print(f"  Warning: Not enough matches to existing 3D points ({len(object_points)}) for {new_view_name}.")
            return False

        object_points = np.array(object_points, dtype=np.float32)
        image_points = np.array(image_points, dtype=np.float32)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, self.K, None,
                                                    confidence=self.opts.pnp_prob, reprojectionError=self.opts.reprojection_thres)
        
        if inliers is None or len(inliers) < 8:
            print(f"  Warning: PnP failed for {new_view_name} (only {len(inliers) if inliers is not None else 0} inliers).")
            return False

        R, _ = cv2.Rodrigues(rvec)
        self.camera_poses[new_view_name] = {'R': R, 't': tvec}
        # print(f"  Estimated pose for {new_view_name} with {len(inliers)} inliers.")
        return True

    def _TriangulateNewPoints(self, name1, name2, matches, is_swapped):
        kp1, _ = self._LoadFeatures(name1)
        kp2, _ = self._LoadFeatures(name2)
        
        new_matches = []
        for match in matches:
            if is_swapped:
                idx1, idx2 = match.trainIdx, match.queryIdx
            else:
                idx1, idx2 = match.queryIdx, match.trainIdx
            
            is_new = True
            for observations in self.point_map.values():
                if (name1 in observations and observations[name1] == idx1) or \
                   (name2 in observations and observations[name2] == idx2):
                    is_new = False
                    break
            if is_new:
                new_matches.append(match)
        
        if not new_matches:
            return
        
        # Get aligned points for the *new* matches only. The old pts1, pts2 were redundant.
        pts1_new, pts2_new = self._GetAlignedMatches(kp1, kp2, new_matches, is_swapped)
        
        R1, t1 = self.camera_poses[name1]['R'], self.camera_poses[name1]['t']
        R2, t2 = self.camera_poses[name2]['R'], self.camera_poses[name2]['t']
        
        P1 = self.K @ np.hstack((R1, t1))
        P2 = self.K @ np.hstack((R2, t2))
        
        points4d_hom = cv2.triangulatePoints(P1, P2, pts1_new.T, pts2_new.T)
        points3d = cv2.convertPointsFromHomogeneous(points4d_hom.T)[:,0,:]

        R1_inv_T = R1.T
        t1_inv = -R1_inv_T @ t1
        pts_in_cam1_frame = (R1_inv_T @ points3d.T + t1_inv).T

        R2_inv_T = R2.T
        t2_inv = -R2_inv_T @ t2
        pts_in_cam2_frame = (R2_inv_T @ points3d.T + t2_inv).T
        
        valid_mask = (pts_in_cam1_frame[:, 2] > 0) & (pts_in_cam2_frame[:, 2] > 0)
        new_points = points3d[valid_mask]
        
        # print(f"  Triangulated {len(new_points)} new valid 3D points between {name1} and {name2}.")
        
        start_idx = len(self.point_cloud)
        self.point_cloud = np.vstack((self.point_cloud, new_points))
        
        valid_matches = np.array(new_matches)[valid_mask]
        for i, match in enumerate(valid_matches):
            pt3d_idx = start_idx + i
            if pt3d_idx not in self.point_map: self.point_map[pt3d_idx] = {}
            
            if is_swapped:
                idx1, idx2 = match.trainIdx, match.queryIdx
            else:
                idx1, idx2 = match.queryIdx, match.trainIdx
            self.point_map[pt3d_idx][name1] = idx1
            self.point_map[pt3d_idx][name2] = idx2

    def _get_point_colors(self):
        self.point_cloud_colors = np.zeros_like(self.point_cloud)
        for i, observations in self.point_map.items():
            if i >= len(self.point_cloud_colors): continue
            first_view_name = list(observations.keys())[0]
            feature_idx = observations[first_view_name]
            
            img_path = os.path.join(self.images_dir, f"{first_view_name}.{self.opts.ext[0]}")
            img = cv2.imread(img_path)
            if img is None: continue

            kp = self.image_data[first_view_name]['kp']
            
            pt = kp[feature_idx].pt
            if 0 <= int(pt[1]) < img.shape[0] and 0 <= int(pt[0]) < img.shape[1]:
                color = img[int(pt[1]), int(pt[0])]
                self.point_cloud_colors[i] = color[::-1]

    def ToPly(self, filename):
        self._get_point_colors()
        pts2ply(self.point_cloud, self.point_cloud_colors, filename)
        print(f"Point cloud saved to {filename}")

    def Visualize3D(self):
        self._get_point_colors()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.point_cloud[:, 0], self.point_cloud[:, 1], self.point_cloud[:, 2],
                   c=self.point_cloud_colors / 255.0, s=3, alpha=0.6, marker='.')
        ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
        plt.show()

    def _bundle_adjustment_residuals(self, params, n_cameras_total, n_cameras_to_optimize, n_points, 
                                     camera_indices, point_indices, points_2d, fixed_cam_poses):
        """
        Computes residuals for BA, using a combination of fixed and optimized camera poses.
        """
        optimized_camera_params = params[:n_cameras_to_optimize * 6].reshape((n_cameras_to_optimize, 6))
        points_3d = params[n_cameras_to_optimize * 6:].reshape((n_points, 3))

        # Reconstruct the full array of camera parameters
        full_camera_params = np.empty((n_cameras_total, 6), dtype=np.float64)
        
        # --- Use the first two fixed poses ---
        full_camera_params[:2,:] = fixed_cam_poses
        full_camera_params[2:,:] = optimized_camera_params

        rvecs = full_camera_params[:,:3]
        tvecs = full_camera_params[:,3:]
        
        Rs = np.empty((n_cameras_total, 3, 3))
        for i in range(n_cameras_total):
            Rs[i], _ = cv2.Rodrigues(rvecs[i])

        R_mats_obs = Rs[camera_indices]
        t_vecs_obs = tvecs[camera_indices]
        points_3d_obs = points_3d[point_indices]

        points_in_cam_frame = np.einsum('...ij,...j->...i', R_mats_obs, points_3d_obs) + t_vecs_obs
        
        Z = points_in_cam_frame[:, 2]
        Z[np.abs(Z) < 1e-8] = 1e-8
        
        points_proj_normalized = points_in_cam_frame[:, :2] / Z[:, np.newaxis]
        
        points_proj_h = np.hstack((points_proj_normalized, np.ones((points_proj_normalized.shape[0], 1))))
        
        points_proj_pixel_h = (self.K @ points_proj_h.T).T
        
        Z_pix = points_proj_pixel_h[:, 2]
        Z_pix[np.abs(Z_pix) < 1e-8] = 1e-8
        
        points_proj_2d = points_proj_pixel_h[:, :2] / Z_pix[:, np.newaxis]
        
        residuals = (points_proj_2d - points_2d).ravel()
        
        return residuals

    def _bundle_adjustment(self):
        """
        Performs Bundle Adjustment, fixing the initial wide-baseline cameras to anchor
        the scene, prevent drift, and resolve scale ambiguity.
        """
        print("\n--- Starting Bundle Adjustment ---")
        
        # --- Build the camera list with the wide baseline pair first ---
        all_cam_names = list(self.camera_poses.keys())

        # Ensure the two baseline cameras are the first two in the list
        other_cam_names = [name for name in all_cam_names if name not in self.baseline_cam_names]
        registered_image_names = self.baseline_cam_names + sorted(other_cam_names)
        # Make sure we have exactly n unique cameras
        assert len(registered_image_names) == len(all_cam_names)
        
        name_to_idx = {name: i for i, name in enumerate(registered_image_names)}
        
        if len(registered_image_names) < 3:
            print("  Not enough registered cameras to perform bundle adjustment. Skipping.")
            return

        n_cameras_total = len(registered_image_names)
        n_cameras_to_optimize = n_cameras_total - 2
        
        n_points = len(self.point_cloud)

        camera_params_to_optimize = np.empty((n_cameras_to_optimize, 6), dtype=np.float64)
        for i in range(n_cameras_to_optimize):
            cam_name = registered_image_names[i+2]
            R = self.camera_poses[cam_name]['R']
            t = self.camera_poses[cam_name]['t']
            rvec, _ = cv2.Rodrigues(R)
            camera_params_to_optimize[i, :3] = rvec.ravel()
            camera_params_to_optimize[i, 3:] = t.ravel()
            
        point_indices, camera_indices, points_2d = [], [], []
        for pt3d_idx, observations in tqdm(self.point_map.items(), desc="  Building observation data"):
            for img_name, pt2d_idx in observations.items():
                if img_name in name_to_idx:
                    point_indices.append(pt3d_idx)
                    camera_indices.append(name_to_idx[img_name])
                    points_2d.append(self.image_data[img_name]['kp'][pt2d_idx].pt)

        initial_params = np.hstack((camera_params_to_optimize.ravel(), self.point_cloud.ravel()))

        n_residuals = len(points_2d) * 2
        sparsity_matrix = lil_matrix((n_residuals, len(initial_params)), dtype=int)
        
        for i in tqdm(range(len(points_2d)), desc="  Building sparsity matrix"):
            cam_idx = camera_indices[i]
            pt_idx = point_indices[i]
            if cam_idx >= 2:
                cam_param_idx = cam_idx - 2
                sparsity_matrix[2*i:2*i+2, cam_param_idx*6 : cam_param_idx*6+6] = 1
            
            point_param_start_idx = n_cameras_to_optimize * 6
            sparsity_matrix[2*i:2*i+2, point_param_start_idx + pt_idx*3 : point_param_start_idx + pt_idx*3+3] = 1

        fixed_cam_poses = np.empty((2, 6), dtype=np.float64)
        for i in range(2):
            cam_name = registered_image_names[i]
            R = self.camera_poses[cam_name]['R']
            t = self.camera_poses[cam_name]['t']
            rvec, _ = cv2.Rodrigues(R)
            fixed_cam_poses[i, :3] = rvec.ravel()
            fixed_cam_poses[i, 3:] = t.ravel()

        print(f"  Optimizing {n_cameras_to_optimize} cameras (2 fixed) and {n_points} points...")
        res = least_squares(
            fun=self._bundle_adjustment_residuals,
            x0=initial_params,
            jac_sparsity=sparsity_matrix, method='trf', loss='soft_l1', verbose=0,
            args=(n_cameras_total, n_cameras_to_optimize, n_points, np.array(camera_indices), 
                  np.array(point_indices), np.array(points_2d), fixed_cam_poses)
        )

        optimized_params = res.x
        optimized_camera_params = optimized_params[:n_cameras_to_optimize * 6].reshape((n_cameras_to_optimize, 6))
        
        for i in range(n_cameras_to_optimize):
            cam_name = registered_image_names[i+2]
            rvec = optimized_camera_params[i, :3]
            tvec = optimized_camera_params[i, 3:]
            R, _ = cv2.Rodrigues(rvec)
            self.camera_poses[cam_name]['R'] = R
            self.camera_poses[cam_name]['t'] = tvec
            
        self.point_cloud = optimized_params[n_cameras_to_optimize * 6:].reshape((n_points, 3))

        print("--- Bundle Adjustment Completed ---")

    def Run(self):
        name1 = self.image_names[0]
        name2 = self.image_names[int(len(self.image_names) / 2)]
        self.baseline_cam_names = [name1, name2]
        
        print(f"--- Running Baseline Estimation between {name1} and {name2} ---")
        self._BaselinePoseEstimation(name1, name2)
        
        registered_views = {name1, name2}
        
        for i in tqdm(range(len(self.image_names)), desc="Processing views"):            
            new_name = self.image_names[i]
            if new_name in registered_views:
                continue
            
            if self._NewViewPoseEstimation(new_name):
                # Triangulate new points with all previously registered views
                for registered_name in registered_views:
                    matches, is_swapped = self._LoadMatches(registered_name, new_name)
                    if len(matches) > self.opts.min_matches:
                        self._TriangulateNewPoints(registered_name, new_name, matches, is_swapped)
                registered_views.add(new_name)

        print('\nReconstruction Completed:')
        print(f'  Point Cloud: {len(self.point_cloud)} 3D points')
        print(f'  Registered Cameras: {len(self.camera_poses)}')

        if len(self.point_cloud) > 0 and len(self.camera_poses) > 1:
            self._bundle_adjustment()

        if len(self.point_cloud) > 0:
            bounds = np.max(self.point_cloud, axis=0) - np.min(self.point_cloud, axis=0)
            center = np.mean(self.point_cloud, axis=0)
            print(f'  Bounds: X[{np.min(self.point_cloud, axis=0)[0]:.2f}, {np.max(self.point_cloud, axis=0)[0]:.2f}] Y[{np.min(self.point_cloud, axis=0)[1]:.2f}, {np.max(self.point_cloud, axis=0)[1]:.2f}] Z[{np.min(self.point_cloud, axis=0)[2]:.2f}, {np.max(self.point_cloud, axis=0)[2]:.2f}]')
            print(f'  Size: {bounds[0]:.2f} x {bounds[1]:.2f} x {bounds[2]:.2f}')
            print(f'  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})')
        
        self.ToPly(os.path.join(self.out_cloud_dir, 'cloud_final.ply'))

        if self.opts.visualize_3d:
            self.Visualize3D()
   

def SetArguments(parser):
    #directory stuff
    parser.add_argument('--data-dir',action='store',type=str,default='../data/',dest='data_dir',
                        help='root directory containing input data (default: ../data/)') 
    parser.add_argument('--dataset',action='store',type=str,default='custom',dest='dataset',
                        help='name of dataset (default: custom)') 
    parser.add_argument('--ext',action='store',type=str,default='jpg,png',dest='ext', 
                        help='comma seperated string of allowed image extensions \
                        (default: jpg,png)') 
    parser.add_argument('--out-dir',action='store',type=str,default='../results/',dest='out_dir',
                        help='root directory to store results in (default: ../results/)') 

    #matching parameters
    parser.add_argument('--features',action='store',type=str,default='SIFT',dest='features',
                        help='[SIFT|ORB] Feature algorithm to use (default: SIFT)')
    parser.add_argument('--matcher',action='store',type=str,default='BFMatcher',dest='matcher',
                        help='[BFMatcher|FlannBasedMatcher] Matching algorithm to use \
                        (default: BFMatcher)') 
    parser.add_argument('--cross-check',action='store_true',default=False,dest='cross_check',
                        help='Whether to cross check feature matching or not \
                        (default: False)') 

    #epipolar geometry parameters (calibration now automatic from HEIC metadata)
    parser.add_argument('--fund-method',action='store',type=str,default='FM_RANSAC',
                        dest='fund_method',help='method to estimate fundamental matrix \
                        (default: FM_RANSAC)')
    parser.add_argument('--outlier-thres',action='store',type=float,default=.9,
                        dest='outlier_thres',help='threhold value of outlier to be used in\
                         fundamental matrix estimation (default: 0.9)')
    parser.add_argument('--fund-prob',action='store',type=float,default=.9,dest='fund_prob',
                        help='confidence in fundamental matrix estimation required (default: 0.9)')
    
    #PnP parameters
    parser.add_argument('--pnp-method',action='store',type=str,default='SOLVEPNP_DLS',
                        dest='pnp_method',help='[SOLVEPNP_DLS|SOLVEPNP_EPNP|..] method used for\
                        PnP estimation, see OpenCV doc for more options (default: SOLVEPNP_DLS')
    parser.add_argument('--pnp-prob',action='store',type=float,default=.99,dest='pnp_prob',
                        help='confidence in PnP estimation required (default: 0.99)')
    parser.add_argument('--reprojection-thres',action='store',type=float,default=8.,
                        dest='reprojection_thres',help='reprojection threshold in PnP estimation \
                        (default: 8.)')

    #misc
    parser.add_argument('--plot-error',action='store_true',default=False,dest='plot_error')
    parser.add_argument('--visualize-3d',action='store_true',default=False,dest='visualize_3d',
                        help='Create interactive 3D visualization of point cloud (default: False)')

def PostprocessArgs(opts): 
    opts.fund_method = getattr(cv2,opts.fund_method)
    opts.ext = opts.ext.split(',')

if __name__=='__main__': 
    parser = argparse.ArgumentParser()
    SetArguments(parser)
    opts = parser.parse_args()
    PostprocessArgs(opts)
    
    sfm = SFM(opts)
    sfm.Run()
