import os
import shutil
import logging
import cv2
import numpy as np
from pickle import load

from .config import Config
from .media_processor import MediaProcessor
from .feature_matcher import FeatureMatcher
from .reconstructor import Reconstructor
from .dimension_extractor import DimensionExtractor
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
        self.dimension_extractor = DimensionExtractor(config)

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
        if self.config.sfm_method == 'custom':
            logging.info("Starting custom 3D reconstruction stage...")
            self.reconstructor.run()
            logging.info("Custom 3D reconstruction stage complete.")
            # Visualize sparse reconstruction if requested
            if self.config.visualize_3d:
                logging.info("Visualizing sparse point cloud (custom SfM)...")
                self.reconstructor._visualize_point_cloud()
        elif self.config.sfm_method == 'colmap':
            logging.info("Running COLMAP sparse reconstruction...")
            self._run_colmap_reconstruction()
            logging.info("COLMAP sparse reconstruction complete.")
            # Visualize COLMAP sparse reconstruction if requested
            if self.config.visualize_3d:
                logging.info("Visualizing sparse point cloud (COLMAP)...")
                self._visualize_colmap_sparse()
        
        # --- Stage 4: Dense Reconstruction (MVS) ---
        if self.config.sfm_method == 'colmap':
            sparse_cameras_path = os.path.join(self.config.colmap_dir, 'sparse', 'cameras.bin')
            if os.path.exists(sparse_cameras_path):
                dense_ply_path = self._run_dense_reconstruction()
                # Visualize dense reconstruction if requested
                if self.config.visualize_3d and dense_ply_path is not None:
                    logging.info("Visualizing dense point cloud (MVS)...")
                    self._visualize_dense_point_cloud(dense_ply_path)
            else:
                logging.warning("COLMAP sparse reconstruction not found. Cannot run dense reconstruction.")
        else:
            logging.info("Dense reconstruction currently only supported with COLMAP method. Use --sfm-method colmap to enable.")

        # --- Stage 5: Dimension Extraction (if enabled and HEIC files available) ---
        if self.config.extract_dimensions:
            heic_files = [f for f in os.listdir(self.config.media_dir) if f.lower().endswith('.heic')]
            if heic_files:
                logging.info("Starting furniture dimension extraction from HEIC files...")
                dimensions_results = self.dimension_extractor.extract_dimensions(
                    [os.path.join(self.config.media_dir, f) for f in heic_files]
                )
                if dimensions_results:
                    logging.info(f"Extracted dimensions from {len(dimensions_results)} HEIC files")
                    for filename, dims in dimensions_results.items():
                        logging.info(f"  {os.path.basename(filename)}: {dims['width_m']}×{dims['height_m']}×{dims['depth_m']}m")
                else:
                    logging.warning("No dimensions could be extracted from HEIC files")
            else:
                logging.info("No HEIC files found, skipping dimension extraction")
        else:
            logging.info("Dimension extraction disabled")

        # --- Stage 6: Panorama Creation ---
        logging.info("Creating panorama visualization...")
        self._create_panorama()
        logging.info("Panorama creation complete.")
    
    def _run_dense_reconstruction(self):
        """
        Executes the dense reconstruction using OpenMVS.
        Requires OpenMVS binaries to be in the system's PATH.
        """
        logging.info("--- Starting Dense Reconstruction (MVS) ---")
        mvs_scene_path = os.path.join(self.config.mvs_dir, 'scene.mvs')
        
        try:
            import subprocess
            
            # OpenMVS binary paths
            mvs_bin = '/usr/local/bin/OpenMVS'
            
            # 1. Create symlink to images instead of copying (faster and saves space)
            mvs_images_dir = os.path.join(self.config.mvs_dir, 'images')
            if os.path.exists(mvs_images_dir):
                if os.path.islink(mvs_images_dir):
                    os.unlink(mvs_images_dir)
                else:
                    shutil.rmtree(mvs_images_dir)
            os.symlink(os.path.abspath(self.config.images_dir), mvs_images_dir)
            logging.info(f"Created symlink to images in MVS directory: {mvs_images_dir}")
            
            # 2. Copy COLMAP sparse files to MVS directory
            mvs_sparse_dir = os.path.join(self.config.mvs_dir, 'sparse')
            if os.path.exists(mvs_sparse_dir):
                shutil.rmtree(mvs_sparse_dir)
            shutil.copytree(os.path.join(self.config.colmap_dir, 'sparse'), mvs_sparse_dir)
            
            # 3. Use COLMAP's InterfaceCOLMAP from MVS directory
            logging.info("Converting COLMAP sparse model to MVS format...")
            result = subprocess.run([f'{mvs_bin}/InterfaceCOLMAP', '-i', '.', '-o', 'scene.mvs'], capture_output=True, text=True, cwd=self.config.mvs_dir)
            if result.returncode != 0:
                logging.error(f"InterfaceCOLMAP failed with return code {result.returncode}")
                logging.error(f"STDOUT: {result.stdout}")
                logging.error(f"STDERR: {result.stderr}")
                return None
            
            logging.info("InterfaceCOLMAP conversion successful")
            
            # 4. Run OpenMVS tools in MVS directory
            logging.info("Running MVS DensifyPointCloud...")
            result = subprocess.run([f'{mvs_bin}/DensifyPointCloud', '-i', 'scene.mvs', '-o', 'scene_dense.mvs', '--resolution-level', '4', '--number-views', '2', '--verbosity', '3'], capture_output=True, text=True, cwd=self.config.mvs_dir)
            if result.returncode != 0:
                logging.error(f"DensifyPointCloud failed with return code {result.returncode}")
                logging.error(f"STDOUT: {result.stdout}")
                logging.error(f"STDERR: {result.stderr}")
                return None
            else:
                logging.info("DensifyPointCloud completed successfully!")
                logging.info(f"Output: {result.stdout[-500:]}")
            
            logging.info("Running MVS ReconstructMesh...")
            result = subprocess.run([f'{mvs_bin}/ReconstructMesh', '-i', 'scene_dense.mvs', '-o', 'scene_mesh.mvs'], capture_output=True, text=True, cwd=self.config.mvs_dir)
            if result.returncode != 0:
                logging.error(f"ReconstructMesh failed: {result.stderr}")
            
            logging.info("Running MVS TextureMesh...")
            result = subprocess.run([f'{mvs_bin}/TextureMesh', '-i', 'scene_mesh.mvs', '-o', 'scene_mesh_textured.mvs'], capture_output=True, text=True, cwd=self.config.mvs_dir)
            if result.returncode != 0:
                logging.error(f"TextureMesh failed: {result.stderr}")

            logging.info("--- Dense Reconstruction Complete ---")
            logging.info(f"✅ MVS outputs saved in: {self.config.mvs_dir}")

        except FileNotFoundError as e:
            logging.error(f"OpenMVS binaries not found at {mvs_bin}: {e}")
            logging.error("Please ensure OpenMVS is installed correctly.")
            return None
        except subprocess.CalledProcessError as e:
            logging.error(f"An error occurred while running an OpenMVS command: {e}")
            return None
        
        # Return dense point cloud path for visualization
        dense_ply_path = os.path.join(self.config.mvs_dir, 'scene_dense.ply')
        if os.path.exists(dense_ply_path):
            logging.info(f"Dense point cloud saved: {dense_ply_path}")
            return dense_ply_path
        else:
            logging.warning(f"Dense point cloud file not found: {dense_ply_path}")
            return None
    


    def _run_colmap_reconstruction(self):
        """Run COLMAP sparse reconstruction."""
        import subprocess
        
        try:
            # Use absolute paths for COLMAP
            db_path = os.path.join(self.config.colmap_dir, 'database.db')
            sparse_path = os.path.join(self.config.colmap_dir, 'sparse')
            os.makedirs(sparse_path, exist_ok=True)
            
            # Feature extraction
            subprocess.run([
                'colmap', 'feature_extractor',
                '--database_path', db_path,
                '--image_path', self.config.images_dir,
                '--ImageReader.single_camera', '1',
                '--ImageReader.camera_model', 'PINHOLE',
                '--ImageReader.camera_params', '1280,1280,960,540'
            ], check=True)
            
            # Feature matching
            subprocess.run([
                'colmap', 'exhaustive_matcher',
                '--database_path', db_path
            ], check=True)
            
            # Sparse reconstruction
            subprocess.run([
                'colmap', 'mapper',
                '--database_path', db_path,
                '--image_path', self.config.images_dir,
                '--output_path', sparse_path
            ], check=True)
            
            # Move results from sparse/0 to sparse/
            sparse_0_path = os.path.join(sparse_path, '0')
            if os.path.exists(sparse_0_path):
                for file in os.listdir(sparse_0_path):
                    shutil.move(os.path.join(sparse_0_path, file), os.path.join(sparse_path, file))
                os.rmdir(sparse_0_path)
                logging.info(f"COLMAP sparse reconstruction successful. Files saved in: {sparse_path}")
            else:
                logging.error("COLMAP mapper failed - no reconstruction generated")
                return
            
            # Skip COLMAP dense reconstruction (requires CUDA)
            logging.info("COLMAP dense reconstruction skipped (requires CUDA, not available on macOS)")
                
        except subprocess.CalledProcessError as e:
            logging.error(f"COLMAP reconstruction failed: {e}")
            logging.error("Check that images have sufficient overlap and features")
        except FileNotFoundError:
            logging.error("COLMAP not found. Please install COLMAP.")
    

    
    def _visualize_dense_point_cloud(self, dense_ply_path: str):
        """Visualize dense point cloud from PLY file."""
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Read PLY file
            points, colors = self._read_ply_file(dense_ply_path)
            if points is None:
                return
                
            # Sample points for visualization
            max_points = 50000
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]
                colors = colors[indices]
            
            fig = plt.figure(figsize=(15, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c=colors/255.0, s=1, alpha=0.6, marker='.')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Dense Point Cloud\n{len(points)} points (sampled from {len(points)} total)')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logging.error(f"Failed to visualize dense point cloud: {e}")
    
    def _read_ply_file(self, ply_path: str):
        """Read points and colors from PLY file (handles binary format)."""
        try:
            import struct
            
            with open(ply_path, 'rb') as f:
                # Read header
                header_lines = []
                while True:
                    line = f.readline().decode('ascii').strip()
                    header_lines.append(line)
                    if line == 'end_header':
                        break
                
                # Parse header
                num_points = 0
                is_binary = False
                for line in header_lines:
                    if line.startswith('element vertex'):
                        num_points = int(line.split()[-1])
                    elif line.startswith('format binary'):
                        is_binary = True
                
                points, colors = [], []
                
                if is_binary:
                    # Read binary data: x(float) y(float) z(float) r(uchar) g(uchar) b(uchar)
                    for _ in range(num_points):
                        data = struct.unpack('<fffBBB', f.read(15))  # 3 floats + 3 bytes
                        points.append([data[0], data[1], data[2]])
                        colors.append([data[3], data[4], data[5]])
                else:
                    # Read ASCII data
                    for _ in range(num_points):
                        line = f.readline().decode('ascii').split()
                        points.append([float(line[0]), float(line[1]), float(line[2])])
                        colors.append([int(line[3]), int(line[4]), int(line[5])])
            
            points = np.array(points)
            colors = np.array(colors)
            
            # Filter out invalid points (NaN, Inf, or extreme values)
            valid_mask = (
                np.isfinite(points).all(axis=1) &  # No NaN or Inf
                (np.abs(points) < 1000).all(axis=1)  # Reasonable coordinate range
            )
            
            if valid_mask.sum() == 0:
                logging.warning("No valid points found in PLY file")
                return None, None
            
            logging.info(f"Filtered {len(points) - valid_mask.sum()} invalid points, keeping {valid_mask.sum()} valid points")
            return points[valid_mask], colors[valid_mask]
        except Exception as e:
            logging.error(f"Failed to read PLY file {ply_path}: {e}")
            return None, None
    
    def _visualize_colmap_sparse(self):
        """Visualize COLMAP sparse reconstruction."""
        try:
            import subprocess
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Convert binary to text format for reading
            sparse_path = os.path.join(self.config.colmap_dir, 'sparse')
            text_path = os.path.join(self.config.colmap_dir, 'sparse_text')
            os.makedirs(text_path, exist_ok=True)
            
            subprocess.run([
                'colmap', 'model_converter',
                '--input_path', sparse_path,
                '--output_path', text_path,
                '--output_type', 'TXT'
            ], check=True)
            
            # Read points3D.txt
            points_file = os.path.join(text_path, 'points3D.txt')
            if os.path.exists(points_file):
                points = []
                colors = []
                with open(points_file, 'r') as f:
                    for line in f:
                        if line.startswith('#') or not line.strip():
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK...
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                            points.append([x, y, z])
                            colors.append([r, g, b])
                
                if points:
                    points = np.array(points)
                    colors = np.array(colors)
                    
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                              c=colors/255.0, s=3, alpha=0.6, marker='.')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    ax.set_zlabel('Z')
                    ax.set_title(f'COLMAP Sparse Reconstruction\n{len(points)} points')
                    plt.show()
                    logging.info(f"Visualized COLMAP sparse reconstruction with {len(points)} points")
                else:
                    logging.warning("No 3D points found in COLMAP reconstruction")
            else:
                logging.error(f"COLMAP points file not found: {points_file}")
                
        except Exception as e:
            logging.error(f"Failed to visualize COLMAP sparse reconstruction: {e}")

    def _clean_directories(self):
        """Removes all previously generated files and directories."""
        logging.info("--- CLEANING PIPELINE FILES ---")
        # Simply remove the entire results directory since everything is now organized there
        dirs_to_clean = [self.config.results_dir]
        
        for dir_path in dirs_to_clean:
            if os.path.exists(dir_path):
                try:
                    if self.config.debug:
                        # In debug mode, show all subdirectories being cleaned
                        subdirs = []
                        for root, dirs, files in os.walk(dir_path):
                            if files:  # Only show directories with files
                                subdirs.append(root)
                        logging.debug(f"Cleaning subdirectories: {subdirs}")
                    
                    total_files = sum([len(files) for r, d, files in os.walk(dir_path)])
                    shutil.rmtree(dir_path)
                    logging.info(f"✅ Cleaned {total_files} files from: {dir_path}")
                except (OSError, PermissionError, FileNotFoundError) as e:
                    logging.error(f"Error removing directory {dir_path}: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error removing directory {dir_path}: {e}")
            else:
                logging.info(f"Directory already clean: {dir_path}")
        
        # Re-create essential directories after cleaning
        self.config.__post_init__()
        logging.info("--- CLEANING COMPLETE ---")

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
        # Get image names that have corresponding feature files
        all_image_names = sorted([os.path.splitext(f)[0] for f in os.listdir(self.config.images_dir) if f.lower().endswith(tuple(self.config.ext))])
        
        # Filter to only include images with feature files
        image_names = []
        for name in all_image_names:
            kp_path = os.path.join(self.config.features_dir, f'kp_{name}.pkl')
            desc_path = os.path.join(self.config.features_dir, f'desc_{name}.pkl')
            if os.path.exists(kp_path) and os.path.exists(desc_path):
                image_names.append(name)
        
        if len(image_names) < 2:
            logging.warning("Not enough images with features to create panorama.")
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
        panorama_path = os.path.join(self.config.panorama_dir, 'panorama.jpg')
        cv2.imwrite(panorama_path, panorama)
        logging.info(f"Panorama saved to: {panorama_path}")
