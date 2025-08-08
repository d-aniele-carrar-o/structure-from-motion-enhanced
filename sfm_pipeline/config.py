import os
from dataclasses import dataclass, field
from typing import List, Any

@dataclass
class Config:
    """
    Central configuration class for the SfM pipeline.
    Holds all paths, parameters, and settings.
    """
    # --- Core Paths ---
    data_dir: str
    dataset: str
    media_dir: str
    out_dir: str

    # --- Flags ---
    debug: bool
    clean: bool

    # --- File Extensions ---
    ext: List[str]

    # --- Media Processing ---
    frame_interval: int
    max_frames: int
    quality_threshold: float

    # --- Feature Matching ---
    features_type: str
    matcher_type: str
    max_features: int
    ratio_threshold: float

    # --- SFM & Geometric Verification ---
    sfm_method: str
    min_matches: int
    fund_method: Any  # cv2 method like cv2.FM_RANSAC
    outlier_threshold: float
    fund_prob: float
    reprojection_threshold: float
    pnp_prob: float
    
    # --- Output & Visualization ---
    save_matches_vis: bool
    visualize_3d: bool
    enable_bundle_adjustment: bool = False

    # --- Derived Paths (auto-generated) ---
    dataset_dir: str = field(init=False)
    results_dir: str = field(init=False)
    # Processing directories (organized under results)
    processing_dir: str = field(init=False)
    images_dir: str = field(init=False)
    videos_dir: str = field(init=False)
    features_dir: str = field(init=False)
    matches_dir: str = field(init=False)
    matches_vis_dir: str = field(init=False)
    calibration_dir: str = field(init=False)
    # Output subdirectories
    sparse_dir: str = field(init=False)
    dense_dir: str = field(init=False)
    colmap_dir: str = field(init=False)
    custom_sfm_dir: str = field(init=False)
    mvs_dir: str = field(init=False)
    panorama_dir: str = field(init=False)

    def __post_init__(self):
        """Generate derived paths and create directories."""
        self.dataset_dir = os.path.join(self.data_dir, self.dataset)
        
        # Organized results structure - everything under results/
        self.results_dir = os.path.join(self.out_dir, self.dataset)
        self.processing_dir = os.path.join(self.results_dir, 'processing')
        
        # Processing directories (moved under results/processing/)
        self.images_dir = os.path.join(self.processing_dir, 'images')
        self.videos_dir = os.path.join(self.processing_dir, 'videos')
        self.features_dir = os.path.join(self.processing_dir, 'features', self.features_type)
        self.matches_dir = os.path.join(self.processing_dir, 'matches', self.matcher_type)
        self.matches_vis_dir = os.path.join(self.processing_dir, 'matches_vis', self.features_type)
        self.calibration_dir = os.path.join(self.processing_dir, 'calibrations')
        
        # Output subdirectories
        self.sparse_dir = os.path.join(self.results_dir, 'sparse')
        self.dense_dir = os.path.join(self.results_dir, 'dense')
        self.colmap_dir = os.path.join(self.results_dir, 'colmap')
        self.custom_sfm_dir = os.path.join(self.results_dir, 'custom_sfm')
        self.mvs_dir = os.path.join(self.results_dir, 'mvs')
        self.panorama_dir = os.path.join(self.results_dir, 'panorama')

        # Create necessary directories
        os.makedirs(self.processing_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.matches_dir, exist_ok=True)
        os.makedirs(self.matches_vis_dir, exist_ok=True)
        os.makedirs(self.calibration_dir, exist_ok=True)
        os.makedirs(self.sparse_dir, exist_ok=True)
        os.makedirs(self.dense_dir, exist_ok=True)
        os.makedirs(self.colmap_dir, exist_ok=True)
        os.makedirs(self.custom_sfm_dir, exist_ok=True)
        os.makedirs(self.mvs_dir, exist_ok=True)
        os.makedirs(self.panorama_dir, exist_ok=True)

    def get_calibration_path(self) -> str:
        """Returns the path to the camera calibration file."""
        return os.path.join(self.calibration_dir, f'{self.dataset}_calibration.json')
