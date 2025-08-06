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
    min_matches: int
    fund_method: Any  # cv2 method like cv2.FM_RANSAC
    outlier_threshold: float
    fund_prob: float
    reprojection_threshold: float
    pnp_prob: float
    
    # --- Output & Visualization ---
    save_matches_vis: bool
    visualize_3d: bool

    # --- Derived Paths (auto-generated) ---
    dataset_dir: str = field(init=False)
    images_dir: str = field(init=False)
    videos_dir: str = field(init=False)
    features_dir: str = field(init=False)
    matches_dir: str = field(init=False)
    matches_vis_dir: str = field(init=False)
    calibration_dir: str = field(init=False)
    results_dir: str = field(init=False)

    def __post_init__(self):
        """Generate derived paths and create directories."""
        self.dataset_dir = os.path.join(self.data_dir, self.dataset)
        self.images_dir = os.path.join(self.dataset_dir, 'images')
        self.videos_dir = os.path.join(self.dataset_dir, 'videos')
        self.features_dir = os.path.join(self.dataset_dir, 'features', self.features_type)
        self.matches_dir = os.path.join(self.dataset_dir, 'matches', self.matcher_type)
        self.matches_vis_dir = os.path.join(self.dataset_dir, 'matches_vis', self.features_type)
        self.calibration_dir = os.path.join(self.dataset_dir, 'calibrations')
        self.results_dir = os.path.join(self.out_dir, self.dataset)

        # Create necessary directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.matches_dir, exist_ok=True)
        os.makedirs(self.matches_vis_dir, exist_ok=True)
        os.makedirs(self.calibration_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def get_calibration_path(self) -> str:
        """Returns the path to the camera calibration file."""
        return os.path.join(self.calibration_dir, f'{self.dataset}_calibration.json')
