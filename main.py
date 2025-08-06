import sys
import os
import cv2
import logging
from argparse import ArgumentParser

from sfm_pipeline.config import Config
from sfm_pipeline.pipeline import Pipeline
from sfm_pipeline.logger import setup_logger


def _get_cv2_method(method_name: str):
    """Safely get OpenCV method with whitelist validation."""
    VALID_METHODS = {
        'FM_RANSAC': cv2.FM_RANSAC,
        'FM_LMEDS': cv2.FM_LMEDS,
        'FM_7POINT': cv2.FM_7POINT,
        'FM_8POINT': cv2.FM_8POINT
    }
    if method_name not in VALID_METHODS:
        raise ValueError(f"Invalid method: {method_name}. Valid options: {list(VALID_METHODS.keys())}")
    return VALID_METHODS[method_name]

def main():
    """
    Main entry point for the Structure from Motion (SfM) pipeline.
    """
    parser = ArgumentParser(description='Complete Structure from Motion (SfM) Pipeline.')

    # --- Crucial Flags ---
    parser.add_argument('--debug', action='store_true',
                        help='Enable verbose debug logging for detailed pipeline performance analysis.')
    parser.add_argument('--clean', action='store_true', default=False,
                        help='Clean all generated files and reprocess media from scratch.')

    # --- Directory Arguments ---
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root directory containing datasets (default: data).')
    parser.add_argument('--dataset', type=str, default='custom',
                        help='Dataset name within the data directory (default: custom).')
    parser.add_argument('--media-dir', type=str, default=None,
                        help='Directory with original media (HEIC, MOV). If not set, defaults to data/{dataset}/media.')
    parser.add_argument('--out-dir', type=str, default='results',
                        help='Root directory to store final results like point clouds (default: results).')

    # --- Media Processing Arguments ---
    parser.add_argument('--frame-interval', type=int, default=20,
                        help='Extract every Nth frame from videos (default: 20).')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Maximum number of frames to extract from each video (default: 100).')
    parser.add_argument('--quality-threshold', type=float, default=25.0,
                        help='Minimum blur score (Laplacian variance) to accept a frame (default: 25.0).')

    # --- Feature Matching Arguments ---
    parser.add_argument('--features-type', type=str, default='SIFT', choices=['SIFT'],
                        help='Feature detection algorithm (default: SIFT).')
    parser.add_argument('--matcher-type', type=str, default='BFMatcher', choices=['BFMatcher', 'FlannBasedMatcher'],
                        help='Feature matching algorithm (default: BFMatcher).')
    parser.add_argument('--max-features', type=int, default=5000,
                        help='Maximum features to extract per image (default: 5000).')
    parser.add_argument('--ratio-threshold', type=float, default=0.75,
                        help="Lowe's ratio test threshold for filtering KNN matches (default: 0.75).")

    # --- Geometric Verification & SFM Arguments ---
    parser.add_argument('--min-matches', type=int, default=50,
                        help='Minimum number of geometrically verified matches required between an image pair (default: 50).')
    parser.add_argument('--fund-method', type=str, default='FM_RANSAC',
                        help='Method for Fundamental Matrix estimation (e.g., FM_RANSAC, FM_LMEDS) (default: FM_RANSAC).')
    parser.add_argument('--outlier-threshold', type=float, default=3.0,
                        help='RANSAC outlier threshold in pixels for Fundamental Matrix estimation (default: 3.0).')
    parser.add_argument('--fund-prob', type=float, default=0.99,
                        help='Confidence level for Fundamental Matrix estimation (default: 0.99).')
    parser.add_argument('--reprojection-threshold', type=float, default=4.0,
                        help='Reprojection error threshold in pixels for PnP pose estimation (default: 4.0).')
    parser.add_argument('--pnp-prob', type=float, default=0.99,
                        help='Confidence level for PnP pose estimation (default: 0.99).')

    # --- Output & Visualization ---
    parser.add_argument('--save-matches-vis', action='store_true', default=True,
                        help='Save images visualizing feature matches after geometric verification.')
    parser.add_argument('--visualize-3d', action='store_true', default=False,
                        help='Display an interactive 3D visualization of the final point cloud.')

    try:
        args = parser.parse_args()
        setup_logger(args.debug)

        # --- Create Configuration ---
        # Resolve the default media directory if not provided
        media_dir = args.media_dir or os.path.join(args.data_dir, args.dataset, 'media')
        
        config = Config(
            data_dir=args.data_dir,
            dataset=args.dataset,
            media_dir=media_dir,
            out_dir=args.out_dir,
            debug=args.debug,
            clean=args.clean,
            ext=['jpg', 'jpeg', 'png'],
            # Media processing
            frame_interval=args.frame_interval,
            max_frames=args.max_frames,
            quality_threshold=args.quality_threshold,
            # Feature matching
            features_type=args.features_type,
            matcher_type=args.matcher_type,
            max_features=args.max_features,
            ratio_threshold=args.ratio_threshold,
            # SFM
            min_matches=args.min_matches,
            fund_method=_get_cv2_method(args.fund_method),
            outlier_threshold=args.outlier_threshold,
            fund_prob=args.fund_prob,
            reprojection_threshold=args.reprojection_threshold,
            pnp_prob=args.pnp_prob,
            # Visualization
            save_matches_vis=args.save_matches_vis,
            visualize_3d=args.visualize_3d,
        )

        logging.info("Configuration loaded. Starting SfM pipeline.")
        if config.debug:
            logging.debug(f"Full configuration:\n{config}")

        # --- Run Pipeline ---
        pipeline = Pipeline(config)
        pipeline.run()

        logging.info("SfM pipeline finished successfully.")

    except AttributeError as e:
        logging.error(f"Invalid OpenCV method name specified. Please check your arguments. Error: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred in the pipeline: {e}")
        if 'args' in locals() and args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
