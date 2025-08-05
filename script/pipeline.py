#!/usr/bin/env python3
"""
Complete Structure from Motion Pipeline
Automatically runs feature matching if needed, then performs SFM reconstruction.
"""

import os
import argparse
from types import SimpleNamespace

# Import existing functions
from featmatch import FeatMatch
from sfm import SFM
from heic_processor import process_heic_images


def check_features_exist(data_dir, dataset, features, matcher, ext):
    """Check if feature and match files exist for all images"""
    images_dir = os.path.join(data_dir, dataset, 'images')
    feat_dir = os.path.join(data_dir, dataset, 'features', features)
    matches_dir = os.path.join(data_dir, dataset, 'matches', matcher)
    
    if not (os.path.exists(feat_dir) and os.path.exists(matches_dir)):
        return False
    
    # Count expected images
    if os.path.exists(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.split('.')[-1].lower() in ext]
        expected_count = len(image_files)
        
        # Check if we have features for all images
        feat_files = [f for f in os.listdir(feat_dir) if f.startswith('kp_')]
        
        return len(feat_files) >= expected_count and len(os.listdir(matches_dir)) > 0
    
    return False


def clean_pipeline_files(data_dir, dataset, out_dir):
    """Clean all pipeline generated files"""
    import shutil
    
    dirs_to_clean = [
        os.path.join(data_dir, dataset, 'features'),
        os.path.join(data_dir, dataset, 'matches'),
        os.path.join(out_dir, dataset)
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Cleaned: {dir_path}")

def run_pipeline(args):
    """Run complete SFM pipeline"""
    
    if args.clean:
        print("Cleaning previous pipeline files...")
        clean_pipeline_files(args.data_dir, args.dataset, args.out_dir)
        print("Clean completed.\n")
    
    # Check for HEIC files and process them first
    images_dir = os.path.join(args.data_dir, args.dataset, 'images')
    if os.path.exists(images_dir):
        heic_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.heic')]
        if heic_files:
            print(f"Found {len(heic_files)} HEIC files. Processing for camera calibration...")
            process_heic_images(args.data_dir, args.dataset)
            print("HEIC processing completed.\n")
    
    # Check if features/matches exist
    if not check_features_exist(args.data_dir, args.dataset, args.features, args.matcher, args.ext):
        print("Features/matches not found. Running feature matching first...")
        
        # Create featmatch options
        featmatch_opts = SimpleNamespace()
        featmatch_opts.data_dir = os.path.join(args.data_dir, args.dataset, 'images')
        featmatch_opts.out_dir = os.path.join(args.data_dir, args.dataset)
        featmatch_opts.ext = args.ext
        featmatch_opts.features = args.features
        featmatch_opts.matcher = args.matcher
        featmatch_opts.cross_check = args.cross_check
        featmatch_opts.print_every = 1
        featmatch_opts.save_results = False
        
        # Run feature matching (pass empty list to use directory scanning)
        FeatMatch(featmatch_opts, [])
        print("Feature matching completed.\n")
    else:
        print("Features/matches found. Skipping feature matching.\n")
    
    # Run SFM
    print("Starting SFM reconstruction...")
    sfm = SFM(args)
    sfm.Run()


def main():
    parser = argparse.ArgumentParser(description='Complete Structure from Motion Pipeline')
    
    # Directory arguments
    parser.add_argument('--data-dir', type=str, default='data', 
                       help='Root directory containing datasets (default: data)')
    parser.add_argument('--dataset', type=str, default='custom',
                       help='Dataset name (default: custom)')
    parser.add_argument('--out-dir', type=str, default='results',
                       help='Output directory (default: results)')
    parser.add_argument('--ext', type=str, default='jpg,png',
                       help='Image extensions (default: jpg,png)')
    
    # Feature/matching arguments
    parser.add_argument('--features', type=str, default='SIFT',
                       help='Feature algorithm (default: SIFT)')
    parser.add_argument('--matcher', type=str, default='BFMatcher',
                       help='Matching algorithm (default: BFMatcher)')
    parser.add_argument('--cross-check', action='store_true', default=True,
                       help='Use cross-check matching')
    
    # SFM arguments (calibration is now automatic from HEIC metadata)
    parser.add_argument('--fund-method', type=str, default='FM_RANSAC',
                       help='Fundamental matrix method (default: FM_RANSAC)')
    parser.add_argument('--outlier-thres', type=float, default=3.0,
                       help='Outlier threshold in pixels (default: 3.0)')
    parser.add_argument('--fund-prob', type=float, default=0.99,
                       help='Fundamental matrix confidence (default: 0.99)')
    parser.add_argument('--pnp-method', type=str, default='SOLVEPNP_DLS',
                       help='PnP method (default: SOLVEPNP_DLS)')
    parser.add_argument('--pnp-prob', type=float, default=0.99,
                       help='PnP confidence (default: 0.99)')
    parser.add_argument('--reprojection-thres', type=float, default=4.0,
                       help='Reprojection threshold in pixels (default: 4.0)')
    
    # Visualization arguments
    parser.add_argument('--visualize-3d', action='store_true', default=False,
                       help='Create 3D visualization')
    parser.add_argument('--plot-error', action='store_true', default=False,
                       help='Plot reprojection errors')
    
    # Clean argument
    parser.add_argument('--clean', action='store_true', default=False,
                       help='Clean all previous pipeline files before running')
    
    args = parser.parse_args()
    
    # Process arguments
    args.ext = args.ext.split(',')
    args.fund_method = getattr(__import__('cv2'), args.fund_method)
    args.calibration_mat = 'auto'  # Always use automatic calibration from HEIC metadata
    
    # Run pipeline
    run_pipeline(args)


if __name__ == '__main__':
    main()