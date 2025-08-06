#!/usr/bin/env python3
"""
Complete Structure from Motion Pipeline
Automatically runs feature matching if needed, then performs SFM reconstruction.
"""

import os
import cv2
import numpy as np
from pickle import load
from tqdm import tqdm
from argparse import ArgumentParser
from types import SimpleNamespace

# Import existing functions
from featmatch import FeatMatch
from sfm import SFM
from media_manager import process_media_directory
from utils import DeserializeKeypoints, DeserializeMatches


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
        os.path.join(data_dir, dataset, 'images'),
        os.path.join(data_dir, dataset, 'videos'),
        os.path.join(data_dir, dataset, 'features'),
        os.path.join(data_dir, dataset, 'matches'),
        os.path.join(data_dir, dataset, 'matches_vis'),
        os.path.join(out_dir, dataset)
    ]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            print(f"Cleaned: {dir_path}")

def create_panorama(args):
    """
    Stitches images into a panorama using a more robust method
    to prevent drift and applies blending for a nicer result.
    """
    print("\nStarting robust panorama creation...")
    images_dir = os.path.join(args.data_dir, args.dataset, 'images')
    feat_dir = os.path.join(args.data_dir, args.dataset, 'features', args.features)
    matches_dir = os.path.join(args.data_dir, args.dataset, 'matches', args.matcher)
    
    image_names = sorted([x.split('.')[0] for x in os.listdir(images_dir) if x.split('.')[-1].lower() in args.ext])
    
    if len(image_names) < 2:
        print("Not enough images to create a panorama.")
        return

    # 1. Load all features
    print("  Loading features...")
    features = {}
    for name in image_names:
        kp_path = os.path.join(feat_dir, f'kp_{name}.pkl')
        desc_path = os.path.join(feat_dir, f'desc_{name}.pkl')
        with open(kp_path, 'rb') as f: kp = load(f)
        with open(desc_path, 'rb') as f: desc = load(f)
        features[name] = (DeserializeKeypoints(kp), desc)

    # 2. Calculate all cumulative homographies relative to the first image
    H_cumulative = np.identity(3)
    # This list will store the raw image and the homography that maps it to the first image's coordinate system
    images_to_warp = []
    
    # Add the first image (the reference frame)
    base_name = image_names[0]
    base_img_path = os.path.join(images_dir, f"{base_name}.{args.ext[0]}")
    base_img = cv2.imread(base_img_path)
    images_to_warp.append({'img': base_img, 'name': base_name, 'H': np.identity(3)})

    for i in tqdm(range(1, len(image_names)), desc="Processing images"):        
        prev_name = image_names[i-1]
        curr_name = image_names[i]
        
        # print(f"  Calculating transform for {curr_name} -> {base_name}...")
        
        kp1, _ = features[prev_name]
        kp2, _ = features[curr_name]
        
        match_path = os.path.join(matches_dir, f'match_{prev_name}_{curr_name}.pkl')
        if not os.path.exists(match_path):
            print(f"    Match file not found: {match_path}. Halting panorama creation.")
            return

        with open(match_path, 'rb') as f:
            matches = DeserializeMatches(load(f))
        
        if len(matches) < args.min_matches:
            # print(f"    Too few matches between {prev_name} and {curr_name} ({len(matches)}). Halting.")
            return

        pts1 = np.float32([kp.pt for kp in kp1])[np.array([m.queryIdx for m in matches])]
        pts2 = np.float32([kp.pt for kp in kp2])[np.array([m.trainIdx for m in matches])]
        
        # Homography to map current image to *previous* image
        H_relative, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        if H_relative is None:
            print(f"    Could not find homography for {curr_name}. Halting.")
            return

        # Accumulate the homography to map current image to the *base* image
        H_cumulative = H_cumulative @ H_relative
        
        curr_img_path = os.path.join(images_dir, f"{curr_name}.{args.ext[0]}")
        curr_img = cv2.imread(curr_img_path)
        images_to_warp.append({'img': curr_img, 'name': curr_name, 'H': H_cumulative})

    # 3. Determine the final canvas size by transforming all image corners
    print("  Determining final canvas size...")
    all_corners = []
    for item in images_to_warp:
        h, w = item['img'].shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, item['H'])
        all_corners.append(warped_corners)

    all_corners = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    
    # 4. Create a translation matrix to move the panorama to the top-left corner
    H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    output_width = x_max - x_min
    output_height = y_max - y_min
    
    # 5. Warp all images onto the final canvas and blend them
    print("  Warping and blending images...")
    # Use float for accumulation to allow for averaging
    panorama = np.zeros((output_height, output_width, 3), np.float32)
    # Keep track of how many images contribute to each pixel for averaging
    pixel_counts = np.zeros((output_height, output_width, 3), np.float32)

    for item in tqdm(images_to_warp, desc="Warping images"):
        H_final = H_translation @ item['H']
        
        # Warp the image to the final canvas size
        warped_img = cv2.warpPerspective(item['img'], H_final, (output_width, output_height)).astype(np.float32)
        
        # Create a mask of where the warped image has content (is not black)
        mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) > 0
        
        # Add the warped image to the panorama accumulator
        panorama += warped_img
        # Add the mask (converted to float) to the pixel counter. We need 3 channels for color.
        pixel_counts += cv2.merge([mask.astype(np.float32)] * 3)

    # Avoid division by zero for pixels that have no contribution
    pixel_counts[pixel_counts == 0] = 1.0
    
    # Average the pixel values by dividing the sum by the count
    panorama = (panorama / pixel_counts).astype(np.uint8)

    # Save the final panorama
    panorama_out_dir = os.path.join(args.out_dir, args.dataset)
    if not os.path.exists(panorama_out_dir):
        os.makedirs(panorama_out_dir)
    
    panorama_path = os.path.join(panorama_out_dir, 'panorama_robust.jpg')
    cv2.imwrite(panorama_path, panorama)
    print(f"\n✅ Robust panorama saved to: {panorama_path}")

def run_pipeline(args):
    """Run complete SFM pipeline"""
    
    if args.clean:
        print("Cleaning previous pipeline files...")
        clean_pipeline_files(args.data_dir, args.dataset, args.out_dir)
        print("Clean completed.\n")
    
    # Check if images directory exists and has images
    images_dir = os.path.join(args.data_dir, args.dataset, 'images')
    need_media_processing = False
    
    if not os.path.exists(images_dir):
        need_media_processing = True
    else:
        image_files = [f for f in os.listdir(images_dir) if f.split('.')[-1].lower() in args.ext]
        if len(image_files) == 0:
            need_media_processing = True
    
    # Auto-detect or use specified media directory if processing needed
    if need_media_processing or args.media_dir:
        media_dir = args.media_dir
        if not media_dir:
            # Check default media directory
            default_media_dir = os.path.join(args.data_dir, args.dataset, 'media')
            if os.path.exists(default_media_dir):
                media_dir = default_media_dir
        
        if media_dir:
            print(f"Processing media directory: {media_dir}")
            process_media_directory(
                media_dir, args.data_dir, args.dataset,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
                quality_threshold=args.quality_threshold
            )
            print("Media processing completed.\n")
        else:
            print(f"No images found and no media directory available.")
            print("Please add media files or run:")
            print(f"  python setup_dataset.py --dataset {args.dataset}")
            print(f"  # Then add media files to data/{args.dataset}/media/")
            return
    
    # Verify we now have images
    if not os.path.exists(images_dir):
        print(f"Images directory still missing: {images_dir}")
        print("Media processing may have failed. Check the error messages above.")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.split('.')[-1].lower() in args.ext]
    if len(image_files) == 0:
        print(f"No images found after processing: {images_dir}")
        print("\nTroubleshooting:")
        print("- For .MOV files: Install ffmpeg (brew install ffmpeg on macOS)")
        print("- For .HEIC files: Install pillow-heif (pip install pillow-heif)")
        print("- Check that your media files are valid and not corrupted")
        return
    
    print(f"Found {len(image_files)} images in {images_dir}")
    
    # Check if features/matches exist
    if not check_features_exist(args.data_dir, args.dataset, args.features, args.matcher, args.ext):
        print("Features/matches not found. Running feature matching first...")
        
        # Create featmatch options
        featmatch_opts = SimpleNamespace()
        featmatch_opts.data_dir = images_dir
        featmatch_opts.out_dir = os.path.join(args.data_dir, args.dataset)
        featmatch_opts.ext = args.ext
        featmatch_opts.features = args.features
        featmatch_opts.matcher = args.matcher
        featmatch_opts.cross_check = args.cross_check
        featmatch_opts.max_features = args.max_features
        featmatch_opts.ratio_threshold = args.ratio_threshold
        featmatch_opts.min_matches = args.min_matches
        featmatch_opts.save_matches_vis = args.save_matches_vis

        # Pass the geometric verification parameters from the main arguments
        featmatch_opts.fund_method = args.fund_method
        featmatch_opts.outlier_thres = args.outlier_thres
        featmatch_opts.fund_prob = args.fund_prob

        featmatch_opts.print_every = 1
        featmatch_opts.save_results = False
        
        # Run feature matching (pass empty list to use directory scanning)
        FeatMatch(featmatch_opts)
        print("Feature matching completed.\n")
    else:
        print("Features/matches found. Skipping feature matching.\n")
    
    if args.create_panorama:
        create_panorama(args)
    
    # Run SFM
    print("Starting SFM reconstruction...")
    sfm = SFM(args)
    sfm.Run()

def main():
    parser = ArgumentParser(description='Complete Structure from Motion Pipeline')
    
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
    parser.add_argument('--max-features', type=int, default=5000,
                       help='Maximum features to extract per image (default: 5000)')
    parser.add_argument('--ratio-threshold', type=float, default=0.75,
                       help='Lowe ratio test threshold (default: 0.75)')
    parser.add_argument('--min-matches', type=int, default=50,
                       help='Minimum matches required between images (default: 50)')
    parser.add_argument('--save-matches-vis', action='store_true', default=True,
                       help='Save images with feature matches drawn on them')
    parser.add_argument('--create-panorama', action='store_true', default=True,
                       help='Create a panorama visualization from the images')
    
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
    
    # Media processing arguments
    parser.add_argument('--media-dir', type=str, default=None,
                       help='Directory containing media files (HEIC, MOV, MP4, etc.)')
    parser.add_argument('--frame-interval', type=int, default=20,
                       help='Extract every N frames from video (default: 5)')
    parser.add_argument('--max-frames', type=int, default=100,
                       help='Maximum frames to extract from video (default: 100)')
    parser.add_argument('--quality-threshold', type=float, default=25,
                       help='Minimum quality threshold for frame extraction (default: 30)')
    
    # Clean argument
    parser.add_argument('--clean', action='store_true', default=False,
                       help='Clean all generated files and reprocess media from scratch')
    
    args = parser.parse_args()
    
    # Process arguments
    args.ext = args.ext.split(',')
    args.fund_method = getattr(__import__('cv2'), args.fund_method)
    args.calibration_mat = 'auto'  # Always use automatic calibration from HEIC metadata
    
    # Run pipeline
    run_pipeline(args)


if __name__ == '__main__':
    main()
