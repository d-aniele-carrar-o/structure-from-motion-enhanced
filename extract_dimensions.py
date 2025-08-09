#!/usr/bin/env python3
"""
Standalone script for extracting furniture dimensions from iPhone 15 Pro HEIC files.
"""

import os
import sys
import logging
from argparse import ArgumentParser

from sfm_pipeline.config import Config
from sfm_pipeline.dimension_extractor import DimensionExtractor
from sfm_pipeline.logger import setup_logger


def main():
    parser = ArgumentParser(description='Extract furniture dimensions from iPhone 15 Pro HEIC files with LiDAR data.')
    
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Root directory containing datasets (default: data).')
    parser.add_argument('--dataset', type=str, default='custom',
                        help='Dataset name within the data directory (default: custom).')
    parser.add_argument('--media-dir', type=str, default=None,
                        help='Directory with HEIC files. If not set, defaults to data/{dataset}/media.')
    parser.add_argument('--out-dir', type=str, default='results',
                        help='Root directory to store results (default: results).')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging.')
    
    args = parser.parse_args()
    setup_logger(args.debug)
    
    # Use standardized directory structure
    media_dir = args.media_dir or os.path.join(args.data_dir, args.dataset, 'media')
    
    if not os.path.exists(media_dir):
        logging.error(f"Media directory not found: {media_dir}")
        sys.exit(1)
    
    # Find iPhone Live Photo files (HEIC or JPG/MPO format)
    image_files = []
    for root, dirs, files in os.walk(media_dir):
        for f in files:
            if f.lower().endswith(('.heic', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, f))
    
    if not image_files:
        logging.error(f"No HEIC/JPG files found in {media_dir}")
        sys.exit(1)
    
    # Sort by modification time, get most recent only
    image_files_with_time = [(f, os.path.getmtime(f)) for f in image_files]
    image_files_with_time.sort(key=lambda x: x[1], reverse=True)
    
    # Take only the most recent file
    most_recent = image_files_with_time[0][0]
    image_files = [most_recent]
    
    logging.info(f"Processing most recent file: {os.path.basename(most_recent)}")
    
    # Create config using standardized structure
    config = Config(
        data_dir=args.data_dir,
        dataset=args.dataset,
        media_dir=media_dir,
        out_dir=args.out_dir,
        debug=args.debug,
        clean=False,
        ext=['jpg'],
        frame_interval=20,
        max_frames=100,
        quality_threshold=25.0,
        features_type='SIFT',
        matcher_type='BFMatcher',
        max_features=5000,
        ratio_threshold=0.75,
        sfm_method='custom',
        min_matches=50,
        fund_method=None,
        outlier_threshold=3.0,
        fund_prob=0.99,
        reprojection_threshold=4.0,
        pnp_prob=0.99,
        save_matches_vis=False,
        visualize_3d=False,
        enable_bundle_adjustment=False,
        extract_dimensions=True
    )
    
    # Extract dimensions
    extractor = DimensionExtractor(config)
    image_paths = image_files
    
    results = extractor.extract_dimensions(image_paths)
    
    if results:
        logging.info("=== DIMENSION EXTRACTION RESULTS ===")
        for filename, dimensions in results.items():
            logging.info(f"{os.path.basename(filename)}:")
            logging.info(f"  Width:    {dimensions['width_m']:.3f} m")
            logging.info(f"  Height:   {dimensions['height_m']:.3f} m") 
            logging.info(f"  Depth:    {dimensions['depth_m']:.3f} m")
            logging.info(f"  Distance: {dimensions['distance_m']:.3f} m")
            logging.info("")
        
        logging.info(f"Results saved to: {config.dimensions_dir}")
    else:
        logging.warning("No dimensions could be extracted from the provided image files")
        logging.info("This could be due to:")
        logging.info("  - Image files don't contain LiDAR depth data (need iPhone Live Photos)")
        logging.info("  - No rectangular furniture detected in images")
        logging.info("  - Edge detection unable to find furniture contours")
        logging.info("  - Check debug images in results/custom/processing/depth_maps/")


if __name__ == '__main__':
    main()