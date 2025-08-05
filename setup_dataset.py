#!/usr/bin/env python3
"""
Setup Dataset Directory Structure
Creates the proper directory structure for SFM pipeline
"""

import os
import argparse

def setup_dataset(data_dir='data', dataset='custom'):
    """Create the standard dataset directory structure"""
    
    # Create main dataset directory
    dataset_dir = os.path.join(data_dir, dataset)
    
    # Create subdirectories
    directories = [
        os.path.join(dataset_dir, 'media'),      # Raw media files (.MOV, .HEIC, etc.)
        os.path.join(dataset_dir, 'images'),     # Processed images (auto-generated)
        os.path.join(dataset_dir, 'videos'),     # Converted videos (auto-generated)
        os.path.join(dataset_dir, 'features'),   # Feature files (auto-generated)
        os.path.join(dataset_dir, 'matches'),    # Match files (auto-generated)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    # Create README in media directory
    readme_content = """# Media Directory

Place your raw media files here:
- iPhone .HEIC photos
- iPhone .MOV videos  
- .MP4 videos
- .JPG/.PNG images

The pipeline will automatically:
1. Convert HEIC → JPG (with camera calibration extraction)
2. Convert MOV → MP4 → extract frames
3. Process all media into the images/ directory
4. Run feature matching and SFM reconstruction

Usage:
  python script/pipeline.py --dataset {dataset} --visualize-3d
""".format(dataset=dataset)
    
    readme_path = os.path.join(dataset_dir, 'media', 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"\nDataset '{dataset}' structure created in '{data_dir}'")
    print(f"📁 Place your iPhone media files in: {os.path.join(dataset_dir, 'media')}")
    print(f"🚀 Then run: python script/pipeline.py --dataset {dataset} --visualize-3d")

def main():
    parser = argparse.ArgumentParser(description='Setup dataset directory structure')
    parser.add_argument('--data-dir', default='data', help='Root data directory (default: data)')
    parser.add_argument('--dataset', default='custom', help='Dataset name (default: custom)')
    
    args = parser.parse_args()
    setup_dataset(args.data_dir, args.dataset)

if __name__ == '__main__':
    main()