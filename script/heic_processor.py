#!/usr/bin/env python3
"""
HEIC Image Processor for SFM Pipeline
Extracts camera parameters from HEIC metadata and converts images to JPG
"""

import os
import json
import numpy as np
import argparse
import subprocess
import sys

try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    from PIL.ExifTags import TAGS
    register_heif_opener()
    HEIF_SUPPORT = True
except ImportError:
    try:
        import pyheif
        from PIL import Image
        from PIL.ExifTags import TAGS
        HEIF_SUPPORT = True
    except ImportError:
        HEIF_SUPPORT = False
        print("Warning: No HEIF support found. Using system tools for conversion...")

def extract_with_exiftool(heic_path):
    """Extract metadata using exiftool as fallback"""
    try:
        result = subprocess.run(['exiftool', '-j', heic_path], 
                              capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)[0]
        
        width = metadata.get('ImageWidth')
        height = metadata.get('ImageHeight')
        focal_length_mm = metadata.get('FocalLength')
        focal_length_35mm = metadata.get('FocalLengthIn35mmFormat')
        
        # Clean up focal length values
        if isinstance(focal_length_mm, str):
            focal_length_mm = float(focal_length_mm.replace('mm', '').strip())
        
        if isinstance(focal_length_35mm, str):
            focal_length_35mm = float(focal_length_35mm.replace('mm', '').strip())
        
        return {
            'width': width,
            'height': height,
            'focal_length_mm': focal_length_mm,
            'focal_length_35mm': focal_length_35mm,
            'metadata': metadata
        }
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Exiftool extraction failed: {e}")
        return None

def extract_camera_params_from_heic(heic_path):
    """Extract camera parameters from HEIC metadata"""
    width, height = None, None
    focal_length_mm, focal_length_35mm = None, None
    
    if HEIF_SUPPORT:
        try:
            with Image.open(heic_path) as img:
                width, height = img.size
                exif = img.getexif()
                
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    
                    if tag == 'FocalLength':
                        if isinstance(value, tuple):
                            focal_length_mm = value[0] / value[1]
                        else:
                            focal_length_mm = float(value)
                            
                    elif tag == 'FocalLengthIn35mmFilm':
                        focal_length_35mm = float(value)
                        
        except Exception as e:
            print(f"PIL extraction failed: {e}")
    
    # Fallback to exiftool
    if not all([width, height, focal_length_mm]):
        print("Trying exiftool for metadata extraction...")
        exif_data = extract_with_exiftool(heic_path)
        if exif_data:
            width = width or exif_data['width']
            height = height or exif_data['height']
            focal_length_mm = focal_length_mm or exif_data['focal_length_mm']
            focal_length_35mm = focal_length_35mm or exif_data['focal_length_35mm']
    
    if not all([width, height, focal_length_mm]):
        print("Could not extract required metadata. Using iPhone defaults.")
        # iPhone defaults for recent models
        width, height = 4032, 3024  # Common iPhone resolution
        focal_length_mm = 4.25  # iPhone main camera approximate
        focal_length_35mm = 26   # iPhone main camera 35mm equivalent
    
    # iPhone sensor dimensions (approximate)
    sensor_width_mm = 6.0  # Modern iPhone sensor width
    
    if focal_length_mm and focal_length_35mm:
        try:
            # Calculate actual sensor width from 35mm equivalent
            actual_sensor_width = (float(focal_length_mm) * 36.0) / float(focal_length_35mm)
            sensor_width_mm = actual_sensor_width
        except (ValueError, TypeError):
            print(f"Warning: Could not calculate sensor width from focal lengths: {focal_length_mm}, {focal_length_35mm}")
    
    # Calculate focal length in pixels
    fx = (focal_length_mm * width) / sensor_width_mm
    fy = fx  # Assume square pixels
    
    # Principal point (center of image)
    cx = width / 2.0
    cy = height / 2.0
    
    # Create camera matrix
    camera_matrix = [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ]
    
    return {
        'camera_matrix': camera_matrix,
        'image_size': [width, height],
        'focal_length_mm': focal_length_mm,
        'focal_length_35mm': focal_length_35mm,
        'sensor_width_mm': sensor_width_mm,
        'fx': fx,
        'fy': fy,
        'cx': cx,
        'cy': cy
    }

def convert_heic_to_jpg(heic_path, jpg_path, quality=95):
    """Convert HEIC to JPG"""
    if HEIF_SUPPORT:
        try:
            with Image.open(heic_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(jpg_path, 'JPEG', quality=quality)
            return True
        except Exception as e:
            print(f"PIL conversion failed: {e}")
    
    # Fallback to sips (macOS built-in)
    try:
        result = subprocess.run(['sips', '-s', 'format', 'jpeg', heic_path, '--out', jpg_path],
                              capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            # Try ImageMagick convert
            result = subprocess.run(['convert', heic_path, jpg_path],
                                  capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Failed to convert {heic_path}. Install pillow-heif, ImageMagick, or use macOS sips.")
            return False

def process_heic_images(data_dir, dataset='custom'):
    """Process all HEIC images in the dataset directory"""
    
    # Handle relative paths from project root
    if not os.path.isabs(data_dir):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, data_dir.lstrip('../'))
    
    images_dir = os.path.join(data_dir, dataset, 'images')
    
    if not os.path.exists(images_dir):
        print(f"Images directory not found: {images_dir}")
        return
    
    # Find all HEIC files
    heic_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.heic')]
    
    if not heic_files:
        print("No HEIC files found in the images directory")
        return
    
    print(f"Found {len(heic_files)} HEIC files")
    
    # Extract camera parameters from the first image (assuming all from same camera)
    first_heic = os.path.join(images_dir, heic_files[0])
    camera_params = extract_camera_params_from_heic(first_heic)
    
    if not camera_params:
        print("Failed to extract camera parameters")
        return
    
    # Create calibrations directory in script folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    calib_dir = os.path.join(script_dir, 'calibrations')
    os.makedirs(calib_dir, exist_ok=True)
    
    # Save camera parameters
    calib_file = os.path.join(calib_dir, 'latest_calibration.json')
    with open(calib_file, 'w') as f:
        json.dump(camera_params, f, indent=2)
    
    print(f"Camera parameters saved to: {calib_file}")
    print(f"Camera matrix:")
    print(f"  fx: {camera_params['fx']:.1f}")
    print(f"  fy: {camera_params['fy']:.1f}")
    print(f"  cx: {camera_params['cx']:.1f}")
    print(f"  cy: {camera_params['cy']:.1f}")
    print(f"  Image size: {camera_params['image_size']}")
    
    # Convert all HEIC files to JPG
    converted_count = 0
    for heic_file in heic_files:
        heic_path = os.path.join(images_dir, heic_file)
        jpg_file = heic_file.rsplit('.', 1)[0] + '.jpg'
        jpg_path = os.path.join(images_dir, jpg_file)
        
        print(f"Converting {heic_file} -> {jpg_file}")
        if convert_heic_to_jpg(heic_path, jpg_path):
            converted_count += 1
        else:
            print(f"Failed to convert {heic_file}")
    
    print(f"\nConversion complete: {converted_count}/{len(heic_files)} files converted")
    
    if converted_count > 0:
        print(f"You can now run SFM with: python script/sfm.py --calibration-mat iphone --ext jpg")
        print(f"Or run feature matching first: python script/featmatch.py --ext jpg")
    else:
        print("\nTo install HEIC support, try:")
        print("  pip install pillow-heif")
        print("Or on macOS, the script will use built-in 'sips' command")

def main():
    parser = argparse.ArgumentParser(description='Process HEIC images for SFM pipeline')
    parser.add_argument('--data-dir', default='../data/', help='Root data directory')
    parser.add_argument('--dataset', default='custom', help='Dataset name')
    
    args = parser.parse_args()
    
    process_heic_images(args.data_dir, args.dataset)

if __name__ == '__main__':
    main()