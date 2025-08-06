#!/usr/bin/env python3
"""
Comprehensive Media Manager for SFM Pipeline
Handles HEIC images, video frame extraction, and iPhone .MOV conversion
"""

import os
import cv2
import json
import argparse
import subprocess
from tqdm import tqdm

# HEIC support
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
                        focal_length_mm = value[0] / value[1] if isinstance(value, tuple) else float(value)
                    elif tag == 'FocalLengthIn35mmFilm':
                        focal_length_35mm = float(value)
        except Exception as e:
            print(f"PIL extraction failed: {e}")
    
    # Fallback to exiftool
    if not all([width, height, focal_length_mm]):
        try:
            result = subprocess.run(['exiftool', '-j', heic_path], 
                                  capture_output=True, text=True, check=True)
            metadata = json.loads(result.stdout)[0]
            width = width or metadata.get('ImageWidth')
            height = height or metadata.get('ImageHeight')
            focal_length_mm = focal_length_mm or float(str(metadata.get('FocalLength', '')).replace('mm', '').strip())
            focal_length_35mm = focal_length_35mm or float(str(metadata.get('FocalLengthIn35mmFormat', '')).replace('mm', '').strip())
        except:
            pass
    
    if not all([width, height, focal_length_mm]):
        print(f"ERROR: Failed to extract required metadata from HEIC file: {heic_path}")
        return None
    
    sensor_width_mm = 6.0
    if focal_length_mm and focal_length_35mm:
        sensor_width_mm = (float(focal_length_mm) * 36.0) / float(focal_length_35mm)
    
    fx = (focal_length_mm * width) / sensor_width_mm
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    
    return {
        'camera_matrix': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        'image_size': [width, height],
        'focal_length_mm': focal_length_mm,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
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
        except Exception:
            pass
    
    # Fallback to system tools
    try:
        subprocess.run(['sips', '-s', 'format', 'jpeg', heic_path, '--out', jpg_path], 
                      capture_output=True, check=True)
        return True
    except:
        try:
            subprocess.run(['convert', heic_path, jpg_path], capture_output=True, check=True)
            return True
        except:
            return False


def convert_mov_to_mp4(mov_path, mp4_path):
    """Convert iPhone .MOV to .MP4 for better compatibility"""
    try:
        cmd = [
            'ffmpeg', '-i', mov_path, '-c:v', 'libx264', '-c:a', 'aac',
            '-preset', 'medium', '-crf', '23', '-y', mp4_path
        ]
        subprocess.run(cmd, capture_output=True, check=True, text=True)
        return True
    except Exception as e:
        print(f"    Conversion error: {e}")
        return False


def calculate_blur_score(image):
    """Calculate blur score using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_video_camera_params(video_path):
    """Extract camera parameters from video metadata - ALWAYS from original file"""
    if not os.path.exists(video_path):
        return None
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    focal_length_mm, focal_length_35mm = None, None
    
    try:
        result = subprocess.run(['exiftool', '-j', video_path], 
                              capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)[0]
        focal_length_35mm = metadata.get('FocalLengthIn35mmFormat')
        if focal_length_35mm:
            focal_length_35mm = float(str(focal_length_35mm).replace('mm', '').strip())
        
        # More robust focal length extraction for different camera modules
        if 'LensModel' in metadata:
             import re
             match = re.search(r'(\d+\.\d+)mm', metadata['LensModel'])
             if match:
                 focal_length_mm = float(match.group(1))

    except Exception:
        pass
    
    if not focal_length_mm:
        print("  WARNING: Could not extract focal length from exiftool. Check video metadata.")
        return None

    if not focal_length_35mm:
        print("  WARNING: Missing 35mm equivalent focal length. Cannot accurately calculate sensor width.")
        return None
    
    sensor_width_mm = (focal_length_mm * 36.0) / focal_length_35mm
    
    fx = (focal_length_mm * width) / sensor_width_mm
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    
    print(f"  SUCCESS: Extracted focal length: {focal_length_mm}mm, 35mm equiv: {focal_length_35mm}mm")
    print(f"  Calculated camera matrix: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    return {
        'camera_matrix': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
        'image_size': [width, height],
        'focal_length_mm': focal_length_mm,
        'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy
    }

def extract_frames_from_video(video_path, output_dir, frame_interval, max_frames, quality_threshold):
    """Extract frames from video with optimal spacing for SFM"""
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {total_frames} frames, {fps:.1f} fps, extracting every {frame_interval} frames")
    
    extracted_count = 0
    frame_count = 0
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        while cap.isOpened() and extracted_count < max_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                blur_score = calculate_blur_score(frame)
                
                if blur_score > quality_threshold:
                    frame_filename = f"frame_{extracted_count:04d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

    return extracted_count

def process_media_directory(input_dir, output_dir, dataset='custom', **kwargs):
    """Process all media files in a directory"""
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    # Setup output structure
    images_dir = os.path.join(output_dir, dataset, 'images')
    videos_dir = os.path.join(output_dir, dataset, 'videos')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    # Find all media files
    media_files = []
    for ext in ['.heic', '.mov', '.mp4', '.avi', '.jpg', '.jpeg', '.png']:
        media_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not media_files:
        print("No media files found")
        return
    
    print(f"Found {len(media_files)} media files")
    
    # Process files by type
    heic_files = [f for f in media_files if f.lower().endswith('.heic')]
    video_files = [f for f in media_files if f.lower().endswith(('.mov', '.mp4', '.avi'))]
    image_files = [f for f in media_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    camera_params = None
    
    # Process HEIC files
    if heic_files:
        print(f"\nProcessing {len(heic_files)} HEIC files...")
        first_heic = os.path.join(input_dir, heic_files[0])
        camera_params = extract_camera_params_from_heic(first_heic)
        
        converted = 0
        for heic_file in heic_files:
            heic_path = os.path.join(input_dir, heic_file)
            jpg_file = heic_file.rsplit('.', 1)[0] + '.jpg'
            jpg_path = os.path.join(images_dir, jpg_file)
            
            if convert_heic_to_jpg(heic_path, jpg_path):
                converted += 1
                print(f"  Converted: {heic_file} -> {jpg_file}")
        
        print(f"Converted {converted}/{len(heic_files)} HEIC files")
    
    # Process video files
    if video_files:
        print(f"\nProcessing {len(video_files)} video files...")
        for video_file in video_files:
            original_video_path = os.path.join(input_dir, video_file)  # ALWAYS use original path
            
            # STEP 1: ALWAYS extract camera parameters from ORIGINAL file FIRST
            print(f"  Extracting camera parameters from ORIGINAL file: {video_file}")
            video_camera_params = extract_video_camera_params(original_video_path)
            
            # STEP 2: Handle file conversion/copying for frame extraction
            processing_video_path = original_video_path
            
            if video_file.lower().endswith('.mov'):
                # Convert MOV to MP4 for better frame extraction compatibility
                mp4_file = video_file.rsplit('.', 1)[0] + '.mp4'
                mp4_path = os.path.join(videos_dir, mp4_file)
                
                if not os.path.exists(mp4_path):
                    print(f"  Converting {video_file} to MP4 for frame extraction...")
                    if convert_mov_to_mp4(original_video_path, mp4_path):
                        processing_video_path = mp4_path
                        print(f"  Converted: {video_file} -> {mp4_file}")
                    else:
                        print(f"  Failed to convert {video_file}, using original")
                        processing_video_path = original_video_path
                else:
                    processing_video_path = mp4_path
                    print(f"  Using existing converted file: {mp4_file}")
            else:
                # Copy other video formats to videos directory
                import shutil
                video_dest = os.path.join(videos_dir, video_file)
                if not os.path.exists(video_dest):
                    shutil.copy2(original_video_path, video_dest)
                processing_video_path = video_dest
            
            # STEP 3: Extract frames (using converted file if available for compatibility)
            print(f"  Extracting frames from {os.path.basename(processing_video_path)}...")
            extracted = extract_frames_from_video(
                processing_video_path, images_dir,
                kwargs['frame_interval'],
                kwargs['max_frames'],
                kwargs['quality_threshold'],
            )
            print(f"  Extracted {extracted} frames")
            
            # STEP 4: Use camera parameters from ORIGINAL file
            if not camera_params and video_camera_params:
                camera_params = video_camera_params
                print(f"  ✓ Using camera parameters from ORIGINAL {video_file}: {video_camera_params['image_size']}")
            elif video_camera_params:
                print(f"  ✓ Camera parameters extracted from ORIGINAL {video_file} (HEIC takes priority)")
            else:
                print(f"  ❌ CRITICAL: No camera parameters extracted from {video_file}")
                print(f"  ❌ Cannot proceed without camera calibration data from original files")
                print(f"  ❌ Check iPhone camera settings or file metadata")
    
    # Copy regular image files
    if image_files:
        print(f"\nCopying {len(image_files)} image files...")
        import shutil
        for img_file in image_files:
            src = os.path.join(input_dir, img_file)
            dst = os.path.join(images_dir, img_file)
            shutil.copy2(src, dst)
            print(f"  Copied: {img_file}")
    
    # Save camera parameters if available
    if camera_params:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        calib_dir = os.path.join(script_dir, 'calibrations')
        os.makedirs(calib_dir, exist_ok=True)
        
        calib_file = os.path.join(calib_dir, f'{dataset}_calibration.json')
        with open(calib_file, 'w') as f:
            json.dump(camera_params, f, indent=2)
        
        print(f"\n✅ Camera calibration saved: {calib_file}")
        print(f"  Resolution: {camera_params['image_size']}")
        print(f"  fx: {camera_params['fx']:.1f}, fy: {camera_params['fy']:.1f}")
        print(f"  cx: {camera_params['cx']:.1f}, cy: {camera_params['cy']:.1f}")
        print(f"  Focal length: {camera_params['focal_length_mm']}mm")
    else:
        print(f"\n❌ CRITICAL ERROR: No camera calibration data extracted from original files!")
        print(f"❌ Cannot proceed with SFM reconstruction without camera parameters.")
        print(f"❌ Ensure your iPhone/HEIC files contain proper camera metadata.")
        print(f"\n📱 iPhone Camera Settings to Enable Metadata:")
        print(f"   Settings > Camera > Formats > Most Compatible (NOT High Efficiency)")
        print(f"   Settings > Privacy & Security > Location Services > Camera > While Using App")
        return  # Stop processing if no calibration available
    
    # Count final images
    final_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nProcessing complete: {len(final_images)} images ready for SFM")
    
    if len(final_images) > 0:
        print(f"Run SFM: python script/pipeline.py --data-dir {output_dir} --dataset {dataset}")


def print_instructions():
    """Print iPhone camera setup instructions"""
    print("""
📱 iPhone Camera Setup for Optimal SFM Results

🎯 CRITICAL SETTINGS:
1. Use main camera (1x lens, NOT ultra-wide/telephoto)
2. Lock focus/exposure: Tap and HOLD subject until "AE/AF LOCK" appears
3. Settings > Camera:
   ✅ Record Video: 4K at 30fps (or 1080p for smaller files)
   ✅ Formats: Most Compatible (NOT High Efficiency)
   ❌ Auto Macro: OFF
   ❌ Scene Detection: OFF

📹 RECORDING TECHNIQUE:
- Move SLOWLY and SMOOTHLY
- Maintain 60-80% overlap between frames
- Keep subject in frame at all times
- Use consistent, bright lighting
- Record 30-60s for objects, 2-3min for rooms

⚠️ AVOID:
❌ Auto-focus hunting (use AE/AF LOCK!)
❌ Fast movements or zooming
❌ Mixed lighting conditions
❌ Reflective surfaces
""")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Media Manager for SFM Pipeline')
    
    parser.add_argument('input_dir', nargs='?', help='Input directory containing media files')
    parser.add_argument('--output-dir', default='data', help='Output data directory (default: data)')
    parser.add_argument('--dataset', default='custom', help='Dataset name (default: custom)')
    
    # Video extraction parameters
    parser.add_argument('--frame-interval', type=int, default=10, 
                       help='Extract every N frames (default: 10)')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames per video (default: 50)')
    parser.add_argument('--quality-threshold', type=float, default=25,
                       help='Minimum blur score (default: 25)')
    
    parser.add_argument('--instructions', action='store_true',
                       help='Show iPhone camera setup instructions')
    
    args = parser.parse_args()
    
    if args.instructions:
        print_instructions()
        return
    
    if not args.input_dir:
        parser.error('input_dir is required when not using --instructions')
    
    process_media_directory(
        args.input_dir, args.output_dir, args.dataset,
        frame_interval=args.frame_interval,
        max_frames=args.max_frames,
        quality_threshold=args.quality_threshold
    )


if __name__ == '__main__':
    main()
