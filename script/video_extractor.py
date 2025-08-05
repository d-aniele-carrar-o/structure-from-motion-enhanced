#!/usr/bin/env python3
"""
Video Frame Extractor for SFM Pipeline
Extracts frames from video with optimal overlap for Structure from Motion
"""

import cv2
import os
import argparse
import numpy as np
from pathlib import Path


def extract_frames_from_video(video_path, output_dir, frame_interval=10, max_frames=50, quality_threshold=50):
    """
    Extract frames from video with optimal spacing for SFM
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        frame_interval: Extract every N frames (controls overlap)
        max_frames: Maximum number of frames to extract
        quality_threshold: Minimum quality score (blur detection)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Extracting every {frame_interval} frames")
    
    def calculate_blur_score(image):
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    extracted_count = 0
    frame_count = 0
    
    while cap.isOpened() and extracted_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Extract frame at specified intervals
        if frame_count % frame_interval == 0:
            # Check frame quality (blur detection)
            blur_score = calculate_blur_score(frame)
            
            if blur_score > quality_threshold:
                # Save frame
                frame_filename = f"frame_{extracted_count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                
                print(f"  Extracted: {frame_filename} (blur score: {blur_score:.1f})")
                extracted_count += 1
            else:
                print(f"  Skipped frame {frame_count} (too blurry: {blur_score:.1f})")
        
        frame_count += 1
    
    cap.release()
    
    print(f"\nExtraction complete:")
    print(f"  Extracted {extracted_count} frames")
    print(f"  Saved to: {output_dir}")
    
    return extracted_count


def print_iphone_instructions():
    """Print detailed iPhone 15 Pro camera setup instructions"""
    
    instructions = """
📱 iPhone 15 Pro Camera Setup for Optimal SFM Results

🎯 CRITICAL SETTINGS (Must be set before recording):

1. OPEN CAMERA APP
   - Use the main camera (1x lens, NOT ultra-wide or telephoto)
   - Switch to VIDEO mode

2. DISABLE ALL AUTOMATIC FEATURES:
   📍 Tap and HOLD on the subject to lock focus and exposure
   - You'll see "AE/AF LOCK" appear on screen
   - This prevents focus hunting and exposure changes
   
3. CAMERA SETTINGS (Settings > Camera):
   ✅ Record Video: 4K at 30 fps (or 1080p at 30fps for smaller files)
   ✅ Formats: Most Compatible (NOT High Efficiency)
   ❌ Auto Macro: OFF
   ❌ Scene Detection: OFF
   ❌ Smart HDR: OFF (if available)

4. MANUAL CONTROLS (if using third-party app like FiLMiC Pro):
   - ISO: Fixed (100-400 range)
   - Shutter Speed: Fixed (1/60s for 30fps video)
   - Focus: Manual lock
   - White Balance: Fixed (daylight/tungsten)

📹 RECORDING TECHNIQUE:

1. LIGHTING:
   - Use consistent, bright lighting
   - Avoid mixed lighting (sunlight + artificial)
   - Overcast outdoor lighting is ideal

2. MOVEMENT:
   - Move SLOWLY and SMOOTHLY
   - Maintain 60-80% overlap between frames
   - Keep the subject in frame at all times
   - Avoid rapid movements or shaking

3. DISTANCE:
   - Stay at consistent distance from subject
   - Don't zoom in/out during recording
   - For objects: 1-3 feet away
   - For rooms: walk slowly around perimeter

4. DURATION:
   - Record 30-60 seconds for small objects
   - Record 2-3 minutes for rooms/larger scenes

⚠️  WHAT TO AVOID:
   ❌ Auto-focus hunting (use AE/AF LOCK!)
   ❌ Exposure changes during recording
   ❌ Fast movements
   ❌ Zooming in/out
   ❌ Mixed lighting conditions
   ❌ Reflective or transparent surfaces
   ❌ Repetitive patterns without texture

✅ VERIFICATION:
   - Play back video - exposure should be constant
   - No focus hunting or breathing
   - Smooth, steady movement
   - Good overlap between consecutive frames

💡 PRO TIP: Record multiple short clips rather than one long clip
   This allows you to reset AE/AF lock if it gets lost.
"""
    
    print(instructions)


def main():
    parser = argparse.ArgumentParser(description='Extract frames from video for SFM pipeline')
    
    parser.add_argument('video_path', nargs='?', help='Path to input video file')
    parser.add_argument('--output-dir', default='data/custom/images', 
                       help='Output directory for frames (default: data/custom/images)')
    parser.add_argument('--interval', type=int, default=10,
                       help='Extract every N frames (default: 10)')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames to extract (default: 50)')
    parser.add_argument('--quality-threshold', type=float, default=50,
                       help='Minimum blur score threshold (default: 50)')
    parser.add_argument('--instructions', action='store_true',
                       help='Show iPhone camera setup instructions')
    
    args = parser.parse_args()
    
    if args.instructions:
        print_iphone_instructions()
        return
    
    if not args.video_path:
        parser.error('video_path is required when not using --instructions')
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    # Extract frames
    extracted_count = extract_frames_from_video(
        args.video_path,
        args.output_dir,
        args.interval,
        args.max_frames,
        args.quality_threshold
    )
    
    if extracted_count > 0:
        print(f"\n🎉 Ready for SFM pipeline!")
        print(f"Run: python script/pipeline.py --data-dir data --dataset custom --ext jpg --visualize-3d")


if __name__ == '__main__':
    main()