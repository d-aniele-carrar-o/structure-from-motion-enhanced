import os
import cv2
import json
import logging
from shutil import copy2
from subprocess import run, CalledProcessError
from typing import Optional, Dict, Any, List
from pillow_heif import register_heif_opener
from PIL import Image, ExifTags

from .config import Config

register_heif_opener()

class MediaProcessor:
    """
    Handles all media pre-processing tasks for the SfM pipeline.
    """
    def __init__(self, config: Config):
        self.config = config
        self.camera_params: Optional[Dict[str, Any]] = None

    def is_complete(self) -> bool:
        try:
            image_files = [f for f in os.listdir(self.config.images_dir) if f.lower().endswith(tuple(self.config.ext))]
            calibration_exists = os.path.exists(self.config.get_calibration_path())
            return bool(image_files) and calibration_exists
        except FileNotFoundError:
            return False

    def run(self):
        if not os.path.exists(self.config.media_dir):
            logging.error(f"Media source directory not found: {self.config.media_dir}")
            return

        all_files = os.listdir(self.config.media_dir)
        media_files = [f for f in all_files if f.lower().endswith(('.heic', '.mov', '.mp4', '.avi', '.jpg', '.jpeg', '.png'))]
        if not media_files:
            logging.warning(f"No media files found in {self.config.media_dir}. Nothing to process.")
            return

        logging.info(f"Found {len(media_files)} media files to process.")
        heic_files = [f for f in media_files if f.lower().endswith('.heic')]
        video_files = [f for f in media_files if f.lower().endswith(('.mov', '.mp4', '.avi'))]
        image_files = [f for f in media_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if heic_files: self._process_heic_files(heic_files)
        if video_files: self._process_video_files(video_files)
        if image_files: self._copy_image_files(image_files)
        self._save_camera_parameters()

    def _process_heic_files(self, files: List[str]):
        logging.info(f"Processing {len(files)} HEIC files...")
        if not self.camera_params:
            self.camera_params = self._extract_camera_params_from_heic(os.path.join(self.config.media_dir, files[0]))
        for heic_file in files:
            heic_path = os.path.join(self.config.media_dir, heic_file)
            jpg_file = os.path.splitext(heic_file)[0] + '.jpg'
            jpg_path = os.path.join(self.config.images_dir, jpg_file)
            if self._convert_heic_to_jpg(heic_path, jpg_path):
                logging.debug(f"Converted HEIC: {heic_file} -> {jpg_file}")
            else:
                logging.error(f"Failed to convert HEIC file: {heic_file}")

    def _process_video_files(self, files: List[str]):
        logging.info(f"Processing {len(files)} video files...")
        for video_file in files:
            original_path = os.path.join(self.config.media_dir, video_file)
            if not self.camera_params:
                self.camera_params = self._extract_camera_params_from_video(original_path)
            processing_path = self._prepare_video_for_extraction(original_path)
            if processing_path:
                logging.info(f"Extracting frames from {os.path.basename(processing_path)}...")
                self._extract_frames_from_video(processing_path)

    def _copy_image_files(self, files: List[str]):
        logging.info(f"Copying {len(files)} existing image files...")
        for img_file in files:
            copy2(os.path.join(self.config.media_dir, img_file), self.config.images_dir)
            logging.debug(f"Copied image: {img_file}")

    def _prepare_video_for_extraction(self, original_path: str) -> Optional[str]:
        video_file = os.path.basename(original_path)
        dest_path = os.path.join(self.config.videos_dir, video_file)
        try:
            copy2(original_path, dest_path)
            logging.debug(f"Copied video to processing directory: {video_file}")
            return dest_path
        except Exception as e:
            logging.error(f"Failed to copy video file {video_file}: {e}")
            return None

    def _extract_frames_from_video(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video for frame extraction: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        logging.debug(f"Video Info: {total_frames} frames, {fps:.1f} FPS.")
        logging.debug(f"Extraction settings: interval={self.config.frame_interval}, max_frames={self.config.max_frames}, quality_thresh={self.config.quality_threshold}")

        extracted_count = 0
        frame_idx = 0
        
        while extracted_count < self.config.max_frames and frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            blur_score = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if blur_score > self.config.quality_threshold:
                frame_filename = f"frame_{os.path.splitext(os.path.basename(video_path))[0]}_{extracted_count:04d}.jpg"
                frame_path = os.path.join(self.config.images_dir, frame_filename)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                extracted_count += 1
            else:
                logging.debug(f"Skipping frame {frame_idx} due to low quality (blur score: {blur_score:.1f})")
            
            frame_idx += self.config.frame_interval
        
        cap.release()
        logging.info(f"Extracted {extracted_count} high-quality frames from {os.path.basename(video_path)}.")

    def _extract_camera_params_from_heic(self, heic_path: str) -> Optional[Dict[str, Any]]:
        logging.debug(f"Attempting to extract camera parameters from HEIC: {os.path.basename(heic_path)}")
        
        with Image.open(heic_path) as img:
            width, height = img.size
            exif = img.getexif()
            exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            focal_length_mm = exif_data.get('FocalLength')
            focal_length_35mm = exif_data.get('FocalLengthIn35mmFilm')
            
            if focal_length_mm and focal_length_35mm:
                return self._calculate_intrinsics(width, height, float(focal_length_mm), float(focal_length_35mm))
            else:
                logging.error(f"Missing focal length data in HEIC file: {os.path.basename(heic_path)}")
        
        return self._extract_params_with_exiftool(heic_path)

    def _extract_camera_params_from_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Extracts camera parameters from video metadata using exiftool."""
        logging.debug(f"Attempting to extract camera parameters from video: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Cannot open video file: {video_path}")
            return None
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        focal_length_mm, focal_length_35mm = None, None
        
        result = run(['exiftool', '-j', video_path], 
                            capture_output=True, text=True, check=True)
        metadata = json.loads(result.stdout)[0]
        focal_length_35mm = metadata.get('FocalLengthIn35mmFormat')
        if focal_length_35mm:
            focal_length_35mm = float(str(focal_length_35mm).replace('mm', '').strip())
        
        # Extract from LensModel if available
        if 'LensModel' in metadata:
            import re
            match = re.search(r'(\d+\.\d+)mm', metadata['LensModel'])
            if match:
                focal_length_mm = float(match.group(1))
        
        if not focal_length_mm:
            logging.error(f"Could not extract focal length from video: {os.path.basename(video_path)}")
            return None

        if not focal_length_35mm:
            logging.error(f"Missing 35mm equivalent focal length in video: {os.path.basename(video_path)}")
            return None
        
        logging.info(f"Extracted camera parameters from video: focal={focal_length_mm}mm, 35mm_equiv={focal_length_35mm}mm")
        return self._calculate_intrinsics(width, height, focal_length_mm, focal_length_35mm)

    def _calculate_intrinsics(self, width: int, height: int, focal_mm: float, focal_35mm: float) -> Dict[str, Any]:
        fx = (focal_35mm / 36.0) * width
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        logging.info(f"Successfully calculated camera intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
        return {'camera_matrix': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], 'image_size': [width, height], 'focal_length_mm': focal_mm}

    def _convert_heic_to_jpg(self, heic_path: str, jpg_path: str, quality: int = 95) -> bool:
        try:
            with Image.open(heic_path) as img:
                img.convert('RGB').save(jpg_path, 'JPEG', quality=quality)
            return True
        except Exception as e:
            logging.error(f"HEIC conversion failed for {os.path.basename(heic_path)}: {e}")
            return False

    def _save_camera_parameters(self):
        if not self.camera_params:
            logging.critical("CRITICAL: No camera calibration data could be extracted.")
            return
        calib_path = self.config.get_calibration_path()
        with open(calib_path, 'w') as f:
            json.dump(self.camera_params, f, indent=4)
        logging.info(f"Camera calibration saved successfully to: {calib_path}")
        logging.info(f"  Resolution: {self.camera_params['image_size']}")
        k = self.camera_params['camera_matrix']
        logging.info(f"  Intrinsics: fx={k[0][0]:.1f}, fy={k[1][1]:.1f}, cx={k[0][2]:.1f}, cy={k[1][2]:.1f}")
