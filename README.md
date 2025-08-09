# Structure from Motion 

Structure from Motion (SFM) from scratch, using Numpy and OpenCV with **iPhone media support**.

![](results/misc-figs/fountain_p11.png)

This repository provides:
* **Complete SFM Pipeline** with iPhone HEIC/MOV support
* **Automatic Camera Calibration** from HEIC metadata
* **LiDAR Dimension Extraction** for furniture measurement (iPhone 15 Pro)
* **Optimized Media Processing** (HEIC→JPG, direct video frame extraction)
* **Dual SfM Methods** (Custom implementation + COLMAP integration)
* **Dense Reconstruction** (OpenMVS + COLMAP depth maps)
* **Organized Output Structure** with clean file management
* Self-reliant SfM tutorials
* Interactive 3D visualization

## 🚀 Quick Start (iPhone Media)

### 1. Setup
```bash
# Install dependencies
pip install opencv-python numpy matplotlib pillow-heif

# Create dataset structure
python setup_dataset.py --dataset my_project
```

### 2. Add Your iPhone Media
Copy your iPhone photos/videos to `data/my_project/media/`:
```
data/
  my_project/
    media/           # 📁 PUT YOUR FILES HERE
      IMG_1234.MOV   # iPhone videos
      IMG_1235.HEIC  # iPhone photos (with LiDAR depth for iPhone 15 Pro)
      ...
```

### 3. Run Complete Pipeline
```bash
# Process media + run SFM + create 3D visualization + extract dimensions
python main.py --dataset my_project --visualize-3d

# Extract only furniture dimensions from HEIC files
python extract_dimensions.py --media-dir data/my_project/media
```

That's it! The pipeline automatically:
- ✅ Converts HEIC→JPG with camera calibration extraction
- ✅ Extracts optimal frames directly from videos (no conversion needed)
- ✅ Runs feature matching and SFM reconstruction
- ✅ Generates sparse and dense reconstructions
- ✅ **Extracts furniture dimensions from iPhone 15 Pro LiDAR depth data**
- ✅ Creates organized output structure
- ✅ Interactive 3D point cloud visualization

## 📱 iPhone Camera Setup

For optimal results, use proper iPhone camera settings:
```bash
python script/media_manager.py --instructions
```

**Key Settings:**
- Use main camera (1x lens, NOT ultra-wide)
- Lock focus/exposure: Tap & HOLD until "AE/AF LOCK" appears
- Record in 4K/1080p at 30fps
- Move SLOWLY with 60-80% overlap between frames

## 📁 Directory Structure

The pipeline uses a clean, organized structure:
```
data/
  my_project/
    media/                 # 📁 Raw iPhone files (.HEIC, .MOV)

results/
  my_project/
    ├── processing/        # 🔧 All intermediate processing files
    │   ├── images/        # Processed images (HEIC→JPG, video frames)
    │   ├── features/      # Feature extraction results
    │   ├── matches/       # Feature matching results
    │   ├── matches_vis/   # Match visualization images
    │   ├── calibrations/  # Camera calibration files
    │   └── depth_maps/    # LiDAR depth visualizations
    │
    ├── custom_sfm/        # Your custom SfM implementation
    │   ├── point_cloud.ply
    │   └── scene.mvs
    ├── colmap/            # COLMAP sparse reconstruction
    │   ├── database.db
    │   └── sparse/
    ├── dense/             # Dense reconstruction outputs
    │   └── colmap/        # COLMAP depth maps (.dmap files)
    ├── mvs/               # OpenMVS dense reconstruction
    │   ├── scene_dense.mvs
    │   └── textured models
    ├── dimensions/        # 📏 Furniture dimension extraction
    │   ├── *_dimensions.json  # Dimension measurements
    │   └── *_dimensions.jpg   # Visualization with measurements
    └── panorama/          # Panorama outputs
        └── panorama.jpg
```

## 🛠️ Advanced Usage

### SfM Method Selection
```bash
# Use custom SfM implementation (default)
python main.py --dataset my_project --sfm-method custom

# Use COLMAP for reconstruction
python main.py --dataset my_project --sfm-method colmap
```

### Custom Media Directory
```bash
# Process media from custom location
python main.py --media-dir /path/to/iphone/export --dataset my_project
```

### Furniture Dimension Extraction
```bash
# Extract dimensions from iPhone 15 Pro HEIC files with LiDAR
python extract_dimensions.py --media-dir data/my_project/media --debug

# Disable dimension extraction in main pipeline
python main.py --dataset my_project --no-extract-dimensions
```

### Video Frame Extraction Settings
```bash
# Extract more frames with higher quality threshold
python main.py --frame-interval 5 --max-frames 100 --quality-threshold 75
```

### Visualization and Analysis
```bash
# View all depth maps
python view_all_dmaps.py results/my_project/dense/

# Clean and restart
python main.py --dataset my_project --clean --visualize-3d
```

## 📚 Tutorials

Detailed tutorials are in the `tutorial/` directory:
1. Chapter 1: Prerequisites
2. Chapter 2: Epipolar Geometry  
3. Chapter 3: 3D Scene Estimations
4. Chapter 4: Putting It Together

## 🔧 Prerequisites

**Required:**
```bash
pip install opencv-python numpy matplotlib
```

**For HEIC support (recommended):**
```bash
pip install pillow-heif
# OR on macOS: uses built-in 'sips' command
```

**For video processing and dense reconstruction:**
```bash
# Install ffmpeg (for video metadata extraction)
brew install ffmpeg  # macOS
# or apt install ffmpeg  # Linux

# For dense reconstruction (optional)
# Install COLMAP: https://colmap.github.io/install.html
# Install OpenMVS: https://github.com/cdcseacave/openMVS
```

## 📊 Results

The pipeline generates:
- **Sparse Point Clouds** (.ply files) from both custom and COLMAP methods
- **Dense Reconstructions** with OpenMVS integration
- **Depth Maps** (.dmap files) for detailed surface analysis
- **Interactive 3D Visualization** (matplotlib)
- **Panorama Stitching** from matched features
- **Organized Output Structure** for easy analysis

### Sample Results
![](results/misc-figs/fountain_p11.png)
*Fountain P11 reconstruction*

![](results/misc-figs/herz_jesus_p8.png)
*Herz Jesus P8 reconstruction*

![](results/misc-figs/entry_p10.png)
*Entry P10 reconstruction*

## 🎯 Key Features

- **📱 iPhone Native Support**: Direct HEIC/MOV processing
- **🔧 Automatic Calibration**: Extracts camera parameters from HEIC metadata
- **📏 LiDAR Dimension Extraction**: Real-world furniture measurements from iPhone 15 Pro depth data
- **🎬 Optimized Frame Extraction**: Direct video processing with blur detection
- **🔀 Dual SfM Methods**: Custom implementation + COLMAP integration
- **🏗️ Dense Reconstruction**: OpenMVS + COLMAP depth map generation
- **🚀 One-Command Pipeline**: From raw media to complete 3D reconstruction
- **📊 Interactive Visualization**: 3D point cloud and depth map viewers
- **🗂️ Organized Structure**: Clean, method-specific output organization
- **🧹 Efficient Processing**: No unnecessary file conversions

## 🤝 Authors
* [Muneeb Aadil](https://muneebaadil.github.io)
* [Sibt ul Hussain](https://sites.google.com/site/sibtulhussain/)

*Enhanced with iPhone media support and unified pipeline*
