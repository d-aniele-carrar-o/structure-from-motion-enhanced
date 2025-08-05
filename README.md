# Structure from Motion 

Structure from Motion (SFM) from scratch, using Numpy and OpenCV with **iPhone media support**.

![](results/misc-figs/fountain_p11.png)

This repository provides:
* **Complete SFM Pipeline** with iPhone HEIC/MOV support
* **Automatic Camera Calibration** from HEIC metadata
* **Unified Media Processing** (HEIC→JPG, MOV→MP4→frames)
* Self-reliant SFM tutorials
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
      IMG_1235.HEIC  # iPhone photos
      ...
```

### 3. Run Complete Pipeline
```bash
# Process media + run SFM + create 3D visualization
python script/pipeline.py --dataset my_project --visualize-3d
```

That's it! The pipeline automatically:
- ✅ Converts HEIC→JPG with camera calibration extraction
- ✅ Converts MOV→MP4 and extracts optimal frames
- ✅ Runs feature matching and SFM reconstruction
- ✅ Creates interactive 3D point cloud visualization

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
  my_project/              # Your dataset name
    media/                 # 📁 Raw iPhone files (.HEIC, .MOV)
    images/                # ✅ Auto-generated processed images
    videos/                # ✅ Auto-generated converted videos
    features/              # ✅ Auto-generated feature files
    matches/               # ✅ Auto-generated match files

results/
  my_project/              # SFM reconstruction results
    point-clouds/          # 3D point cloud files (.ply)
    errors/                # Reprojection error plots

script/
  calibrations/            # Camera calibration files
```

## 🛠️ Advanced Usage

### Custom Media Directory
```bash
# Process media from custom location
python script/pipeline.py --media-dir /path/to/iphone/export --dataset my_project
```

### Video Frame Extraction Settings
```bash
# Extract more frames with higher quality threshold
python script/pipeline.py --frame-interval 5 --max-frames 100 --quality-threshold 75
```

### Individual Components
```bash
# Process media only
python script/media_manager.py /path/to/media --dataset my_project

# Run SFM on existing processed images
python script/pipeline.py --dataset my_project --visualize-3d

# Clean and restart
python script/pipeline.py --dataset my_project --clean --visualize-3d
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

**For MOV conversion:**
```bash
# Install ffmpeg
brew install ffmpeg  # macOS
# or apt install ffmpeg  # Linux
```

## 📊 Results

The pipeline generates:
- **3D Point Clouds** (.ply files) viewable in MeshLab
- **Interactive 3D Visualization** (matplotlib)
- **Reprojection Error Analysis**
- **Camera Pose Estimation**

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
- **🎬 Smart Frame Extraction**: Blur detection and optimal frame spacing
- **🚀 One-Command Pipeline**: From raw media to 3D reconstruction
- **📊 Interactive Visualization**: 3D point cloud viewer
- **🧹 Clean Architecture**: Organized directory structure

## 🤝 Authors
* [Muneeb Aadil](https://muneebaadil.github.io)
* [Sibt ul Hussain](https://sites.google.com/site/sibtulhussain/)

*Enhanced with iPhone media support and unified pipeline*
