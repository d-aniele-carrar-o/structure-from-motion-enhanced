# Live iPhone Dimension Extraction Setup

## Method 1: Continuity Camera (Recommended)
**Requirements:** macOS Ventura+ and iPhone with iOS 16+

### Setup:
1. **Enable Continuity Camera:**
   - iPhone: Settings > General > AirPlay & Handoff > Continuity Camera (ON)
   - Mac: System Settings > General > AirDrop & Handoff > iPhone Wireless Camera (ON)

2. **Connect:**
   - Ensure both devices on same WiFi network
   - iPhone and Mac signed into same Apple ID
   - Run the live detector:
   ```bash
   python live_dimension_extractor.py
   ```

## Method 2: USB Connection
### Setup:
1. **Connect iPhone via USB-C/Lightning cable**
2. **Trust the computer** when prompted on iPhone
3. **Run the detector:**
   ```bash
   python live_dimension_extractor.py
   ```

## Method 3: Third-Party Streaming Apps
### Option A: EpocCam
1. **Install EpocCam on iPhone** (App Store)
2. **Install EpocCam drivers on Mac** (kinoni.com)
3. **Connect both to same WiFi**
4. **Run detector**

### Option B: Camo
1. **Install Camo on iPhone** (App Store)  
2. **Install Camo on Mac** (reincubate.com)
3. **Connect and run detector**

## Usage Instructions

### Controls:
- **'q'** - Quit application
- **'s'** - Save current detection with dimensions
- **'d'** - Cycle through distance estimates (1.0m → 1.5m → 2.0m → 2.5m → 3.0m)

### Live Feedback:
- **Green contour** - Detected furniture outline
- **Blue rectangle** - Bounding box with corner markers
- **Yellow text** - Real-time measurements and confidence
- **Green circle** (top-right) - Stable detection indicator
- **Yellow circle** - Detection in progress

### Best Practices:
1. **Good lighting** - Avoid harsh shadows
2. **Steady hands** - Hold iPhone stable for 2-3 seconds
3. **Clear edges** - Ensure furniture edges are visible
4. **Proper distance** - Use 'd' key to adjust distance estimate
5. **Fill frame** - Furniture should occupy 30-80% of frame

### Troubleshooting:
- **No camera detected:** Try different USB ports or restart both devices
- **Poor detection:** Adjust lighting, distance, or angle
- **Unstable measurements:** Hold steadier, wait for green stability indicator

## Real-time Depth (Future Enhancement)
For true depth data, we would need:
- iPhone app with LiDAR access
- Custom streaming protocol
- ARKit integration

Current version uses distance estimation - adjust with 'd' key for accuracy.