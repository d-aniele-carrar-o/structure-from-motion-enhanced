#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import struct
import glob
import os

def read_dmap(filepath):
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        width = struct.unpack('<I', f.read(4))[0]
        height = struct.unpack('<I', f.read(4))[0]
        f.seek(64)
        depth_data = np.frombuffer(f.read(width * height * 4), dtype=np.float32)
        depth_map = depth_data.reshape((height, width)).copy()
        depth_map[depth_map <= 0] = np.nan
    return depth_map

def view_all_dmaps(search_dir="."):
    # Search for .dmap files in the specified directory and subdirectories
    dmap_files = sorted(glob.glob(os.path.join(search_dir, "**/*.dmap"), recursive=True))
    
    if not dmap_files:
        print(f"No .dmap files found in {search_dir}")
        return
    
    for i, filepath in enumerate(dmap_files):
        depth_map = read_dmap(filepath)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(depth_map, cmap='plasma', interpolation='nearest')
        plt.colorbar(label='Depth')
        plt.title(f'Depth Map {i+1}/{len(dmap_files)}: {os.path.basename(filepath)}')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    import sys
    search_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    view_all_dmaps(search_dir)