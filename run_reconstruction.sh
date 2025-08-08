#!/bin/bash

# --- CONFIGURATION ---
# Set the path to your source images
IMAGE_DIR="/Users/danielecarraro/Documents/VSCODE/structure-from-motion/data/custom/images"


# --- 1. SPARSE RECONSTRUCTION (COLMAP) ---
echo "--- Step 1: Running COLMAP for sparse reconstruction... ---"

# Feature Extraction with the correct PINHOLE model and your known intrinsics
colmap feature_extractor \
  --database_path database.db \
  --image_path "$IMAGE_DIR" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model PINHOLE \
  --ImageReader.camera_params "1280,1280,960,540"

# Feature Matching
colmap exhaustive_matcher --database_path database.db

# Sparse Reconstruction (Mapping)
mkdir sparse
colmap mapper \
  --database_path database.db \
  --image_path "$IMAGE_DIR" \
  --output_path sparse

# Move model files to the expected location
mv sparse/0/* sparse/
echo "--- COLMAP complete. Sparse model created. ---"
echo ""


# --- 2. DENSE RECONSTRUCTION (OpenMVS) ---
echo "--- Step 2: Running OpenMVS for dense reconstruction... ---"
mkdir dense

# Convert the COLMAP model to the OpenMVS format
InterfaceCOLMAP -i . -o dense/scene.mvs

# 1. Densify the point cloud (this is the long step)
echo "Densifying point cloud... this will take a while."
DensifyPointCloud -i dense/scene.mvs -o dense/scene_dense.mvs -w dense --resolution-level 1

# 2. Reconstruct the mesh from the dense point cloud
echo "Reconstructing the mesh..."
ReconstructMesh -i dense/scene_dense.mvs -o dense/scene_mesh.mvs -w dense

# 3. Refine the mesh to improve quality
echo "Refining the mesh..."
RefineMesh -i dense/scene_mesh.mvs -o dense/scene_mesh_refined.mvs -w dense

# 4. Texture the mesh with the original images
echo "Texturing the mesh..."
TextureMesh -i dense/scene_mesh_refined.mvs -o dense/scene_final.mvs --export-type obj -w dense

echo ""
echo "--- Pipeline Complete! ---"
echo "Final textured model saved as: dense/scene_final.obj"
