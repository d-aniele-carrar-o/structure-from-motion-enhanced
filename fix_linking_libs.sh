#!/bin/bash

# --- Configuration ---
MVS_BIN_DIR="/usr/local/bin/OpenMVS"
LIB_DIR="/opt/anaconda3/lib"

# --- Libraries to fix (Boost + Python) ---
LIBRARIES_TO_FIX=(
  "libboost_iostreams.dylib"
  "libboost_program_options.dylib"
  "libboost_system.dylib"
  "libboost_filesystem.dylib"
  "libboost_serialization.dylib"
  "libpython3.12.dylib"
)

# --- Main Loop ---
echo "Attempting to fix all executables in $MVS_BIN_DIR..."

# Loop through every file in the directory
for exec_path in "$MVS_BIN_DIR"/*; do
  # Check if it's a regular, executable file
  if [ -f "$exec_path" ] && [ -x "$exec_path" ]; then
    echo "--- Fixing $(basename "$exec_path") ---"
    for lib_name in "${LIBRARIES_TO_FIX[@]}"; do
      sudo install_name_tool -change "@rpath/$lib_name" "$LIB_DIR/$lib_name" "$exec_path"
    done
  fi
done

echo "--- Fix process complete. ---"
