r"""
Central configuration for the KITTI autonomous perception & risk pipeline.

All scripts read paths from here instead of hardcoding them, so the project
runs on any machine as long as you set these three environment variables
(or just edit the defaults below to point at your local KITTI download).

Usage:
    export KITTI_FRAMES_DIR=/path/to/2011_09_26_drive_0005_sync/.../image_02/data
    export KITTI_OXTS_DIR=/path/to/2011_09_26_drive_0005_sync/.../oxts/data
    python final_autonomous_dashboard.py

Or on Windows (PowerShell):
    $env:KITTI_FRAMES_DIR="C:\path\to\image_02\data"
"""

import os

# Root directory for this project (so relative output paths always resolve
# correctly regardless of where the script is invoked from).
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Input data ---------------------------------------------------------
# Path to the KITTI raw sequence "image_02/data" folder (left camera frames).
KITTI_FRAMES_DIR = os.environ.get(
    "KITTI_FRAMES_DIR",
    os.path.join(PROJECT_ROOT, "data", "2011_09_26_drive_0005_sync",
                 "2011_09_26", "2011_09_26_drive_0005_sync",
                 "image_02", "data"),
)

# Path to the KITTI raw sequence "oxts/data" folder (GPS/IMU logs).
KITTI_OXTS_DIR = os.environ.get(
    "KITTI_OXTS_DIR",
    os.path.join(PROJECT_ROOT, "data", "2011_09_26_drive_0005_sync",
                 "2011_09_26", "2011_09_26_drive_0005_sync",
                 "oxts", "data"),
)

# --- Model checkpoints ---------------------------------------------------
CONVLSTM_MODEL_PATH = os.path.join(PROJECT_ROOT, "convlstm_kitti.pth")
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")

# --- Output directories (created on demand by each script) --------------
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CONVLSTM_PREDICTION_DIR = os.path.join(PROJECT_ROOT, "convlstm_prediction")
DIFFUSION_FUTURE_DIR = os.path.join(PROJECT_ROOT, "diffusion_future")
SIMVP_FUTURE_DIR = os.path.join(PROJECT_ROOT, "simvp_future")
OBJECT_PATCH_DIR = os.path.join(PROJECT_ROOT, "object_patches")

os.makedirs(OUTPUT_DIR, exist_ok=True)
