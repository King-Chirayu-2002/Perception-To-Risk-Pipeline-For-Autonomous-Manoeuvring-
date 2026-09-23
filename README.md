# KITTI Autonomous Perception & Risk Pipeline

Real-time perception-to-risk pipeline for autonomous driving, built and evaluated on the
[KITTI](https://www.cvlibs.net/datasets/kitti/) raw dataset. Combines CUDA-accelerated
monocular depth estimation (MiDaS), YOLOv8 object detection, and SORT multi-object
tracking to estimate per-object 3D position from camera projection geometry. Adds
physics-based and Monte Carlo probabilistic trajectory prediction, Time-to-Collision (TTC)
risk scoring, and a ConvLSTM model trained to forecast future frames for early
collision-risk heatmaps. Includes classical lane detection (Canny edge detection + Hough
transform) and a bird's-eye-view planner, with every module integrated into a single live
autonomous-driving dashboard, debugged end-to-end against real KITTI sequences.

Built as an independent exploration of the full **perception → prediction → risk** stack
used in autonomous vehicles and robotics — going beyond single-model demos to an
integrated, debuggable system.

## What's in here

| Module | What it does |
|---|---|
| `depth_module.py` | CUDA-accelerated monocular depth estimation (MiDaS, via `torch.hub`) |
| `camera_model.py` | Pixel → 3D camera-frame projection using KITTI camera intrinsics |
| `bbox_3d_estimator.py` | Approximate 3D bounding boxes from 2D detections + depth |
| `sort.py` | Multi-object tracking (SORT — vendored, see [Credits](#credits)) |
| `trajectory_predictor.py` | Physics-based (constant-velocity) trajectory prediction |
| `probabilistic_predictor.py` | Monte Carlo sampled future trajectories |
| `ttc_montecarlo.py` | Time-to-Collision risk scoring via Monte Carlo sampling |
| `convlstm_predict_trained.py`, `train_convlstm.py` | ConvLSTM future-frame prediction (trained on KITTI sequences) |
| `lane_detector.py` | Classical lane detection (Canny + Hough transform) |
| `bev_visualizer.py` | Bird's-eye-view scene visualization |
| `final_autonomous_dashboard.py` | Integrates every module above into one live dashboard |
| `demo/` | Standalone scripts demonstrating individual modules |
| `heatmap/` | Video-level risk heatmap generation |

## Setup

```bash
git clone https://github.com/<your-username>/kitti-autonomous-perception-risk.git
cd kitti-autonomous-perception-risk
pip install -r requirements.txt
```

CUDA is used automatically if available; the pipeline falls back to CPU otherwise
(slower, but functional). MiDaS weights are downloaded automatically on first run via
`torch.hub`.

### Getting KITTI data

This project uses the KITTI raw sequence `2011_09_26_drive_0005_sync`. Download it from
the [KITTI raw data page](https://www.cvlibs.net/datasets/kitti/raw_data.php) and point
the pipeline at it via environment variables (see `config.py`):

```bash
export KITTI_FRAMES_DIR=/path/to/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/image_02/data
export KITTI_OXTS_DIR=/path/to/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync/oxts/data
```

On Windows (PowerShell):

```powershell
$env:KITTI_FRAMES_DIR="C:\path\to\image_02\data"
$env:KITTI_OXTS_DIR="C:\path\to\oxts\data"
```

If you don't set these, `config.py` defaults to `./data/2011_09_26_drive_0005_sync/...` —
just place the downloaded sequence there instead.

### Running it

```bash
# Full integrated dashboard
python final_autonomous_dashboard.py

# Individual module demos
python demo/detection_tracking_demo.py
python demo/bev_planner_demo.py
python demo/risk_heatmap_demo.py
```

## Optional: future-frame prediction baselines (SimVP / OpenSTL)

Some future-frame prediction experiments (`simvp_inference_simple.py`,
`diffusion_future_prediction.py`) build on two external research codebases that are
**not vendored in this repo**:

- [SimVP](https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction) (CVPR 2022)
- [OpenSTL](https://github.com/chengtan9907/OpenSTL)

To use these scripts, clone the relevant repo alongside this one and adjust the import
path at the top of the script, e.g.:

```bash
git clone https://github.com/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction.git SimVP-master
git clone https://github.com/chengtan9907/OpenSTL.git OpenSTL-OpenSTL-Lightning
```

The core pipeline (`final_autonomous_dashboard.py` and everything it depends on) does
**not** require either of these — they're only used by the standalone future-frame
prediction baseline scripts.

## Results

- Trained a ConvLSTM model on KITTI sequences to forecast future frames; used the
  predicted frames to generate early collision-risk heatmaps ahead of real-time detections.
- Monte Carlo TTC scoring surfaces high-risk objects before a simple constant-velocity
  estimate would flag them, by sampling velocity uncertainty.
- Full dashboard runs end-to-end against real KITTI sequences with live detection,
  tracking, depth, trajectory prediction, and risk overlay.

See the `*.png` and `*.mp4` files in this repo for sample output (detection/tracking,
trajectory prediction, BEV planning, and future-frame prediction).

## Credits

- **MiDaS** — Ranftl et al., Intel ISL, loaded via `torch.hub` at runtime.
- **YOLOv8** — [Ultralytics](https://github.com/ultralytics/ultralytics).
- **SORT** — Bewley et al., *"Simple Online and Realtime Tracking"*, vendored in
  `sort.py` under its original **GPLv3** license. See the license header in that file.
- **SimVP** / **OpenSTL** — used as optional external baselines for future-frame
  prediction (see [above](#optional-future-frame-prediction-baselines-simvp--openstl));
  not vendored in this repo.
- **KITTI dataset** — Geiger et al., Karlsruhe Institute of Technology / TTI Chicago.

## Status / honest scope

This is an integration and systems-engineering project, not novel research — the value
is in wiring together depth estimation, detection, tracking, prediction, and risk
scoring into one debuggable, real-time pipeline against real sensor data, not in any
single model being new. Some experimental scripts (diffusion-based future-frame
prediction) are exploratory and less polished than the core dashboard.

