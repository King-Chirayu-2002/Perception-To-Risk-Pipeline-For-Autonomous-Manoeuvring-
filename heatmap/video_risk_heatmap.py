import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

from trajectory_predictor import TrajectoryPredictor
from ttc_montecarlo import TTCMonteCarlo
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR


# =========================
# CONFIG
# =========================

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "global_risk_heatmap.png")

FPS = 10
FUTURE_TIME = 2.0


# =========================
# LOAD MODELS
# =========================

yolo = YOLO("yolov8n.pt")

tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

ttc_mc = TTCMonteCarlo()


# =========================
# LOAD FRAMES
# =========================

frames = sorted(os.listdir(FRAMES_DIR))

first_frame = cv2.imread(os.path.join(FRAMES_DIR,frames[0]))

h,w = first_frame.shape[:2]


# GLOBAL RISK MAP
global_risk = np.zeros((h,w),dtype=np.float32)

track_history = {}


# =========================
# PROCESS VIDEO
# =========================

for f in frames:

    frame = cv2.imread(os.path.join(FRAMES_DIR,f))

    results = yolo(frame)[0]

    detections = []

    for box in results.boxes:

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        detections.append([x1,y1,x2,y2,conf])

    dets = np.array(detections) if detections else np.empty((0,5))

    tracks = tracker.update(dets)


    ego_x = w//2
    ego_y = h-10


    # ======================
    # TRACK OBJECTS
    # ======================

    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2


        if tid not in track_history:
            track_history[tid] = []

        track_history[tid].append((cx,cy))

        if len(track_history[tid]) > 20:
            track_history[tid].pop(0)


        # ======================
        # VELOCITY ESTIMATION
        # ======================

        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)


        # ======================
        # FUTURE POSITION
        # ======================

        future_x = int(cx + vx*FUTURE_TIME/FPS)
        future_y = int(cy + vy*FUTURE_TIME/FPS)


        # ======================
        # COLLISION RISK
        # ======================

        samples = ttc_mc.compute_ttc_samples(
            (cx,cy),(vx,vy),(ego_x,ego_y)
        )

        prob = ttc_mc.collision_probability(samples)


        # ======================
        # ADD TO GLOBAL HEATMAP
        # ======================

        radius = int(40 + prob*120)

        cv2.circle(
            global_risk,
            (future_x,future_y),
            radius,
            prob,
            -1
        )


# =========================
# NORMALIZE HEATMAP
# =========================

global_risk = global_risk / (global_risk.max()+1e-6)

heatmap = cv2.applyColorMap(
    (global_risk*255).astype(np.uint8),
    cv2.COLORMAP_JET
)


# =========================
# OVERLAY WITH FRAME
# =========================

overlay = cv2.addWeighted(
    first_frame,
    0.6,
    heatmap,
    0.7,
    0
)


# =========================
# SAVE OUTPUT
# =========================

cv2.imwrite(OUTPUT_PATH,overlay)

print("Saved global risk heatmap:",OUTPUT_PATH)