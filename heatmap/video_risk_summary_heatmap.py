import os
import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sort import Sort

from trajectory_predictor import TrajectoryPredictor
from ttc_montecarlo import TTCMonteCarlo
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR


# ======================
# CONFIG
# ======================

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "risk_summary_heatmap.png")

FPS = 10

frames = sorted(os.listdir(FRAMES_DIR))[:154]


# ======================
# OBJECT CLASSES
# ======================

SIGNIFICANT_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person"
]


# ======================
# LOAD MODELS
# ======================

yolo = YOLO("yolov8n.pt")

tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

ttc_mc = TTCMonteCarlo()


# ======================
# INITIALIZE STATS
# ======================

risk_stats = {}

for cls in SIGNIFICANT_CLASSES:
    risk_stats[cls] = [0,0,0]   # low, medium, high


track_history = {}


# ======================
# PROCESS VIDEO
# ======================

for f in frames:

    frame = cv2.imread(os.path.join(FRAMES_DIR,f))

    h,w = frame.shape[:2]

    ego_x = w//2
    ego_y = h-10


    results = yolo(frame)[0]

    detections = []
    labels = []

    for box in results.boxes:

        label = yolo.names[int(box.cls[0])]

        if label not in SIGNIFICANT_CLASSES:
            continue

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        conf = float(box.conf[0])

        if conf < 0.4:
            continue

        detections.append([x1,y1,x2,y2,conf])
        labels.append(label)

    dets = np.array(detections) if detections else np.empty((0,5))

    tracks = tracker.update(dets)


    for i,trk in enumerate(tracks):

        x1,y1,x2,y2,tid = trk.astype(int)

        label = labels[i] if i < len(labels) else "car"

        cx = (x1+x2)//2
        cy = (y1+y2)//2


        if tid not in track_history:
            track_history[tid] = []

        track_history[tid].append((cx,cy))

        if len(track_history[tid]) > 10:
            track_history[tid].pop(0)


        # ======================
        # VELOCITY
        # ======================

        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)


        # ======================
        # TTC RISK
        # ======================

        samples = ttc_mc.compute_ttc_samples(
            (cx,cy),(vx,vy),(ego_x,ego_y)
        )

        prob = ttc_mc.collision_probability(samples)


        # ======================
        # RISK CATEGORY
        # ======================

        if prob < 0.3:
            risk_level = 0
        elif prob < 0.6:
            risk_level = 1
        else:
            risk_level = 2

        risk_stats[label][risk_level] += 1


# ======================
# NORMALIZE DATA
# ======================

heatmap_data = []

object_labels = []

for obj in SIGNIFICANT_CLASSES:

    values = np.array(risk_stats[obj])

    total = values.sum() + 1e-6

    heatmap_data.append(values / total * 100)

    object_labels.append(obj.capitalize())


heatmap_data = np.array(heatmap_data)


# ======================
# PLOT HEATMAP
# ======================

plt.figure(figsize=(10,6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn_r",
    xticklabels=["Low Risk","Medium Risk","High Risk"],
    yticklabels=object_labels
)

plt.title("Collision Risk Distribution Across Video (154 Frames)")

plt.xlabel("Risk Level")

plt.ylabel("Object Class")

plt.tight_layout()

plt.savefig(OUTPUT_PATH)

print("Saved heatmap:",OUTPUT_PATH)