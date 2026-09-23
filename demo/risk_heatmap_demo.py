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

# ========================
# CONFIG
# ========================

FRAMES_DIR = KITTI_FRAMES_DIR

SAVE_PATH = os.path.join(PROJECT_ROOT, "risk_heatmap_overlay.png")

CONF_THRESHOLD = 0.4
FPS = 10

# ========================
# INITIALIZE
# ========================

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)
ttc_mc = TTCMonteCarlo()

frames = sorted(os.listdir(FRAMES_DIR))

saved = False

# ========================
# MAIN LOOP
# ========================

for frame_name in frames:

    frame = cv2.imread(os.path.join(FRAMES_DIR, frame_name))

    if frame is None:
        continue

    h, w = frame.shape[:2]

    ego_x = w // 2
    ego_y = h - 20

    risk_map = np.zeros((h, w), dtype=np.float32)

    # ========================
    # YOLO DETECTION
    # ========================

    results = model(frame, conf=CONF_THRESHOLD)[0]

    detections = []

    for box in results.boxes:

        label = model.names[int(box.cls[0])]

        if label not in ["car","person","truck","bus","bicycle"]:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        detections.append([x1,y1,x2,y2,0.9])

    dets_np = np.array(detections) if detections else np.empty((0,5))

    tracks = tracker.update(dets_np)

    # ========================
    # OBJECT PROCESSING
    # ========================

    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)

        obj_pos = (cx,cy)
        obj_vel = (vx,vy)
        ego_pos = (ego_x,ego_y)

        ttc_samples = ttc_mc.compute_ttc_samples(
            obj_pos,
            obj_vel,
            ego_pos
        )

        collision_prob = ttc_mc.collision_probability(ttc_samples)

        # risk radius proportional to collision probability
        radius = int(40 + 100 * collision_prob)

        cv2.circle(
            risk_map,
            (cx,cy),
            radius,
            collision_prob,
            -1
        )

    # ========================
    # CREATE TRANSPARENT HEATMAP
    # ========================

    if risk_map.max() > 0:

        norm = risk_map / (risk_map.max() + 1e-6)

        heatmap = cv2.applyColorMap(
            (norm*255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        # transparent overlay
        frame_overlay = cv2.addWeighted(
            frame,      # original frame
            0.8,
            heatmap,    # heatmap
            0.35,
            0
        )

    else:
        frame_overlay = frame

    # ========================
    # SAVE GOOD FRAME
    # ========================

    if not saved and len(tracks) >= 10:

        cv2.imwrite(SAVE_PATH, frame_overlay)

        print("Saved heatmap overlay:", SAVE_PATH)

        saved = True

    cv2.imshow("Risk Heatmap Overlay", frame_overlay)

    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()