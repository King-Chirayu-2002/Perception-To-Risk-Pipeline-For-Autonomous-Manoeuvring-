import os
import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort
from trajectory_predictor import TrajectoryPredictor
from probabilistic_predictor import ProbabilisticPredictor
from ttc_montecarlo import TTCMonteCarlo
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR


# ======================
# CONFIG
# ======================

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "physics_scene_demo.mp4")

CONF_THRESHOLD = 0.4
FPS = 10

# ======================
# INITIALIZE
# ======================

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)
prob_predictor = ProbabilisticPredictor(fps=FPS)

ttc_mc = TTCMonteCarlo()

frames = sorted(os.listdir(FRAMES_DIR))

track_history = {}

video_writer = None

print("Running physics-aware scene demo...")

# ======================
# MAIN LOOP
# ======================

for frame_idx, frame_name in enumerate(frames[:300]):

    frame = cv2.imread(os.path.join(FRAMES_DIR, frame_name))

    if frame is None:
        continue

    h, w = frame.shape[:2]

    ego_x = w//2
    ego_y = h-20

    risk_map = np.zeros((h,w),dtype=np.float32)

    if video_writer is None:

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            FPS,
            (w,h)
        )

    # ======================
    # YOLO DETECTION
    # ======================

    results = model(frame, conf=CONF_THRESHOLD)[0]

    detections = []

    for box in results.boxes:

        label = model.names[int(box.cls[0])]

        if label not in ["car","person","truck","bus","bicycle"]:
            continue

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        detections.append([x1,y1,x2,y2,0.9])

    dets_np = np.array(detections) if detections else np.empty((0,5))

    tracks = tracker.update(dets_np)

    # ======================
    # PROCESS OBJECTS
    # ======================

    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        # store history
        if tid not in track_history:
            track_history[tid] = []

        track_history[tid].append((cx,cy))

        if len(track_history[tid]) > 20:
            track_history[tid].pop(0)

        history = track_history[tid]

        # ======================
        # DRAW TRAJECTORY HISTORY
        # ======================

        for i in range(1,len(history)):

            cv2.line(
                frame,
                history[i-1],
                history[i],
                (255,0,0),
                2
            )

        # ======================
        # VELOCITY
        # ======================

        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)

        # ======================
        # FUTURE TRAJECTORY
        # ======================

        future_path = trajectory_predictor.predict_trajectory(tid)

        for i in range(1,len(future_path)):

            cv2.line(
                frame,
                future_path[i-1],
                future_path[i],
                (0,0,255),
                2
            )

        # ======================
        # PROBABILISTIC TRAJECTORIES
        # ======================

        prob_paths = prob_predictor.sample_trajectories(cx,cy,vx,vy)

        for path in prob_paths:

            for pt in path:

                cv2.circle(frame,pt,1,(0,165,255),-1)

        # ======================
        # COLLISION RISK
        # ======================

        obj_pos = (cx,cy)
        obj_vel = (vx,vy)
        ego_pos = (ego_x,ego_y)

        ttc_samples = ttc_mc.compute_ttc_samples(
            obj_pos,
            obj_vel,
            ego_pos
        )

        collision_prob = ttc_mc.collision_probability(ttc_samples)

        radius = int(40 + 120*collision_prob)

        cv2.circle(
            risk_map,
            (cx,cy),
            radius,
            collision_prob,
            -1
        )

        # ======================
        # RISK LABEL
        # ======================

        label = "SAFE"

        if collision_prob > 0.5:
            label = "RISK"

        cv2.putText(
            frame,
            f"{label} P={collision_prob:.2f}",
            (x1,y2+20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,0,255) if label=="RISK" else (0,255,0),
            2
        )

        # ======================
        # BOUNDING BOX
        # ======================

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"ID {tid}",
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    # ======================
    # RISK HEATMAP
    # ======================

    if risk_map.max() > 0:

        norm = risk_map/(risk_map.max()+1e-6)

        heatmap = cv2.applyColorMap(
            (norm*255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        frame = cv2.addWeighted(frame,0.8,heatmap,0.4,0)

    # ======================
    # EGO VEHICLE
    # ======================

    cv2.circle(frame,(ego_x,ego_y),8,(255,255,0),-1)

    cv2.putText(
        frame,
        "EGO",
        (ego_x-20,ego_y-15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255,255,0),
        2
    )

    # ======================
    # TITLE
    # ======================

    cv2.putText(
        frame,
        "Physics-Aware Scene Understanding",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    video_writer.write(frame)

    cv2.imshow("Physics Scene Demo",frame)

    if cv2.waitKey(30)==ord('q'):
        break


video_writer.release()

cv2.destroyAllWindows()

print("Demo video saved:",OUTPUT_VIDEO)