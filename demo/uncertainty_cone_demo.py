import os
import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort
from trajectory_predictor import TrajectoryPredictor
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR


# =====================
# CONFIG
# =====================

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "uncertainty_demo.mp4")

CONF_THRESHOLD = 0.4
FPS = 10


# =====================
# INITIALIZE
# =====================

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

frames = sorted(os.listdir(FRAMES_DIR))

video_writer = None

track_history = {}


# =====================
# DRAW UNCERTAINTY CONE
# =====================

def draw_uncertainty_cone(frame,cx,cy,vx,vy,length=120,spread=0.5):

    speed = np.sqrt(vx**2 + vy**2)

    if speed < 2:
        return

    angle = np.arctan2(vy,vx)

    left_angle = angle + spread
    right_angle = angle - spread

    left = (
        int(cx + length*np.cos(left_angle)),
        int(cy + length*np.sin(left_angle))
    )

    right = (
        int(cx + length*np.cos(right_angle)),
        int(cy + length*np.sin(right_angle))
    )

    pts = np.array([(cx,cy),left,right])

    cv2.polylines(frame,[pts],True,(255,255,0),2)


# =====================
# MAIN LOOP
# =====================

for frame_name in frames[:300]:

    frame = cv2.imread(os.path.join(FRAMES_DIR,frame_name))

    if frame is None:
        continue

    h,w = frame.shape[:2]

    if video_writer is None:

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video_writer = cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            FPS,
            (w,h)
        )

    results = model(frame,conf=CONF_THRESHOLD)[0]

    detections = []

    for box in results.boxes:

        label = model.names[int(box.cls[0])]

        if label not in ["car","person","truck","bus","bicycle"]:
            continue

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        detections.append([x1,y1,x2,y2,0.9])

    dets_np = np.array(detections) if detections else np.empty((0,5))

    tracks = tracker.update(dets_np)


    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2


        if tid not in track_history:
            track_history[tid] = []

        track_history[tid].append((cx,cy))

        if len(track_history[tid]) > 20:
            track_history[tid].pop(0)

        history = track_history[tid]


        for i in range(1,len(history)):

            cv2.line(frame,history[i-1],history[i],(255,0,0),2)


        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)


        future = trajectory_predictor.predict_trajectory(tid)

        for i in range(1,len(future)):

            cv2.line(frame,future[i-1],future[i],(0,0,255),2)


        # DRAW UNCERTAINTY CONE
        draw_uncertainty_cone(frame,cx,cy,vx,vy)


        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,f"ID {tid}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,255,0),2)


    video_writer.write(frame)

    cv2.imshow("Motion Uncertainty Demo",frame)

    if cv2.waitKey(30)==ord('q'):
        break


video_writer.release()
cv2.destroyAllWindows()

print("Saved uncertainty demo:",OUTPUT_VIDEO)