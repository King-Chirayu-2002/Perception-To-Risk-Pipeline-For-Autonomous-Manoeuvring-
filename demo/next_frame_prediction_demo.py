import os
import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort
from trajectory_predictor import TrajectoryPredictor
from future_frame_generator import FutureFrameGenerator
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR


# ========================
# CONFIG
# ========================

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "future_frames")

CONF_THRESHOLD = 0.4
FPS = 10

NUM_FUTURE_FRAMES = 3

os.makedirs(OUTPUT_DIR,exist_ok=True)


# ========================
# INITIALIZATION
# ========================

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

future_generator = FutureFrameGenerator(FPS)

frames = sorted(os.listdir(FRAMES_DIR))

last_frame = None
objects = []


# ========================
# PROCESS VIDEO
# ========================

for frame_name in frames:

    frame = cv2.imread(os.path.join(FRAMES_DIR,frame_name))

    if frame is None:
        continue

    last_frame = frame.copy()

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

    objects = []

    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)

        patch,mask = future_generator.extract_object(
            frame,(x1,y1,x2,y2)
        )

        if patch is not None:

            objects.append({
                "patch":patch,
                "mask":mask,
                "center":(cx,cy),
                "velocity":(vx/FPS,vy/FPS)
            })


# ========================
# GENERATE FUTURE FRAMES
# ========================

print("Generating future frames...")

for t in range(1,NUM_FUTURE_FRAMES+1):

    predicted = last_frame.copy()

    for obj in objects:

        patch = obj["patch"]
        mask = obj["mask"]

        cx,cy = obj["center"]
        vx,vy = obj["velocity"]

        # move object further for each future frame
        new_x = int(cx + vx*t*2)
        new_y = int(cy + vy*t*2)

        predicted = future_generator.paste_object(
            predicted,
            patch,
            mask,
            (new_x,new_y)
        )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"predicted_t+{t}.png"
    )

    cv2.imwrite(save_path,predicted)

    print("Saved:",save_path)

    cv2.imshow(f"Future Frame t+{t}",predicted)


cv2.waitKey(0)
cv2.destroyAllWindows()