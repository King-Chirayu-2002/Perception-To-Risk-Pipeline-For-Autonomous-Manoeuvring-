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

SAVE_DIR = os.path.join(PROJECT_ROOT, "future_scene_output")

CONF_THRESHOLD = 0.4
FPS = 10

os.makedirs(SAVE_DIR,exist_ok=True)


# ========================
# INITIALIZE
# ========================

print("Initializing multi-hypothesis future prediction...")

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

future_generator = FutureFrameGenerator(FPS)

frames = sorted(os.listdir(FRAMES_DIR))

saved = False


# ========================
# MAIN LOOP
# ========================

for frame_name in frames:

    frame = cv2.imread(os.path.join(FRAMES_DIR,frame_name))

    if frame is None:
        continue

    h,w = frame.shape[:2]

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


    # ========================
    # OBJECT PROCESSING
    # ========================

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
    # GENERATE FUTURE SCENES
    # ========================

    futures = future_generator.generate_multi_future(
        frame,
        objects,
        ego_velocity=0,
        steps=5
    )


    # ========================
    # DISPLAY RESULTS
    # ========================

    if not saved and len(tracks) >= 5:

        straight = futures["straight"][-1]
        left = futures["left"][-1]
        right = futures["right"][-1]


        cv2.imwrite(
            os.path.join(SAVE_DIR,"future_straight.png"),
            straight
        )

        cv2.imwrite(
            os.path.join(SAVE_DIR,"future_left.png"),
            left
        )

        cv2.imwrite(
            os.path.join(SAVE_DIR,"future_right.png"),
            right
        )

        print("Saved future predictions in:",SAVE_DIR)

        saved = True


    # show last frame predictions

    straight = futures["straight"][-1]
    left = futures["left"][-1]
    right = futures["right"][-1]

    combined = np.hstack([left,straight,right])

    cv2.imshow("Future Scene Hypotheses",combined)

    if cv2.waitKey(30) == ord('q'):
        break


cv2.destroyAllWindows()