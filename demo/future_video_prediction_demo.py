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


# =========================
# CONFIG
# =========================

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "future_prediction_video.mp4")

CONF_THRESHOLD = 0.4
FPS = 10
FUTURE_STEPS = 5


# =========================
# INITIALIZATION
# =========================

print("Initializing future video prediction...")

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)
future_generator = FutureFrameGenerator(FPS)

frames = sorted(os.listdir(FRAMES_DIR))

last_frame = None
objects = []


# =========================
# PROCESS VIDEO
# =========================

for frame_name in frames:

    frame = cv2.imread(os.path.join(FRAMES_DIR, frame_name))

    if frame is None:
        continue

    last_frame = frame.copy()

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


# =========================
# CREATE VIDEO WRITER
# =========================

h,w = last_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    2,          # slow fps for visualization
    (w,h)
)


# =========================
# GENERATE FUTURE FRAMES
# =========================

print("Generating future frames...")

base = last_frame.copy()

for step in range(1, FUTURE_STEPS+1):

    frame_future = base.copy()

    for obj in objects:

        patch = obj["patch"]
        mask = obj["mask"]

        cx,cy = obj["center"]
        vx,vy = obj["velocity"]

        # move object forward
        new_x = int(cx + vx*step)
        new_y = int(cy + vy*step)

        frame_future = future_generator.paste_object(
            frame_future,
            patch,
            mask,
            (new_x,new_y)
        )

    video.write(frame_future)

    cv2.imshow("Future Prediction", frame_future)

    if cv2.waitKey(600) == ord('q'):
        break


video.release()
cv2.destroyAllWindows()

print("Saved future prediction video:", OUTPUT_VIDEO)