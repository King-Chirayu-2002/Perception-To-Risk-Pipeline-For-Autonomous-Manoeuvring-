import os
import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort
from trajectory_predictor import TrajectoryPredictor
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR


# ========================
# CONFIG
# ========================

FRAMES_DIR = KITTI_FRAMES_DIR

SAVE_PATH = os.path.join(PROJECT_ROOT, "clean_predicted_frame.png")

CONF_THRESHOLD = 0.4
FPS = 10


# ========================
# INITIALIZATION
# ========================

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

frames = sorted(os.listdir(FRAMES_DIR))

last_frame = None
objects = []


# ========================
# PROCESS VIDEO
# ========================

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

        objects.append({
            "bbox":(x1,y1,x2,y2),
            "center":(cx,cy),
            "velocity":(vx/FPS,vy/FPS)
        })


# ========================
# BACKGROUND CLEANING
# ========================

background = last_frame.copy()

mask = np.zeros(background.shape[:2],dtype=np.uint8)

for obj in objects:

    x1,y1,x2,y2 = obj["bbox"]

    cv2.rectangle(mask,(x1,y1),(x2,y2),255,-1)


# inpaint removed objects
background = cv2.inpaint(
    background,
    mask,
    5,
    cv2.INPAINT_TELEA
)


# ========================
# GENERATE FUTURE FRAME
# ========================

future = background.copy()

for obj in objects:

    x1,y1,x2,y2 = obj["bbox"]

    cx,cy = obj["center"]

    vx,vy = obj["velocity"]

    patch = last_frame[y1:y2,x1:x2]

    if patch.size == 0:
        continue

    h,w = patch.shape[:2]

    new_x = int(cx + vx)
    new_y = int(cy + vy)

    x_start = int(new_x - w/2)
    y_start = int(new_y - h/2)

    x_end = x_start + w
    y_end = y_start + h

    if x_start<0 or y_start<0 or x_end>=future.shape[1] or y_end>=future.shape[0]:
        continue

    future[y_start:y_end,x_start:x_end] = patch


# ========================
# SAVE RESULT
# ========================

cv2.imwrite(SAVE_PATH,future)

print("Saved clean predicted frame:",SAVE_PATH)

cv2.imshow("Clean Future Prediction",future)

cv2.waitKey(0)
cv2.destroyAllWindows()