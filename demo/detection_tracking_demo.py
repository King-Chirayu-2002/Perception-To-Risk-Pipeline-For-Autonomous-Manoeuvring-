import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR

FRAMES_DIR = KITTI_FRAMES_DIR
SAVE_PATH = os.path.join(PROJECT_ROOT, "detection_tracking_result.png")

model = YOLO("yolov8n.pt")
tracker = Sort()

frames = sorted(os.listdir(FRAMES_DIR))

saved = False

for frame_name in frames:

    frame_path = os.path.join(FRAMES_DIR, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    results = model(frame, conf=0.4)[0]

    detections = []

    for box in results.boxes:

        label = model.names[int(box.cls[0])]

        if label not in ["car","person","truck","bus","bicycle"]:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        detections.append([x1,y1,x2,y2,0.9])

    dets_np = np.array(detections) if detections else np.empty((0,5))

    tracks = tracker.update(dets_np)

    # draw boxes
    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,f"ID {tid}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,255,0),2)

    # save a good frame automatically
    if not saved and len(tracks) >= 5:

        cv2.imwrite(SAVE_PATH, frame)

        print("Saved best frame as:", SAVE_PATH)

        saved = True

    cv2.imshow("Detection + Tracking", frame)

    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()