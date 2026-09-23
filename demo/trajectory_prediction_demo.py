import os
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR


# ========================
# CONFIG
# ========================

FRAMES_DIR = KITTI_FRAMES_DIR

SAVE_PATH = os.path.join(PROJECT_ROOT, "trajectory_pred_result.png")

CONF_THRESHOLD = 0.4
MAX_HISTORY = 20


# ========================
# INITIALIZE
# ========================

model = YOLO("yolov8n.pt")
tracker = Sort()

frames = sorted(os.listdir(FRAMES_DIR))

track_history = {}

saved = False


# ========================
# MAIN LOOP
# ========================

for frame_name in frames:

    frame_path = os.path.join(FRAMES_DIR, frame_name)

    frame = cv2.imread(frame_path)

    if frame is None:
        continue


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
    # TRAJECTORY PROCESSING
    # ========================

    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2


        # ========================
        # STORE HISTORY
        # ========================

        if tid not in track_history:
            track_history[tid] = []

        track_history[tid].append((cx,cy))

        if len(track_history[tid]) > MAX_HISTORY:
            track_history[tid].pop(0)

        history = track_history[tid]


        # ========================
        # DRAW TRAJECTORY HISTORY
        # ========================

        for i in range(1,len(history)):

            cv2.line(
                frame,
                history[i-1],
                history[i],
                (255,0,0),
                2
            )


        # ========================
        # FUTURE PREDICTION
        # ========================

        if len(history) >= 4:

            # average velocity
            dx = (history[-1][0] - history[0][0]) / len(history)
            dy = (history[-1][1] - history[0][1]) / len(history)

            px,py = history[-1]

            future_points = []


            # turning estimation
            turn = (history[-1][0] - history[-2][0]) - (history[-2][0] - history[-3][0])


            for k in range(20):

                px += dx * 2
                py += dy * 2

                # apply slight curvature
                px += turn * 0.3 * k

                future_points.append((int(px),int(py)))


            # ========================
            # DRAW FUTURE PATH
            # ========================

            for i in range(1,len(future_points)):

                cv2.line(
                    frame,
                    future_points[i-1],
                    future_points[i],
                    (0,0,255),
                    2
                )


            # ========================
            # MOTION CONE
            # ========================

            cone_length = 60

            angle = np.arctan2(dy,dx)

            left_angle = angle + np.pi/6
            right_angle = angle - np.pi/6

            left_point = (
                int(cx + cone_length*np.cos(left_angle)),
                int(cy + cone_length*np.sin(left_angle))
            )

            right_point = (
                int(cx + cone_length*np.cos(right_angle)),
                int(cy + cone_length*np.sin(right_angle))
            )

            cone_pts = np.array([
                (cx,cy),
                left_point,
                right_point
            ])

            cv2.fillPoly(frame,[cone_pts],(0,0,255))


        # ========================
        # DRAW BOUNDING BOX
        # ========================

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


    # ========================
    # SAVE BEST FRAME
    # ========================

    if not saved and len(tracks) >= 5:

        cv2.imwrite(SAVE_PATH, frame)

        print("Saved trajectory result:", SAVE_PATH)

        saved = True


    cv2.imshow("Trajectory Prediction", frame)

    if cv2.waitKey(30) == ord('q'):
        break


cv2.destroyAllWindows()