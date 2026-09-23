import os
import cv2
import numpy as np
from ultralytics import YOLO

from sort import Sort
from depth_module import DepthEstimator
from camera_model import KITTICameraModel
from bbox_3d_estimator import BoundingBox3DEstimator
from trajectory_predictor import TrajectoryPredictor
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR
from ttc_montecarlo import TTCMonteCarlo


# =========================
# CONFIG
# =========================

FRAMES_DIR = KITTI_FRAMES_DIR

SAVE_PATH = os.path.join(PROJECT_ROOT, "bev_planner_result.png")

CONF_THRESHOLD = 0.4
FPS = 10


# =========================
# INITIALIZATION
# =========================

print("Initializing BEV planner...")

model = YOLO("yolov8n.pt")
tracker = Sort()

depth_estimator = DepthEstimator()
camera_model = KITTICameraModel()

bbox3d = BoundingBox3DEstimator(camera_model)

trajectory_predictor = TrajectoryPredictor(fps=FPS)

ttc_mc = TTCMonteCarlo()

frames = sorted(os.listdir(FRAMES_DIR))

saved = False


# =========================
# BEV MAP
# =========================

def create_bev():

    bev = np.zeros((650,650,3),dtype=np.uint8)

    # road
    cv2.rectangle(bev,(285,0),(365,650),(40,40,40),-1)

    cv2.line(bev,(285,0),(285,650),(120,120,120),2)
    cv2.line(bev,(365,0),(365,650),(120,120,120),2)

    # ego vehicle
    cv2.rectangle(bev,(310,570),(340,620),(255,255,0),-1)

    # distance rings
    for r in [120,240,360]:
        cv2.circle(bev,(325,595),r,(70,70,70),1)

    cv2.putText(
        bev,
        "BEV Planner View",
        (220,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (200,200,200),
        2
    )

    return bev


# =========================
# MAIN LOOP
# =========================

for frame_name in frames:

    frame = cv2.imread(os.path.join(FRAMES_DIR,frame_name))

    if frame is None:
        continue

    bev = create_bev()

    depth_map = depth_estimator.predict_depth(frame)

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


        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)


        depth = bbox3d.estimate_depth(depth_map,x1,y1,x2,y2)

        P,L,W,H = bbox3d.compute_3d_box(x1,y1,x2,y2,depth)


        # =========================
        # BEV PROJECTION
        # =========================

        scale = 7

        bx = int(325 + P[0]*scale)
        by = int(595 - P[2]*scale)

        bx = np.clip(bx,0,649)
        by = np.clip(by,0,649)


        # =========================
        # COLLISION RISK
        # =========================

        obj_pos = (cx,cy)
        obj_vel = (vx,vy)
        ego_pos = (frame.shape[1]//2,frame.shape[0]-20)

        ttc_samples = ttc_mc.compute_ttc_samples(
            obj_pos,obj_vel,ego_pos
        )

        collision_prob = ttc_mc.collision_probability(ttc_samples)


        color = (
            int(255*collision_prob),
            int(255*(1-collision_prob)),
            0
        )


        # =========================
        # DRAW OBJECT
        # =========================

        cv2.circle(bev,(bx,by),9,color,-1)

        cv2.putText(
            bev,
            f"ID {tid}",
            (bx+5,by-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255,255,255),
            1
        )


        # =========================
        # VELOCITY ARROW
        # =========================

        ex = int(bx + vx*0.03)
        ey = int(by - vy*0.03)

        cv2.arrowedLine(
            bev,
            (bx,by),
            (ex,ey),
            (255,255,255),
            2
        )


        # =========================
        # FUTURE TRAJECTORY
        # =========================

        future = trajectory_predictor.predict_trajectory(tid)

        for p in future:

            fx = int(bx + (p[0]-cx)*0.15)
            fy = int(by - (p[1]-cy)*0.15)

            if 0 <= fx < 650 and 0 <= fy < 650:

                cv2.circle(bev,(fx,fy),3,(0,255,255),-1)


    # =========================
    # SAVE RESULT
    # =========================

    if not saved and len(tracks) >= 5:

        cv2.imwrite(SAVE_PATH,bev)

        print("Saved BEV planner:",SAVE_PATH)

        saved = True


    cv2.imshow("BEV Planner",bev)

    if cv2.waitKey(30) == ord('q'):
        break


cv2.destroyAllWindows()