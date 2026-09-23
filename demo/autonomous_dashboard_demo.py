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

# ======================
# CONFIG
# ======================

FRAMES_DIR = KITTI_FRAMES_DIR
OUTPUT_VIDEO = os.path.join(PROJECT_ROOT, "autonomous_dashboard_demo.mp4")

CONF_THRESHOLD = 0.4
FPS = 10
FUTURE_TIME = 2.0

# ======================
# INITIALIZE
# ======================

model = YOLO("yolov8n.pt")
tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)
ttc_mc = TTCMonteCarlo()

frames = sorted(os.listdir(FRAMES_DIR))
track_history = {}

video_writer = None

# ======================
# BEV MAP FUNCTION
# ======================

def create_bev():
    bev = np.zeros((400,400,3),dtype=np.uint8)

    # road
    cv2.rectangle(bev,(170,0),(230,400),(50,50,50),-1)

    # ego vehicle
    cv2.rectangle(bev,(190,330),(210,360),(255,255,0),-1)

    cv2.putText(bev,"BEV Planner",(110,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(200,200,200),2)

    return bev


# ======================
# MAIN LOOP
# ======================

for frame_name in frames[:300]:

    frame = cv2.imread(os.path.join(FRAMES_DIR,frame_name))
    if frame is None:
        continue

    h,w = frame.shape[:2]

    ego_x = w//2
    ego_y = h-10

    risk_map = np.zeros((h,w),dtype=np.float32)
    bev = create_bev()

    if video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            OUTPUT_VIDEO,fourcc,FPS,(w+400,h)
        )

    # ======================
    # DETECTION
    # ======================

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

    # ======================
    # PROCESS OBJECTS
    # ======================

    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        # track history
        if tid not in track_history:
            track_history[tid] = []

        track_history[tid].append((cx,cy))

        if len(track_history[tid]) > 20:
            track_history[tid].pop(0)

        history = track_history[tid]

        # trajectory history
        for i in range(1,len(history)):
            cv2.line(frame,history[i-1],history[i],(255,0,0),2)

        trajectory_predictor.update(tid,cx,cy)
        vx,vy = trajectory_predictor.get_velocity(tid)

        # future trajectory
        future_x = int(cx + vx*FUTURE_TIME/FPS)
        future_y = int(cy + vy*FUTURE_TIME/FPS)

        cv2.line(frame,(cx,cy),(future_x,future_y),(0,0,255),2)
        cv2.circle(frame,(future_x,future_y),5,(0,0,255),-1)

        # TTC collision risk
        ttc_samples = ttc_mc.compute_ttc_samples(
            (cx,cy),(vx,vy),(ego_x,ego_y)
        )

        collision_prob = ttc_mc.collision_probability(ttc_samples)

        radius = int(40 + 120*collision_prob)

        cv2.circle(
            risk_map,
            (future_x,future_y),
            radius,
            collision_prob,
            -1
        )

        # draw bounding box
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame,f"ID {tid}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(0,255,0),2)

        # BEV projection
        bx = int(200 + (cx-w/2)*0.2)
        by = int(350 - cy*0.2)

        if 0<=bx<400 and 0<=by<400:
            cv2.circle(bev,(bx,by),6,(0,0,255),-1)

    # ======================
    # RISK HEATMAP
    # ======================

    if risk_map.max()>0:
        norm = risk_map/(risk_map.max()+1e-6)
        heatmap = cv2.applyColorMap(
            (norm*255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        frame = cv2.addWeighted(frame,0.8,heatmap,0.4,0)

    # ======================
    # SAFE PATH
    # ======================

    for y in range(ego_y,0,-20):
        cv2.circle(frame,(ego_x,y),3,(0,255,0),-1)

    cv2.circle(frame,(ego_x,ego_y),8,(255,255,0),-1)

    cv2.putText(frame,"EGO",
                (ego_x-20,ego_y-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,(255,255,0),2)

    cv2.putText(frame,"Autonomous Driving Dashboard",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,(255,255,255),2)

    # combine camera + BEV
    combined = np.zeros((h,w+400,3),dtype=np.uint8)
    combined[:,:w] = frame
    combined[:,w:] = cv2.resize(bev,(400,h))

    video_writer.write(combined)

    cv2.imshow("Autonomous Dashboard",combined)

    if cv2.waitKey(30)==ord('q'):
        break

video_writer.release()
cv2.destroyAllWindows()

print("Saved dashboard demo:",OUTPUT_VIDEO)