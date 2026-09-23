import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

from sort import Sort
from depth_module import DepthEstimator
from ego_motion import EgoMotionEstimator
from camera_model import KITTICameraModel
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR

from trajectory_predictor import TrajectoryPredictor
from probabilistic_predictor import ProbabilisticPredictor
from collision_predictor import CollisionPredictor
from bbox_3d_estimator import BoundingBox3DEstimator
from lane_detector import LaneDetector
from ttc_montecarlo import TTCMonteCarlo
from future_frame_generator import FutureFrameGenerator

from evaluation_metrics import EvaluationMetrics


# ==============================
# CONFIG
# ==============================

FRAMES_DIR = KITTI_FRAMES_DIR
OXTS_DIR = KITTI_OXTS_DIR

FPS = 10
CONF_THRESHOLD = 0.4


# ==============================
# INITIALIZATION
# ==============================

print("Initializing physics-aware scene system...")

model = YOLO("yolov8n.pt")

tracker = Sort()

depth_estimator = DepthEstimator()

ego_motion = EgoMotionEstimator(OXTS_DIR, fps=FPS)

camera_model = KITTICameraModel()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

prob_predictor = ProbabilisticPredictor(fps=FPS)

collision_predictor = CollisionPredictor()

bbox3d = BoundingBox3DEstimator(camera_model)

lane_detector = LaneDetector()

ttc_mc = TTCMonteCarlo()

future_generator = FutureFrameGenerator(FPS)

metrics = EvaluationMetrics()

frames = sorted(os.listdir(FRAMES_DIR))

frame_count = 0


# ==============================
# BEV PLANNER FUNCTIONS
# ==============================

def create_bev_map():

    bev = np.zeros((600,600,3), dtype=np.uint8)

    cv2.rectangle(bev,(260,0),(340,600),(40,40,40),-1)

    cv2.line(bev,(260,0),(260,600),(120,120,120),2)
    cv2.line(bev,(340,0),(340,600),(120,120,120),2)

    cv2.rectangle(bev,(285,520),(315,580),(255,255,0),-1)

    for r in [80,160,240]:
        cv2.circle(bev,(300,550),r,(60,60,60),1)

    cv2.putText(bev,"BEV Planner",(210,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(200,200,200),2)

    return bev


def draw_bev_object(bev, x, z, prob):

    bx = int(300 + x*8)
    by = int(550 - z*8)

    if 0 <= bx < 600 and 0 <= by < 600:

        color = (int(255*prob), int(255*(1-prob)), 0)

        cv2.circle(bev,(bx,by),8,color,-1)


def draw_velocity_arrow(bev, x, z, vx, vz):

    bx = int(300 + x*8)
    by = int(550 - z*8)

    ex = int(bx + vx*2)
    ey = int(by - vz*2)

    cv2.arrowedLine(bev,(bx,by),(ex,ey),(255,255,255),2)


def draw_bev_traj(bev, traj):

    for p in traj:

        bx = int(300 + p[0]*0.2)
        by = int(550 - p[1]*0.2)

        if 0<=bx<600 and 0<=by<600:

            cv2.circle(bev,(bx,by),3,(0,0,255),-1)


# ==============================
# MAIN LOOP
# ==============================

for frame_idx, frame_name in enumerate(frames[:300]):

    frame = cv2.imread(os.path.join(FRAMES_DIR, frame_name))

    if frame is None:
        continue

    frame_count += 1

    h, w = frame.shape[:2]

    bev = create_bev_map()

    depth_map = depth_estimator.predict_depth(frame)

    ego_velocity = ego_motion.get_ego_velocity(frame_idx)

    ego_x = w//2
    ego_y = h-20


    lanes = lane_detector.detect_lanes(frame)

    frame = lane_detector.draw_lanes(frame, lanes)


    results = model(frame, conf=CONF_THRESHOLD)[0]

    detections = []

    for box in results.boxes:

        label = model.names[int(box.cls[0])]

        if label not in ["car","person","truck","bus","bicycle"]:
            continue

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        detections.append((x1,y1,x2,y2,label))


    dets_np = np.array(
        [[x1,y1,x2,y2,0.9] for x1,y1,x2,y2,_ in detections]
    ) if detections else np.empty((0,5))

    tracks = tracker.update(dets_np)

    metrics.update_tracks(tracks)


    risk_map = np.zeros((h,w),dtype=np.float32)

    risk_bev = np.zeros((600,600),dtype=np.float32)

    objects = []


    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        trajectory_predictor.update(tid,cx,cy)

        future_path = trajectory_predictor.predict_trajectory(tid)

        vx,vy = trajectory_predictor.get_velocity(tid)


        prob_paths = prob_predictor.sample_trajectories(cx,cy,vx,vy)

        for path in prob_paths:
            for pt in path:
                cv2.circle(frame,pt,2,(0,165,255),-1)


        depth = bbox3d.estimate_depth(depth_map,x1,y1,x2,y2)

        P,L,W,H = bbox3d.compute_3d_box(x1,y1,x2,y2,depth)

        bbox3d.draw_3d_box(frame,x1,y1,x2,y2)


        obj_pos = (cx,cy)
        obj_vel = (vx,vy)
        ego_pos = (ego_x,ego_y)

        ttc_samples = ttc_mc.compute_ttc_samples(obj_pos,obj_vel,ego_pos)

        collision_prob = ttc_mc.collision_probability(ttc_samples)


        if len(ttc_samples) > 0:

            metrics.add_ttc(np.mean(ttc_samples))


        radius = int(50 + 120*collision_prob)

        cv2.circle(risk_map,(cx,cy),radius,collision_prob,-1)


        draw_bev_object(bev,P[0],P[2],collision_prob)

        draw_velocity_arrow(bev,P[0],P[2],vx,vy)

        draw_bev_traj(bev,future_path)


        bx = int(300 + P[0]*8)
        by = int(550 - P[2]*8)

        if 0<=bx<600 and 0<=by<600:

            cv2.circle(risk_bev,(bx,by),
                       int(40+80*collision_prob),
                       collision_prob,-1)


        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(frame,f"ID {tid}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(0,255,0),2)


        patch, mask = future_generator.extract_object(
            frame,(x1,y1,x2,y2)
        )

        if patch is not None:

            objects.append({
                "patch":patch,
                "mask":mask,
                "center":(cx,cy),
                "velocity":(vx/FPS,vy/FPS)
            })


    if risk_map.max() > 0:

        norm = risk_map/(risk_map.max()+1e-6)

        heatmap = cv2.applyColorMap(
            (norm*255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        frame = cv2.addWeighted(frame,0.8,heatmap,0.4,0)


    if risk_bev.max() > 0:

        norm = risk_bev/(risk_bev.max()+1e-6)

        heat = cv2.applyColorMap(
            (norm*255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        bev = cv2.addWeighted(bev,0.7,heat,0.5,0)


    cv2.circle(frame,(ego_x,ego_y),6,(255,255,0),-1)

    cv2.imshow("Scene Understanding", frame)

    cv2.imshow("Dynamic BEV Risk Planner", bev)


    if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()


# ==============================
# FINAL EVALUATION
# ==============================

print("\nTotal frames processed:", frame_count)

metrics.print_results()

metrics.plot_ttc_graph()

metrics.plot_global_trajectories()