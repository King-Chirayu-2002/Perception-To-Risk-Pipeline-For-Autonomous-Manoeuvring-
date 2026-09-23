import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from sort import Sort

# your modules
from trajectory_predictor import TrajectoryPredictor
from ttc_montecarlo import TTCMonteCarlo
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR

# ConvLSTM model
from convlstm_predict_trained import ConvLSTM


# ==============================
# CONFIG
# ==============================

FRAMES_DIR = KITTI_FRAMES_DIR

MODEL_PATH = CONVLSTM_MODEL_PATH

FPS = 10
FUTURE_TIME = 2.0

frames = sorted(os.listdir(FRAMES_DIR))


# ==============================
# LOAD MODELS
# ==============================

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo = YOLO("yolov8n.pt")

tracker = Sort()

trajectory_predictor = TrajectoryPredictor(fps=FPS)

ttc_mc = TTCMonteCarlo()

convlstm = ConvLSTM().to(device)
convlstm.load_state_dict(torch.load(MODEL_PATH,map_location=device))
convlstm.eval()


track_history = {}

frame_buffer = []


# ==============================
# MAIN LOOP
# ==============================

for f in frames[:300]:

    frame = cv2.imread(os.path.join(FRAMES_DIR,f))

    h,w = frame.shape[:2]

    frame_buffer.append(frame)

    if len(frame_buffer) > 4:
        frame_buffer.pop(0)

    # panel copies
    detection_view = frame.copy()
    traj_view = frame.copy()
    risk_view = frame.copy()
    future_view = frame.copy()


    # =========================
    # OBJECT DETECTION
    # =========================

    results = yolo(frame)[0]

    detections = []

    for box in results.boxes:

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        detections.append([x1,y1,x2,y2,0.9])

        cv2.rectangle(
            detection_view,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )

    dets = np.array(detections) if detections else np.empty((0,5))

    tracks = tracker.update(dets)


    # =========================
    # TRAJECTORY + RISK
    # =========================

    risk_map = np.zeros((h,w),dtype=np.float32)

    ego_x = w//2
    ego_y = h-10


    for trk in tracks:

        x1,y1,x2,y2,tid = trk.astype(int)

        cx = (x1+x2)//2
        cy = (y1+y2)//2

        if tid not in track_history:
            track_history[tid] = []

        track_history[tid].append((cx,cy))

        if len(track_history[tid]) > 20:
            track_history[tid].pop(0)

        hist = track_history[tid]

        for i in range(1,len(hist)):

            cv2.line(
                traj_view,
                hist[i-1],
                hist[i],
                (255,0,0),
                2
            )

        trajectory_predictor.update(tid,cx,cy)

        vx,vy = trajectory_predictor.get_velocity(tid)

        future_x = int(cx + vx*FUTURE_TIME/FPS)
        future_y = int(cy + vy*FUTURE_TIME/FPS)

        cv2.line(
            traj_view,
            (cx,cy),
            (future_x,future_y),
            (0,0,255),
            2
        )

        cv2.circle(traj_view,(future_x,future_y),5,(0,0,255),-1)


        # TTC risk

        samples = ttc_mc.compute_ttc_samples(
            (cx,cy),(vx,vy),(ego_x,ego_y)
        )

        prob = ttc_mc.collision_probability(samples)

        radius = int(40 + prob*120)

        cv2.circle(
            risk_map,
            (future_x,future_y),
            radius,
            prob,
            -1
        )


    if risk_map.max() > 0:

        norm = risk_map/(risk_map.max()+1e-6)

        heatmap = cv2.applyColorMap(
            (norm*255).astype(np.uint8),
            cv2.COLORMAP_JET
        )

        risk_view = cv2.addWeighted(
            risk_view,
            0.8,
            heatmap,
            0.4,
            0
        )


    # =========================
    # FUTURE FRAME PREDICTION
    # =========================

    if len(frame_buffer) == 4:

        seq = []

        for fr in frame_buffer:

            img = cv2.resize(fr,(256,128))

            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

            img = img/255.0

            seq.append(img)

        seq = np.stack(seq)

        seq = torch.tensor(seq).permute(0,3,1,2).unsqueeze(0).float()

        seq = seq.to(device)

        with torch.no_grad():

            pred = convlstm(seq)

        pred = pred.squeeze().permute(1,2,0).cpu().numpy()

        pred = (pred*255).astype(np.uint8)

        future_view = cv2.resize(pred,(w,h))


    # =========================
    # BEV VIEW
    # =========================

    bev = np.zeros((h,w,3),dtype=np.uint8)

    cv2.rectangle(bev,(w//2-30,h-80),(w//2+30,h),(255,255,0),-1)

    cv2.putText(
        bev,
        "BEV Planner",
        (30,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )


    # =========================
    # BUILD DASHBOARD
    # =========================

    top = np.hstack([detection_view,traj_view])

    bottom = np.hstack([future_view,bev])

    dashboard = np.vstack([top,bottom])

    cv2.imshow("Autonomous Driving Dashboard",dashboard)

    if cv2.waitKey(30)==ord('q'):
        break


cv2.destroyAllWindows()