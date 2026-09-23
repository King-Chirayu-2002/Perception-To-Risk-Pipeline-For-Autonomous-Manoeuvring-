import os
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from sort import Sort
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR

# =============================
# CONFIG
# =============================
FRAMES_DIR = KITTI_FRAMES_DIR
CONF_THRESHOLD = 0.4
FPS = 10.0
FUTURE_TIME = 2.0
MAX_HISTORY = 15
SAVE_VIDEO = True
OUTPUT_VIDEO = "kitti_physics_TTC_dual.mp4"
MODE = "CLEAN"  # CLEAN or DEBUG

# =============================
# IOU FUNCTION
# =============================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxA[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# =============================
# EGO-CENTRIC PHYSICS
# =============================
def classify_ego_relation(cx, cy, vx, vy, ego_x, ego_y):
    rel_x = ego_x - cx
    rel_y = ego_y - cy
    dot = rel_x * vx + rel_y * vy
    speed = np.sqrt(vx**2 + vy**2)

    if speed < 5:
        return "Static"
    elif dot > 0:
        return "Approaching Ego"
    elif dot < 0:
        return "Moving Away"
    else:
        return "Crossing Path"

# =============================
# UNCERTAINTY CONE
# =============================
def draw_uncertainty_cone(frame, cx, cy, vx, vy, length=80, spread=0.4):
    speed = np.sqrt(vx**2 + vy**2)
    if speed < 2:
        return

    direction = np.arctan2(vy, vx)
    angles = [direction - spread, direction + spread]

    pts = [(cx, cy)]
    for a in angles:
        px = int(cx + length * np.cos(a))
        py = int(cy + length * np.sin(a))
        pts.append((px, py))

    cv2.polylines(frame, [np.array(pts, np.int32)], True, (255, 255, 0), 2)

# =============================
# MOTION CONSISTENCY SCORE
# =============================
def motion_consistency(history):
    if len(history) < 5:
        return "Analyzing"

    velocities = []
    for i in range(1, len(history)):
        dx = history[i][0] - history[i-1][0]
        dy = history[i][1] - history[i-1][1]
        velocities.append(np.sqrt(dx**2 + dy**2))

    std = np.std(velocities)
    if std < 2:
        return "Consistent"
    elif std < 6:
        return "Moderate"
    else:
        return "Inconsistent"

# =============================
# FAST CONTINUOUS TTC RISK FIELD
# =============================
def update_ttc_risk_field_fast(risk_map, predictions, ego_x, ego_y):
    h, w = risk_map.shape

    for tid, (fx, fy, vx, vy) in predictions.items():
        dx = fx - ego_x
        dy = fy - ego_y
        distance = np.sqrt(dx**2 + dy**2)
        speed = np.sqrt(vx**2 + vy**2) + 1e-5
        ttc = distance / speed

        # Physics-based risk scaling
        if ttc < 0.5:
            risk = 1.0
            radius = 120
        elif ttc < 1.5:
            risk = 0.7
            radius = 100
        elif ttc < 3.0:
            risk = 0.4
            radius = 80
        else:
            risk = 0.15
            radius = 60

        # Draw smooth Gaussian risk blob (FAST using circle + blur)
        temp = np.zeros_like(risk_map, dtype=np.float32)
        cv2.circle(temp, (fx, fy), radius, risk, -1)
        temp = cv2.GaussianBlur(temp, (0, 0), sigmaX=25, sigmaY=25)
        risk_map += temp

    return risk_map

# =============================
# MODELS
# =============================
model = YOLO("yolov8n.pt")
tracker = Sort(max_age=10, min_hits=3, iou_threshold=0.3)

track_history = defaultdict(list)
frames = sorted(os.listdir(FRAMES_DIR))
video_writer = None

print(f"[INFO] Total frames: {len(frames)}")

for frame_name in frames[:300]:
    frame = cv2.imread(os.path.join(FRAMES_DIR, frame_name))
    if frame is None:
        continue

    h, w = frame.shape[:2]
    ego_x, ego_y = w // 2, h - 10

    if SAVE_VIDEO and video_writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w, h))

    risk_map = np.zeros((h, w), dtype=np.float32)

    # YOLO Detection
    results = model(frame, conf=CONF_THRESHOLD)[0]
    detections = []

    for box in results.boxes:
        label = model.names[int(box.cls[0])]
        if label not in ["car", "person", "bicycle", "truck", "bus", "motorcycle"]:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append({"bbox":[x1,y1,x2,y2], "label":label})

    dets_np = np.array([[*d["bbox"], 0.9] for d in detections]) if detections else np.empty((0,5))
    tracks = tracker.update(dets_np)

    predictions = {}

    for trk in tracks:
        x1, y1, x2, y2, tid = trk.astype(int)
        cx, cy = (x1+x2)//2, (y1+y2)//2

        hist = track_history[tid]
        hist.append((cx, cy))
        if len(hist) > MAX_HISTORY:
            hist.pop(0)

        # Associate label
        best_label = "object"
        best_iou = 0
        for det in detections:
            i = iou([x1,y1,x2,y2], det["bbox"])
            if i > best_iou:
                best_iou = i
                best_label = det["label"]

        if len(hist) >= 2:
            (px, py), (qx, qy) = hist[-2], hist[-1]
            vx = (qx-px)*FPS
            vy = (qy-py)*FPS

            ego_relation = classify_ego_relation(cx, cy, vx, vy, ego_x, ego_y)
            consistency = motion_consistency(hist)

            fx = int(cx + vx*FUTURE_TIME)
            fy = int(cy + vy*FUTURE_TIME)
            fx = np.clip(fx, 0, w-1)
            fy = np.clip(fy, 0, h-1)
            predictions[tid] = (fx, fy, vx, vy)

            if MODE == "CLEAN":
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),2)
                cv2.putText(frame,f"{best_label} | {ego_relation}",
                            (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            else:
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
                cv2.putText(frame,f"{best_label} | {ego_relation} | {consistency}",
                            (x1,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)

                cv2.arrowedLine(frame,(cx,cy),
                                (int(cx+vx*0.1),int(cy+vy*0.1)),
                                (0,255,255),2)
                cv2.circle(frame,(fx,fy),6,(0,0,255),-1)
                cv2.line(frame,(cx,cy),(fx,fy),(0,0,255),1)
                draw_uncertainty_cone(frame,cx,cy,vx,vy)

    # ===== CONTINUOUS TTC RISK FIELD =====
    risk_map = update_ttc_risk_field_fast(risk_map, predictions, ego_x, ego_y)

    if risk_map.max() > 0:
        norm = risk_map / (risk_map.max() + 1e-6)
        heatmap = cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_JET)

        if MODE == "CLEAN":
            frame = cv2.addWeighted(frame, 0.85, heatmap, 0.25, 0)
        else:
            frame = cv2.addWeighted(frame, 0.7, heatmap, 0.4, 0)

        cv2.putText(frame,"TTC Risk Field (Physics Map)",
                    (20,80),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    if MODE == "DEBUG":
        cv2.circle(frame,(ego_x,ego_y),8,(0,255,0),-1)
        cv2.putText(frame,"EGO VEHICLE",(ego_x-60,ego_y-15),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.putText(frame,f"MODE: {MODE}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1.0,
                (0,255,255) if MODE=="DEBUG" else (0,200,0),2)

    cv2.imshow("Physics-Consistent Scene Understanding", frame)
    if SAVE_VIDEO:
        video_writer.write(frame)

    key = cv2.waitKey(int(1000/FPS)) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("d"):
        MODE = "DEBUG"
    elif key == ord("c"):
        MODE = "CLEAN"

cv2.destroyAllWindows()
if SAVE_VIDEO:
    video_writer.release()

print("✅ Saved video:", OUTPUT_VIDEO)