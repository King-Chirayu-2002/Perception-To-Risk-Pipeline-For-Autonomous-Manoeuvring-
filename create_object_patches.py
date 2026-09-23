import os
import cv2
from ultralytics import YOLO
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR

FRAMES_DIR = KITTI_FRAMES_DIR
PATCH_DIR = OBJECT_PATCH_DIR

os.makedirs(PATCH_DIR, exist_ok=True)

model = YOLO("yolov8n.pt")

frames = sorted(os.listdir(FRAMES_DIR))

patch_id = 0

for f in frames:

    img = cv2.imread(os.path.join(FRAMES_DIR,f))

    results = model(img)[0]

    for box in results.boxes:

        label = model.names[int(box.cls[0])]

        if label not in ["car","truck","bus","person","bicycle"]:
            continue

        x1,y1,x2,y2 = map(int,box.xyxy[0])

        patch = img[y1:y2,x1:x2]

        if patch.size == 0:
            continue

        patch = cv2.resize(patch,(64,64))

        path = os.path.join(
            PATCH_DIR,
            f"patch_{patch_id:06d}.png"
        )

        cv2.imwrite(path,patch)

        patch_id += 1

print("Saved patches:",patch_id)