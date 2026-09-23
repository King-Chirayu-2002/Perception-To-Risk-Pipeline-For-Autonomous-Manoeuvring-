import os
import cv2
import torch
import numpy as np
import imageio
from torch import nn
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR


# -----------------------------
# SIMPLE SIMVP NETWORK
# -----------------------------

class SimVP(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU()
        )

        self.temporal = nn.Conv3d(
            64,
            64,
            (3,3,3),
            padding=1
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,3,3,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):

        B,T,C,H,W = x.shape

        x = x.view(B*T,C,H,W)

        x = self.encoder(x)

        x = x.view(B,T,64,H,W)

        x = x.permute(0,2,1,3,4)

        x = self.temporal(x)

        x = x.permute(0,2,1,3,4)

        x = x[:,-1]

        x = self.decoder(x)

        return x


# -----------------------------
# CONFIG
# -----------------------------

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_DIR = SIMVP_FUTURE_DIR

INPUT_FRAMES = 4
FUTURE_FRAMES = 5
IMAGE_SIZE = (256,128)

os.makedirs(OUTPUT_DIR,exist_ok=True)


# -----------------------------
# LOAD LAST FRAMES
# -----------------------------

frame_files = sorted(os.listdir(FRAMES_DIR))

frames = []

for f in frame_files[-INPUT_FRAMES:]:

    img = cv2.imread(os.path.join(FRAMES_DIR,f))

    img = cv2.resize(img,IMAGE_SIZE)

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    img = img/255.0

    frames.append(img)

frames = np.stack(frames)

frames = torch.tensor(frames).float()

frames = frames.permute(0,3,1,2)

frames = frames.unsqueeze(0)


# -----------------------------
# LOAD MODEL
# -----------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimVP().to(device)

model.eval()

frames = frames.to(device)


# -----------------------------
# GENERATE FUTURE FRAMES
# -----------------------------

generated = []

current = frames

with torch.no_grad():

    for i in range(FUTURE_FRAMES):

        pred = model(current)

        generated.append(pred.cpu())

        pred = pred.unsqueeze(1)

        current = torch.cat([current[:,1:],pred],dim=1)


# -----------------------------
# SAVE FRAMES
# -----------------------------

video_frames = []

for i,f in enumerate(generated):

    img = f.squeeze().permute(1,2,0).numpy()

    img = (img*255).astype(np.uint8)

    path = os.path.join(
        OUTPUT_DIR,
        f"future_{i+1}.png"
    )

    cv2.imwrite(path,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))

    video_frames.append(img)

    print("Saved:",path)


# -----------------------------
# SAVE VIDEO
# -----------------------------

video_path = os.path.join(
    OUTPUT_DIR,
    "future_prediction.mp4"
)

writer = imageio.get_writer(video_path,fps=2)

for f in video_frames:
    writer.append_data(f)

writer.close()

print("Saved video:",video_path)