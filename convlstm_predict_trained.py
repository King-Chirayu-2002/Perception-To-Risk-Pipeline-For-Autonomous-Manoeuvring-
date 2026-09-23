import os
import cv2
import torch
import numpy as np
import imageio
import torch.nn as nn
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR


# ==========================
# ConvLSTM Cell
# ==========================

class ConvLSTMCell(nn.Module):

    def __init__(self, in_dim, hid_dim):
        super().__init__()

        self.conv = nn.Conv2d(
            in_dim + hid_dim,
            4 * hid_dim,
            3,
            padding=1
        )

        self.hid_dim = hid_dim

    def forward(self, x, h, c):

        combined = torch.cat([x, h], 1)

        gates = self.conv(combined)

        i, f, o, g = torch.split(
            gates,
            self.hid_dim,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


# ==========================
# ConvLSTM Model
# ==========================

class ConvLSTM(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Conv2d(3, 32, 3, padding=1)

        self.lstm = ConvLSTMCell(32, 32)

        self.decoder = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, seq):

        B, T, C, H, W = seq.shape

        h = torch.zeros(B, 32, H, W).to(seq.device)
        c = torch.zeros(B, 32, H, W).to(seq.device)

        for t in range(T):

            x = self.encoder(seq[:, t])
            h, c = self.lstm(x, h, c)

        out = self.decoder(h)

        return torch.sigmoid(out)


# ==========================
# CONFIG
# ==========================

FRAMES_DIR = KITTI_FRAMES_DIR

MODEL_PATH = CONVLSTM_MODEL_PATH

OUTPUT_DIR = CONVLSTM_PREDICTION_DIR

INPUT_FRAMES = 4
FUTURE_FRAMES = 5
IMAGE_SIZE = (256,128)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================
# LOAD FRAMES
# ==========================

frame_files = sorted(os.listdir(FRAMES_DIR))

frames = []

for f in frame_files[-INPUT_FRAMES:]:

    img = cv2.imread(os.path.join(FRAMES_DIR, f))

    img = cv2.resize(img, IMAGE_SIZE)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img / 255.0

    frames.append(img)

frames = np.stack(frames)

frames = torch.tensor(frames).float()

frames = frames.permute(0,3,1,2)

frames = frames.unsqueeze(0)


# ==========================
# LOAD TRAINED MODEL
# ==========================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvLSTM().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

frames = frames.to(device)


# ==========================
# FUTURE PREDICTION
# ==========================

generated = []

current = frames

with torch.no_grad():

    for i in range(FUTURE_FRAMES):

        pred = model(current)

        generated.append(pred.cpu())

        pred = pred.unsqueeze(1)

        current = torch.cat([current[:,1:], pred], dim=1)


# ==========================
# SAVE FRAMES
# ==========================

video_frames = []

for i,f in enumerate(generated):

    img = f.squeeze().permute(1,2,0).numpy()

    img = (img*255).astype(np.uint8)

    path = os.path.join(
        OUTPUT_DIR,
        f"future_{i+1}.png"
    )

    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    video_frames.append(img)

    print("Saved:", path)


# ==========================
# SAVE VIDEO
# ==========================

video_path = os.path.join(
    OUTPUT_DIR,
    "convlstm_future_video.mp4"
)

writer = imageio.get_writer(
    video_path,
    format="FFMPEG",
    fps=2
)

for f in video_frames:
    writer.append_data(f)

writer.close()

print("Saved video:", video_path)