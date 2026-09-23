import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR


# ======================
# Dataset
# ======================

class KITTIDataset(Dataset):

    def __init__(self, frames_dir, seq_len=4, size=(256,128)):

        self.frames = sorted(os.listdir(frames_dir))
        self.dir = frames_dir
        self.seq_len = seq_len
        self.size = size

    def __len__(self):
        return len(self.frames) - self.seq_len

    def __getitem__(self, idx):

        seq = []

        for i in range(self.seq_len):

            img = cv2.imread(
                os.path.join(self.dir, self.frames[idx+i])
            )

            img = cv2.resize(img,self.size)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img / 255.0

            seq.append(img)

        target = cv2.imread(
            os.path.join(self.dir, self.frames[idx+self.seq_len])
        )

        target = cv2.resize(target,self.size)
        target = cv2.cvtColor(target,cv2.COLOR_BGR2RGB)
        target = target / 255.0

        seq = torch.tensor(seq).permute(0,3,1,2).float()
        target = torch.tensor(target).permute(2,0,1).float()

        return seq, target


# ======================
# ConvLSTM
# ======================

class ConvLSTMCell(nn.Module):

    def __init__(self, in_dim, hid_dim):

        super().__init__()

        self.conv = nn.Conv2d(
            in_dim + hid_dim,
            4*hid_dim,
            3,
            padding=1
        )

        self.hid_dim = hid_dim

    def forward(self,x,h,c):

        combined = torch.cat([x,h],1)
        gates = self.conv(combined)

        i,f,o,g = torch.split(
            gates,
            self.hid_dim,
            dim=1
        )

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f*c + i*g
        h = o*torch.tanh(c)

        return h,c


class ConvLSTM(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Conv2d(3,32,3,padding=1)

        self.lstm = ConvLSTMCell(32,32)

        self.decoder = nn.Conv2d(32,3,3,padding=1)

    def forward(self,seq):

        B,T,C,H,W = seq.shape

        h = torch.zeros(B,32,H,W).to(seq.device)
        c = torch.zeros(B,32,H,W).to(seq.device)

        for t in range(T):

            x = self.encoder(seq[:,t])
            h,c = self.lstm(x,h,c)

        out = self.decoder(h)

        return torch.sigmoid(out)


# ======================
# CONFIG
# ======================

FRAMES_DIR = KITTI_FRAMES_DIR

EPOCHS = 10
BATCH_SIZE = 4


dataset = KITTIDataset(FRAMES_DIR)

loader = DataLoader(dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ConvLSTM().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss()


# ======================
# TRAINING LOOP
# ======================

for epoch in range(EPOCHS):

    total_loss = 0

    for seq,target in loader:

        seq = seq.to(device)
        target = target.to(device)

        pred = model(seq)

        loss = criterion(pred,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} Loss {total_loss:.4f}")


# ======================
# SAVE MODEL
# ======================

torch.save(
    model.state_dict(),
    "convlstm_kitti.pth"
)

print("Model saved")