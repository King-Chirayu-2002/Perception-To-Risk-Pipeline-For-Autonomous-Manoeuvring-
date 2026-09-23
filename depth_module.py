import cv2
import torch
import numpy as np

class DepthEstimator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use SMALL model for real-time performance (GPU optimized)
        self.model = torch.hub.load(
            "intel-isl/MiDaS",
            "MiDaS_small",
            trust_repo=True
        )

        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load(
            "intel-isl/MiDaS",
            "transforms",
            trust_repo=True
        )
        self.transform = transforms.small_transform

        print(f"✅ MiDaS Depth Model Loaded on: {self.device}")

    def predict_depth(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)
        return depth