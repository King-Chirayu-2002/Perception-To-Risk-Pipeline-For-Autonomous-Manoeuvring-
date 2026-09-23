import os
import cv2
import torch
import imageio
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
import os
from config import PROJECT_ROOT, KITTI_FRAMES_DIR, KITTI_OXTS_DIR, CONVLSTM_MODEL_PATH, CONVLSTM_PREDICTION_DIR, DIFFUSION_FUTURE_DIR, SIMVP_FUTURE_DIR, OBJECT_PATCH_DIR

# =====================================
# CONFIG
# =====================================

FRAMES_DIR = KITTI_FRAMES_DIR

OUTPUT_DIR = DIFFUSION_FUTURE_DIR

NUM_FUTURE_FRAMES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================
# LOAD LAST FRAME
# =====================================

frames = sorted(os.listdir(FRAMES_DIR))

last_frame_path = os.path.join(FRAMES_DIR, frames[-1])

print("Using last frame:", last_frame_path)

frame = cv2.imread(last_frame_path)

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

image = Image.fromarray(frame)

# =====================================
# LOAD DIFFUSION MODEL
# =====================================

print("Loading Stable Video Diffusion model...")

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
)

pipe.to("cuda")

# =====================================
# GENERATE FUTURE VIDEO
# =====================================

print("Generating future frames...")

result = pipe(
    image,
    num_frames=NUM_FUTURE_FRAMES,
    decode_chunk_size=2
)

frames = result.frames[0]

# =====================================
# SAVE GENERATED FRAMES
# =====================================

saved_frames = []

for i, frame in enumerate(frames):

    frame_path = os.path.join(
        OUTPUT_DIR,
        f"future_frame_{i+1}.png"
    )

    frame.save(frame_path)

    saved_frames.append(frame)

    print("Saved:", frame_path)

# =====================================
# SAVE FUTURE VIDEO
# =====================================

video_path = os.path.join(
    OUTPUT_DIR,
    "future_prediction.mp4"
)

writer = imageio.get_writer(video_path, fps=2)

for frame in saved_frames:
    writer.append_data(frame)

writer.close()

print("Saved future video:", video_path)