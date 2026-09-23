import os

class EgoMotionEstimator:
    """
    Ego-Motion Estimator using KITTI OXTS odometry data.
    Extracts forward velocity (vf) per frame for physics-consistent modeling.
    """

    def __init__(self, oxts_dir, fps=10.0):
        self.oxts_dir = oxts_dir
        self.fps = fps
        self.oxts_files = sorted(os.listdir(oxts_dir))
        self.velocities = self._load_velocities()
        print(f"✅ Ego-Motion Loaded from OXTS: {len(self.velocities)} frames")

    def _load_velocities(self):
        velocities = []
        for file in self.oxts_files:
            path = os.path.join(self.oxts_dir, file)
            with open(path, 'r') as f:
                data = list(map(float, f.readline().strip().split()))
                # KITTI OXTS format:
                # index 8 = vf (forward velocity in m/s)
                vf = data[8]
                velocities.append(vf)
        return velocities

    def get_ego_velocity(self, frame_idx):
        if frame_idx >= len(self.velocities):
            return self.velocities[-1]
        return self.velocities[frame_idx]