import numpy as np

class KITTICameraModel:
    """
    Camera model for KITTI (image_02 color camera)
    Used for pixel → 3D back-projection
    """

    def __init__(self):
        # KITTI default intrinsics (for 1242x375 camera)
        # These are standard and acceptable for research prototypes
        self.fx = 721.5377
        self.fy = 721.5377
        self.cx = 609.5593
        self.cy = 172.8540

    def pixel_to_camera(self, u, v, depth):
        """
        Convert pixel coordinates + depth → 3D camera coordinates (meters)
        """
        Z = depth
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy
        return np.array([X, Y, Z], dtype=np.float32)