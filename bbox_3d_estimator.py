import numpy as np
import cv2


class BoundingBox3DEstimator:
    """
    Estimate approximate 3D bounding box from
    2D bounding box and depth map.
    """

    def __init__(self, camera_model):

        self.camera = camera_model

    def estimate_depth(self, depth_map, x1, y1, x2, y2):

        h, w = depth_map.shape

        x1 = max(0, x1)
        x2 = min(w-1, x2)
        y1 = max(0, y1)
        y2 = min(h-1, y2)

        region = depth_map[y1:y2, x1:x2]

        if region.size == 0:
            return 0

        depth = np.median(region)

        return depth

    def compute_3d_box(self, x1, y1, x2, y2, depth):

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        P = self.camera.pixel_to_camera(cx, cy, depth)

        box_w = (x2 - x1)
        box_h = (y2 - y1)

        scale = depth * 0.02

        length = box_w * scale
        height = box_h * scale
        width = length * 0.6

        return P, length, width, height

    def draw_3d_box(self, frame, x1, y1, x2, y2):

        offset = int((x2-x1)*0.25)

        p1 = (x1, y1)
        p2 = (x2, y1)
        p3 = (x2, y2)
        p4 = (x1, y2)

        p5 = (x1-offset, y1-offset)
        p6 = (x2-offset, y1-offset)
        p7 = (x2-offset, y2-offset)
        p8 = (x1-offset, y2-offset)

        cv2.rectangle(frame, p1, p3, (0,255,255), 2)

        cv2.rectangle(frame, p5, p7, (0,255,255), 2)

        cv2.line(frame, p1, p5, (0,255,255), 2)
        cv2.line(frame, p2, p6, (0,255,255), 2)
        cv2.line(frame, p3, p7, (0,255,255), 2)
        cv2.line(frame, p4, p8, (0,255,255), 2)