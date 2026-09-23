import numpy as np
from collections import deque

class TrajectoryPredictor:
    """
    Physics-aware trajectory predictor for tracked objects.
    Uses history of object centers to estimate motion direction
    and predict future positions.
    """

    def __init__(self, history_len=10, future_steps=10, fps=10):
        self.history_len = history_len
        self.future_steps = future_steps
        self.fps = fps
        self.track_histories = {}

    def update(self, track_id, cx, cy):
        """
        Update trajectory history for object
        """
        if track_id not in self.track_histories:
            self.track_histories[track_id] = deque(maxlen=self.history_len)

        self.track_histories[track_id].append((cx, cy))

    def get_velocity(self, track_id):
        """
        Estimate velocity from history
        """
        hist = self.track_histories.get(track_id, [])

        if len(hist) < 2:
            return 0, 0

        (x1, y1), (x2, y2) = hist[-2], hist[-1]

        vx = (x2 - x1) * self.fps
        vy = (y2 - y1) * self.fps

        return vx, vy

    def predict_trajectory(self, track_id):
        """
        Predict future trajectory using constant velocity model
        """
        hist = self.track_histories.get(track_id, [])

        if len(hist) < 2:
            return []

        cx, cy = hist[-1]
        vx, vy = self.get_velocity(track_id)

        future_points = []

        for t in range(1, self.future_steps + 1):
            fx = int(cx + vx * t / self.fps)
            fy = int(cy + vy * t / self.fps)
            future_points.append((fx, fy))

        return future_points

    def classify_direction(self, track_id):
        """
        Determine object motion direction
        """
        hist = self.track_histories.get(track_id, [])

        if len(hist) < 5:
            return "Analyzing"

        dx_total = hist[-1][0] - hist[0][0]
        dy_total = hist[-1][1] - hist[0][1]

        angle = np.degrees(np.arctan2(dy_total, dx_total))

        speed = np.sqrt(dx_total**2 + dy_total**2)

        if speed < 10:
            return "Stopping"

        if -20 <= angle <= 20:
            return "Straight"

        if angle > 20:
            return "Right"

        if angle < -20:
            return "Left"

        return "Unknown"

    def trajectory_curvature(self, track_id):
        """
        Estimate curvature of trajectory
        """
        hist = self.track_histories.get(track_id, [])

        if len(hist) < 5:
            return 0

        pts = np.array(hist)

        dx = np.gradient(pts[:,0])
        dy = np.gradient(pts[:,1])

        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        curvature = np.mean(np.abs(dx * ddy - dy * ddx) /
                            (dx**2 + dy**2 + 1e-6)**1.5)

        return curvature