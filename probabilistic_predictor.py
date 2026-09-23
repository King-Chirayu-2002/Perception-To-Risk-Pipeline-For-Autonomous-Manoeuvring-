import numpy as np
import random

class ProbabilisticPredictor:
    """
    Generates multiple possible future trajectories using
    Monte-Carlo motion simulation.
    """

    def __init__(self, fps=10, future_steps=10, num_samples=8):
        self.fps = fps
        self.future_steps = future_steps
        self.num_samples = num_samples

    def sample_trajectories(self, cx, cy, vx, vy):

        trajectories = []

        for _ in range(self.num_samples):

            # add gaussian noise to velocity
            vx_noise = vx + random.gauss(0, abs(vx)*0.15 + 1)
            vy_noise = vy + random.gauss(0, abs(vy)*0.15 + 1)

            path = []

            for t in range(1, self.future_steps+1):

                fx = int(cx + vx_noise * t / self.fps)
                fy = int(cy + vy_noise * t / self.fps)

                path.append((fx, fy))

            trajectories.append(path)

        return trajectories