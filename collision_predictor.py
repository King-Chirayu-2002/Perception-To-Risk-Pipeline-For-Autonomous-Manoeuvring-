import numpy as np


class CollisionPredictor:
    """
    Predicts potential collision between ego vehicle and objects
    based on predicted trajectories.
    """

    def __init__(self, ego_width=80, ego_height=120):
        self.ego_width = ego_width
        self.ego_height = ego_height

    def ego_future_path(self, ego_x, ego_y, steps=10, step_size=20):
        """
        Predict ego vehicle forward motion in image space
        """
        path = []

        for i in range(1, steps + 1):
            fy = int(ego_y - i * step_size)
            fx = ego_x
            path.append((fx, fy))

        return path

    def check_collision(self, ego_path, object_path, threshold=40):
        """
        Checks if object path intersects ego path
        """
        for e in ego_path:
            for o in object_path:

                dist = np.sqrt((e[0] - o[0])**2 + (e[1] - o[1])**2)

                if dist < threshold:
                    return True

        return False