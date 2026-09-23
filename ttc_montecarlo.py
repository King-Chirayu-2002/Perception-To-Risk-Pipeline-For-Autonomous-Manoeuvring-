import numpy as np


class TTCMonteCarlo:

    def __init__(self, samples=30):

        self.samples = samples


    def compute_ttc_samples(self, obj_pos, obj_vel, ego_pos):

        ttc_values = []

        for _ in range(self.samples):

            vx = obj_vel[0] + np.random.normal(0, abs(obj_vel[0])*0.2 + 0.5)
            vy = obj_vel[1] + np.random.normal(0, abs(obj_vel[1])*0.2 + 0.5)

            rel_x = obj_pos[0] - ego_pos[0]
            rel_y = obj_pos[1] - ego_pos[1]

            rel_vx = vx
            rel_vy = vy

            dist = np.sqrt(rel_x**2 + rel_y**2)

            rel_speed = np.sqrt(rel_vx**2 + rel_vy**2) + 1e-6

            ttc = dist / rel_speed

            ttc_values.append(ttc)

        return np.array(ttc_values)


    def collision_probability(self, ttc_samples, threshold=3.0):

        collisions = ttc_samples < threshold

        prob = np.sum(collisions) / len(ttc_samples)

        return prob