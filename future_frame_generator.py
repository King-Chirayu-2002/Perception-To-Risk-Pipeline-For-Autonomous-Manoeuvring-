import cv2
import numpy as np


class FutureFrameGenerator:

    def __init__(self, fps=10):
        self.fps = fps


    def extract_object(self, frame, bbox):

        x1, y1, x2, y2 = bbox
        patch = frame[y1:y2, x1:x2].copy()

        if patch.size == 0:
            return None, None

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        return patch, mask


    def paste_object(self, frame, patch, mask, center):

        h, w = patch.shape[:2]
        cx, cy = center

        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = x1 + w
        y2 = y1 + h

        if x1 < 0 or y1 < 0:
            return frame
        if x2 > frame.shape[1] or y2 > frame.shape[0]:
            return frame

        roi = frame[y1:y2, x1:x2]

        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        fg = cv2.bitwise_and(patch, patch, mask=mask)

        combined = cv2.add(bg, fg)

        frame[y1:y2, x1:x2] = combined

        return frame


    def shift_background(self, frame, ego_velocity):

        dy = int(ego_velocity * 2)

        M = np.float32([[1, 0, 0],
                        [0, 1, dy]])

        shifted = cv2.warpAffine(frame, M,
                                 (frame.shape[1], frame.shape[0]))

        return shifted


    # =============================
    # MULTI HYPOTHESIS FUTURE
    # =============================

    def generate_multi_future(self, frame, objects,
                              ego_velocity=0,
                              steps=8):

        futures = {
            "straight": [],
            "left": [],
            "right": []
        }

        base = frame.copy()

        for t in range(1, steps + 1):

            straight = self.shift_background(base.copy(),
                                             ego_velocity * t / self.fps)

            left = straight.copy()
            right = straight.copy()

            for obj in objects:

                patch = obj["patch"]
                mask = obj["mask"]

                cx, cy = obj["center"]
                vx, vy = obj["velocity"]

                # straight
                sx = int(cx + vx * t)
                sy = int(cy + vy * t)

                # left hypothesis
                lx = int(cx + vx * t - 0.5 * t)
                ly = int(cy + vy * t)

                # right hypothesis
                rx = int(cx + vx * t + 0.5 * t)
                ry = int(cy + vy * t)

                straight = self.paste_object(straight,
                                             patch,
                                             mask,
                                             (sx, sy))

                left = self.paste_object(left,
                                         patch,
                                         mask,
                                         (lx, ly))

                right = self.paste_object(right,
                                          patch,
                                          mask,
                                          (rx, ry))

            futures["straight"].append(straight)
            futures["left"].append(left)
            futures["right"].append(right)

        return futures