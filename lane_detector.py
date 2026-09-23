import cv2
import numpy as np


class LaneDetector:

    def detect_lanes(self, frame):

        h, w = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5,5), 0)

        edges = cv2.Canny(blur, 50, 150)

        mask = np.zeros_like(edges)

        polygon = np.array([[
            (0, h),
            (w, h),
            (int(w*0.6), int(h*0.55)),
            (int(w*0.4), int(h*0.55))
        ]])

        cv2.fillPoly(mask, polygon, 255)

        roi = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi,
            1,
            np.pi/180,
            threshold=40,
            minLineLength=40,
            maxLineGap=50
        )

        return lines


    def draw_lanes(self, frame, lines):

        if lines is None:
            return frame

        for line in lines:
            x1,y1,x2,y2 = line[0]

            cv2.line(
                frame,
                (x1,y1),
                (x2,y2),
                (255,0,0),
                3
            )

        return frame