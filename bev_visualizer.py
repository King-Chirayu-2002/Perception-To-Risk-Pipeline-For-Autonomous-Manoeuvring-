import numpy as np
import cv2


class BEVVisualizer:

    def __init__(self, width=500, height=500):

        self.w = width
        self.h = height


    def create_map(self):

        bev = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        cv2.line(bev,(self.w//2,0),(self.w//2,self.h),(100,100,100),2)

        return bev


    def draw_object(self, bev, x, y, color=(0,255,0)):

        bx = int(self.w/2 + x*10)
        by = int(self.h - y*10)

        if 0 <= bx < self.w and 0 <= by < self.h:

            cv2.circle(bev,(bx,by),5,color,-1)


    def draw_trajectory(self, bev, traj):

        for p in traj:

            bx = int(self.w/2 + p[0]*10)
            by = int(self.h - p[1]*10)

            if 0<=bx<self.w and 0<=by<self.h:

                cv2.circle(bev,(bx,by),2,(0,0,255),-1)

        return bev