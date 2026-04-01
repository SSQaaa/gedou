import cv2
import time
from config import CAM_INDEX, CAM_WIDTH, CAM_HEIGHT

class FrameGrabber:
    def __init__(self, index=CAM_INDEX, width=CAM_WIDTH, height=CAM_HEIGHT):
        self.cap = cv2.VideoCapture(index)

        if not self.cap.isOpened():
            raise RuntimeError("Camera open failed")

        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*'MJPG'))
                     
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.cap.set(cv2.CAP_PROP_FPS, 400)


    def get(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def close(self):
        if self.cap.isOpened():
            self.cap.release()
        time.sleep(1)   # 给内核时间释放