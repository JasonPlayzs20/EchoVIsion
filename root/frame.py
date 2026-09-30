import json

import cv2
from ultralytics import YOLO


class Frame:
    def __init__(self,cam1, cam2, timing):
        self.model = YOLO("yolo26n.pt")
        self.cam1 = cam1
        self.cam2 = cam2
        self.timing = timing
        with open("calibration.json", 'r') as f:
            data = json.load(f)
            self.mtx = data['mtx']
            self.dist = data['dist']

    def un_distortion(self):
        imgUP = self.cam1
        imgDOWN = self.cam2
        h1, w1 = imgUP.shape[:2]
        h2, w2 = imgDOWN.shape[:2]
        newcameramtxUP, roiUP = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w1, h1), 1, (w1, h1))
        newcameramtxDOWN, roiDOWN = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w2, h1), 1, (w2, h2))

        # undistort
        dstUP = cv2.undistort(self.img, self.mtx, self.dist, None, newcameramtxUP)

        # crop the image
        x, y, w, h = roiUP
        dstUP = dstUP[y:y + h, x:x + w]
        cv2.imwrite('calibresult1.png', dstUP)

        # undistort
        dstDOWN = cv2.undistort(self.img, self.mtx, self.dist, None, newcameramtxDOWN)

        # crop the image
        x, y, w, h = roiDOWN
        dstDOWN = dstDOWN[y:y + h, x:x + w]
        cv2.imwrite('calibresult1.png', dstDOWN)

    def yolo(self, searching_object):
        results = self.model(self.cam1[0])
        result = results[0]

        annotated_frame = result.plot()

        print(result.verbose())