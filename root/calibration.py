'''
Code partially sourced from https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html
Modified by Jason - live video version.
'''
import time

import numpy as np
import cv2 as cv
import glob
import json

cam = cv.VideoCapture(0)
start = time.perf_counter()

while time.perf_counter()-start < 5:
    ret, frame = cam.read()
    cv.imshow('frame', frame)

#new
ret, frame = cam.read()
cv.imshow('frame', frame)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

cam.release()

# images = glob.glob('*.jpg')
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv.findChessboardCorners(gray, (7,6), None)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

    corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    imgpoints.append(corners2)

    # Draw and display the corners
    cv.drawChessboardCorners(frame, (7,6), corners2, ret)
    cv.imshow('img', frame)
    cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


data = {
    "ret":ret,
    "mtx":mtx,
    "dist":dist,
    "rvecs":rvecs,
    "tvecs":tvecs
}

with open('calibration.json', 'w') as f:
    json.dump(data, f, indent=4)
