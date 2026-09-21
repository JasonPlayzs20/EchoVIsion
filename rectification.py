import cv2
import numpy as np

# Placeholders from stereo calibration 
C1 = np.array([
    [600 , 0, 640], #leftmost val is focal length 
    [0, 600, 360],
    [0, 0, 1]
], dtype=np.float64)

# Distortion coefficients for an ideal camera. change later to actual values from calibration
D1 = np.zeros(5, dtype=np.float64)

C2 = np.array([
    [600, 0, 640],
    [0, 600, 360],
    [0, 0, 1]
], dtype=np.float64)

D2 = np.zeros(5, dtype=np.float64)

# Placeholder rotation and translation between the two cameras. change later to actual values from calibration
R = np.eye(3, dtype=np.float64)

T = np.array([
    [-0.20],
    [0.0],
    [0.0]
], dtype=np.float64)

image_size = (1280, 720)

# Compute rectification transforms
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    C1,
    D1,
    C2,
    D2,
    image_size,
    R,
    T
)

# Precompute pixel-remapping tables
left_map_x, left_map_y = cv2.initUndistortRectifyMap(
    C1,
    D1,
    R1,
    P1,
    image_size,
    cv2.CV_32FC1
)

right_map_x, right_map_y = cv2.initUndistortRectifyMap(
    C2,
    D2,
    R2,
    P2,
    image_size,
    cv2.CV_32FC1
)