import cv2

stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=5
)

def compute_disparity(left_rect_img, right_rect_img):
    left_gray = cv2.cvtColor(left_rect_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_rect_img, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(left_gray, right_gray)
    disparity = disparity.astype("float32") / 16.0

    return disparity

def compute_depth(disparity, focal_length, baseline):
    depth = ( focal_length * baseline ) / disparity
    return depth
