import cv2
def get_avaliable_cameras():
    available_cameras = []

    for i in range(6):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

cameras = get_avaliable_cameras()
if len(cameras) != 0:
    print("Found {} cameras".format(len(cameras)))
    for camera in cameras:
        print("Camera {} detected".format(camera))
else:
    print("nun :(")