from ultralytics import YOLO
import cv2


model = YOLO("yolo26n.pt")
cv2.namedWindow("test")

up_cam = cv2.VideoCapture(0)
down_cam = cv2.VideoCapture(1) #ma iphone :D
while True:
    retu, imp_up = up_cam.read()
    retd, imp_down = down_cam.read()

    cv2.imshow("test", imp_up)
    cv2.imshow("test2", imp_down)
    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()
up_cam.release()
down_cam.release()