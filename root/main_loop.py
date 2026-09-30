import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cv2.namedWindow("test")
cv2.namedWindow("test2")

up_cam = cv2.VideoCapture(1)
down_cam = cv2.VideoCapture(0) #ma iphone :D
while True:
    retu, imp_up = up_cam.read()
    retd, imp_down = down_cam.read()

    cv2.imshow("test", imp_up)
    cv2.imshow("test2", imp_down)
    key = cv2.waitKey(10)
    # results = model.track(imp_up)
    results = model(imp_up)
    results2 = model(imp_down)

    # cv2.imshow("test", imp_up)
    # cv2.imshow("test", imp_up)
    result = results[0]
    result2 = results2[0]

    annotated_frame = result.plot()
    cv2.imshow("test2", annotated_frame)
    annotated_frame2 = result2.plot()
    cv2.imshow("test3", annotated_frame2)
    print("AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
    print(result.verbose())

    # cv2.imshow("test", result)


    if key == 27:
        break

cv2.destroyAllWindows()
up_cam.release()
# down_cam.release()