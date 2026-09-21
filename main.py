from ultralytics import YOLO
import cv2
import time
from rectification import left_map_x, left_map_y, right_map_x, right_map_y
from depth import compute_depth, compute_disparity

#initialization
model = YOLO("yolo26n.pt")

left_cam = cv2.VideoCapture(0)
right_cam = cv2.VideoCapture(1)

focal_length = 600 #estimate - real value found during calibration
baseline = 0.20 #meters

#checks if the two frames are synced. 
def isSameFrame(frame1, frame2):
    dt = abs(frame1 - frame2)
    if dt < 0.005:
        return True
    return False

while True:
    #grab() - check if image was successfully loaded
    #this is a rough check for timing, but not perfect. Improve later

    left_ok = left_cam.grab()
    tL = time.perf_counter()

    right_ok = right_cam.grab()
    tR = time.perf_counter()    

    if not left_ok or not right_ok:
        print("Failed to grab frames")
        break

    left_ok, IL = left_cam.retrieve()
    right_ok, IR = right_cam.retrieve()

    if not left_ok or not right_ok:
        print("Failed to retrieve frames")
        break

    if isSameFrame(tL, tR):
        IL_rect = cv2.remap(IL, left_map_x, left_map_y, cv2.INTER_LINEAR)
        IR_rect = cv2.remap(IR, right_map_x, right_map_y, cv2.INTER_LINEAR)

        cv2.imshow("Left", IL_rect)
        cv2.imshow("Right", IR_rect)

        results = model(IL_rect)
        result = results[0] #the matrix containing all the image pixels

        disparity = compute_disparity(IL_rect, IR_rect)
        depth = compute_depth(disparity, focal_length, baseline)

        annotated_frame = result.plot()
        cv2.imshow("Annotated Left", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

left_cam.release()
right_cam.release()
cv2.destroyAllWindows()


