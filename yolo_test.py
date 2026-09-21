from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("test_image.jpg")
results[0].show()