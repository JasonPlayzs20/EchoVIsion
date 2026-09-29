import time
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model("test_image.jpg")

n = 20
start = time.perf_counter()
for _ in range(n):
    model("test_image.jpg", verbose=False)
elapsed = time.perf_counter() - start

print(f"Average per image: {elapsed / n * 1000:.1f} ms")
print(f"Throughput: {n / elapsed:.1f} FPS")