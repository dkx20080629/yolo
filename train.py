from ultralytics import YOLO
from torch import cuda

model = YOLO(r"yolo11n.pt")
model.to("cuda")
if __name__ == '__main__':
    model.train(data=fr"D:\Desktop\yolo\trash8000\data.yaml", epochs=2000, imgsz=640, cache=False, device=0)
