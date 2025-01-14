from ultralytics import YOLO
model = YOLO(r"C:\Users\User\Desktop\yolo\best (6).pt")
model.export(format="ncnn")
