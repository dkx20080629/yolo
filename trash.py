from ultralytics import YOLO
model = YOLO(r"C:\Users\User\Desktop\code\best.pt")
model.export(format="ncnn")
