import torch
from ultralytics import YOLO

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = YOLO("yolo26n-pose.pt")

model.train(data='configs/config_scale_model.yaml', epochs=100, batch=32, imgsz=640, augment=True)
# model.train(data='configs/config_spline_model.yaml', epochs=100, batch=32, imgsz=640, augment=True)
