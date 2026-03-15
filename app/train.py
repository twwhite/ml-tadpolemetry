import os
import torch
from ultralytics import YOLO
from datetime import datetime

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = YOLO("yolo26n.pt")
cwd = os.getcwd()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_name = "scale"
model.train(
    data=f'app/training/configs/config_{model_name}_model.yml',
    epochs=100,
    batch=32,
    imgsz=640,
    augment=True,
    project=f"{cwd}/app/models/{model_name}_model_output",
    name=f"run_{timestamp}"
)

# model.train(data='configs/config_spline_model.yaml', epochs=100, batch=32, imgsz=640, augment=True)
