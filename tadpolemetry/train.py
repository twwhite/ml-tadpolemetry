import os
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO

from .logging import get_logger

log = get_logger(__name__)


VALID_MODELS = ["scale", "spline"]


def train(
    model_type: str, config: Path | None = None, epochs: int = 100, batch: int = 32
):
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    if model_type not in VALID_MODELS:
        raise ValueError(f"model_type must be one of {VALID_MODELS}")

    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    config = config or Path(f"data/training/configs/config_{model_type}_model.yml")
    if not config.exists():
        raise FileNotFoundError(f"Config not found: {config}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cwd = os.getcwd()

    model = YOLO("yolo26n.pt") if model_type == "scale" else YOLO("yolo26s-pose.pt")

    model.train(
        data=str(config),
        epochs=epochs,
        batch=batch,
        imgsz=1280,
	degrees=180,
        project=f"{cwd}/runs/{model_type}_model_output",
        name=f"run_{timestamp}",
    )
