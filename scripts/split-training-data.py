import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Annotated

import typer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = typer.Typer()


@app.command()
def main(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
):
    root = Path(input_dir)
    images_dir = root / "images"
    labels_dir = root / "labels"

    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        logger.error(f"No .jpg files found in {images_dir}")
        raise typer.Exit(1)

    random.shuffle(image_files)

    n_train = int(len(image_files) * train_ratio)
    splits = {"train": image_files[:n_train], "val": image_files[n_train:]}

    for split, files in splits.items():
        out_images = Path(output_dir) / split / "images"
        out_labels = Path(output_dir) / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        for img_path in files:
            label_path = labels_dir / img_path.with_suffix(".txt").name
            shutil.move(img_path, out_images / img_path.name)
            if label_path.exists():
                shutil.move(label_path, out_labels / label_path.name)
            else:
                logger.warning(f"No label found for {img_path.name}")

    logger.info(f"Done: {len(splits['train'])} train, {len(splits['val'])} val")


if __name__ == "__main__":
    app()
