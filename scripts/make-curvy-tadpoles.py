import logging
import random
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import typer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = typer.Typer()


def warp_image_and_keypoints(
    image: np.ndarray,
    keypoints: list[tuple[float, float]],
    curviness: float,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Apply a sine-based warp perpendicular to the rostrum-tailtip axis."""
    h, w = image.shape[:2]

    rostrum = np.array([keypoints[0][0] * w, keypoints[0][1] * h])
    tailtip = np.array([keypoints[-1][0] * w, keypoints[-1][1] * h])

    axis = tailtip - rostrum
    length = np.linalg.norm(axis)
    axis_unit = axis / length
    perp_unit = np.array([-axis_unit[1], axis_unit[0]])

    # Build displacement field
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.stack([xs, ys], axis=-1).astype(np.float32)

    # For each pixel, compute how far along the spine axis it is (0-1)
    relative = coords - rostrum
    t = (relative @ axis_unit) / length
    direction = random.choice([-1, 1])
    displacement = direction * curviness * length * np.sin(np.pi * t)

    map_x = (coords[..., 0] - displacement * perp_unit[0]).astype(np.float32)
    map_y = (coords[..., 1] - displacement * perp_unit[1]).astype(np.float32)

    warped = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Warp keypoints by the same displacement
    warped_keypoints = []
    for kx, ky in keypoints:
        px, py = kx * w, ky * h
        rel = np.array([px, py]) - rostrum
        t = np.dot(rel, axis_unit) / length
        disp = curviness * length * np.sin(np.pi * t)
        new_px = px + disp * perp_unit[0]
        new_py = py + disp * perp_unit[1]
        warped_keypoints.append((new_px / w, new_py / h))

    return warped, warped_keypoints


def read_label(path: Path) -> tuple[list, list[tuple[float, float]], list[int]]:
    """Returns (raw_values, keypoints, visibilities) from a YOLO label line."""
    with open(path) as f:
        values = f.readline().strip().split()
    floats = [float(v) for v in values]
    cls, cx, cy, bw, bh, *kp_values = floats
    keypoints = [(kp_values[i], kp_values[i + 1]) for i in range(0, len(kp_values), 3)]
    visibilities = [int(kp_values[i + 2]) for i in range(0, len(kp_values), 3)]
    return [cls, cx, cy, bw, bh], keypoints, visibilities


def write_label(
    path: Path,
    bbox: list,
    keypoints: list[tuple[float, float]],
    visibilities: list[int],
):
    """Write a YOLO pose label file with updated bbox and keypoints."""
    # Recompute bbox from keypoints
    xs = [kp[0] for kp in keypoints]
    ys = [kp[1] for kp in keypoints]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)

    kp_str = " ".join(
        f"{kx:.6f} {ky:.6f} {v}" for (kx, ky), v in zip(keypoints, visibilities)
    )
    with open(path, "w") as f:
        f.write(f"{int(bbox[0])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp_str}\n")


@app.command()
def main(
    image_path: str,
    curviness: float = 0.1,
):
    img_path = Path(image_path)
    label_path = img_path.with_suffix(".txt")

    if not label_path.exists():
        logger.error(f"Label not found: {label_path}")
        raise typer.Exit(1)

    image = cv2.imread(str(img_path))
    bbox, keypoints, visibilities = read_label(label_path)

    warped_image, warped_keypoints = warp_image_and_keypoints(
        image, keypoints, curviness
    )

    uid = uuid4().hex[:8]
    out_img = img_path.parent / f"out/{img_path.stem}_curvy_{uid}{img_path.suffix}"
    out_label = label_path.parent / f"out/{label_path.stem}_curvy_{uid}.txt"

    cv2.imwrite(str(out_img), warped_image)
    write_label(out_label, bbox, warped_keypoints, visibilities)

    logger.info(f"Saved: {out_img}")
    logger.info(f"Saved: {out_label}")


if __name__ == "__main__":
    app()
