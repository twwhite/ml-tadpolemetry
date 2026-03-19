import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from IPython import embed
from ultralytics import YOLO

from .logging import get_logger

log = get_logger(__name__)


@dataclass
class MeasurementResult:
    filename: str
    length_mm: float | None
    failure_reason: str | None

    @property
    def success(self) -> bool:
        return self.length_mm is not None


@dataclass
class SplineResult:
    labeled_kp: dict
    segment_lengths: list[float]


@dataclass
class ScaleResult:
    mean_ruler_delta_px: float
    tick_coords: list[tuple]


class TadpolemetryError(Exception):
    pass


class ScaleNotDetectedError(TadpolemetryError):
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"{filename}: {reason}")


class SplineNotDetectedError(TadpolemetryError):
    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        super().__init__(f"{filename}: {reason}")


class MeasurementPipeline:
    TADPOLE_KEYPOINTS = [
        "pos_rostrum",
        "pos_tailtip",
        "pos_tailbase",
        "pos_tailbase_third",
        "pos_tailtip_third",
    ]
    TADPOLE_CONNECTIONS = [
        ("pos_rostrum", "pos_tailbase"),
        ("pos_tailbase", "pos_tailbase_third"),
        ("pos_tailbase_third", "pos_tailtip_third"),
        ("pos_tailtip_third", "pos_tailtip"),
    ]
    SCALE_MODEL_CONF = 0.25
    SPLINE_MODEL_CONF = 0.25

    def __init__(self, scale_weights: Path, spline_weights: Path):
        if not scale_weights.exists():
            raise FileNotFoundError(f"Scale weights not found: {scale_weights}")

        if not spline_weights.exists():
            raise FileNotFoundError(f"Spline weights not found: {spline_weights}")

        self.scale_model = YOLO(scale_weights)
        self.spline_model = YOLO(spline_weights)

    def _run_scale_model(self, img_path: str, skip_scale: bool = False) -> ScaleResult:
        """Execute scale model, return ruler measuremenet delta"""
        scale_result = self.scale_model(img_path, conf=self.SCALE_MODEL_CONF)[0]

        if scale_result.boxes:
            ruler_ticks_xywh = [
                [float(j) for j in n.xywh[0]] for n in scale_result.boxes
            ]
            ruler_ticks_centers = [(n[0], n[1]) for n in ruler_ticks_xywh]
        else:
            raise ScaleNotDetectedError(img_path, "No scale keypoints detected.")

        # ruler_deltas = [
        #     np.linalg.norm(ruler_kp[i] - ruler_kp[i + 1])
        #     for i in range(len(ruler_kp) - 1)
        # ]

        mean_ruler_delta_px = 150 if skip_scale else 150
        # mean_ruler_delta_px = float(sum(ruler_deltas) / len(ruler_deltas))

        return ScaleResult(
            mean_ruler_delta_px=mean_ruler_delta_px, tick_coords=ruler_ticks_centers
        )

    def _run_spline_model(self, img_path: str) -> SplineResult:
        """Execute spline model, return body keypoints"""
        tadpole_result = self.spline_model(img_path, conf=self.SPLINE_MODEL_CONF)[0]
        tadpole_kp = tadpole_result.keypoints.xy[0].cpu().numpy()

        if len(tadpole_kp) < 5:
            raise SplineNotDetectedError(
                img_path, "Incomplete spline keypoints detected."
            )

        labeled_kp = dict(zip(self.TADPOLE_KEYPOINTS, tadpole_kp))

        segment_lengths = [
            float(np.linalg.norm(labeled_kp[a] - labeled_kp[b]))
            for a, b in self.TADPOLE_CONNECTIONS
        ]
        return SplineResult(labeled_kp=labeled_kp, segment_lengths=segment_lengths)

    def process(
        self, file: Path, output_dir: Path, skip_scale: bool = False
    ) -> MeasurementResult:

        img_path = str(file)

        if not Path(img_path).exists():
            return MeasurementResult(file.name, None, "Image file not found")

        img = cv2.imread(img_path)

        if img is None:
            return MeasurementResult(file.name, None, "Failed to load image")

        log.debug(f"process start for {img_path}")

        ruler_data = self._run_scale_model(img_path, skip_scale=skip_scale)
        spline_data = self._run_spline_model(img_path)

        length_mm = sum(spline_data.segment_lengths) / ruler_data.mean_ruler_delta_px

        # --- Annotate image ---
        for x, y in spline_data.labeled_kp.values():
            cv2.circle(img, (int(x), int(y)), 24, (0, 0, 255), -1)
        for a, b in self.TADPOLE_CONNECTIONS:
            x1, y1 = spline_data.labeled_kp[a]
            x2, y2 = spline_data.labeled_kp[b]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 12)

        for x, y in ruler_data.tick_coords:
            cv2.circle(img, (int(x), int(y)), 24, (0, 0, 255), -1)

        text = f"Tadpole Length {round(length_mm, 2)} mm"
        cv2.putText(
            img,
            text,
            (50, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            8,
            cv2.LINE_AA,
        )

        # --- Save annotated image ---
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / file.name), img)

        return MeasurementResult(file.name, length_mm, None)
