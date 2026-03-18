import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
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


class MeasurementPipeline:
    TADPOLE_KEYPOINTS = [
        "pos_rostrum",
        "pos_tailtip",
        "pos_tailbase",
        "pos_tailbase_third",
        "pos_tailtip_third",
    ]
    TADPOLE_CONNECTIONS = [(0, 2), (2, 3), (3, 4), (4, 1)]
    SCALE_CONNECTIONS = list(zip(range(4), range(1, 5)))
    SCALE_MODEL_CONF = 0.25
    SPLINE_MODEL_CONF = 0.25

    def __init__(self, scale_weights: Path, spline_weights: Path):
        if not scale_weights.exists():
            raise FileNotFoundError(f"Scale weights not found: {scale_weights}")

        if not spline_weights.exists():
            raise FileNotFoundError(f"Spline weights not found: {spline_weights}")

        self.scale_model = YOLO(scale_weights)
        self.spline_model = YOLO(spline_weights)

    def process(
        self,
        file: Path,
        output_dir: Path,
        skip_scale: bool = False,
        skip_spline: bool = False,
    ) -> MeasurementResult:
        img_path = str(file)

        if not Path(img_path).exists():
            return MeasurementResult(file.name, None, "Image file not found")

        log.debug(f"process start for {img_path}")

        # --- Scale model ---
        if skip_scale:
            mean_ruler_delta = 150  # Start value
        else:
            scale_result = self.scale_model(img_path, conf=self.SCALE_MODEL_CONF)[0]

            if scale_result.keypoints:
                ruler_kp = scale_result.keypoints.xy[0].cpu().numpy()
            else:
                log.warning(f"No scale keypoints detected for {img_path}. Canceling.")
                return MeasurementResult(
                    file.name, None, "Scale keypoints not detected"
                )

            if len(ruler_kp) < 2:
                log.warning(f"No scale bar detected {img_path}")
                return MeasurementResult(file.name, None, "Scale bar not detected")

            ruler_deltas = [
                np.linalg.norm(ruler_kp[i] - ruler_kp[i + 1])
                for i in range(len(ruler_kp) - 1)
            ]

            mean_ruler_delta = sum(ruler_deltas) / len(ruler_deltas)
        log.debug(f"Calculated mean ruler delta of {mean_ruler_delta}px per 1mm")

        # --- Tadpole model ---
        tadpole_result = self.spline_model(img_path, conf=self.SPLINE_MODEL_CONF)[0]
        img = tadpole_result.plot()
        tadpole_kp = tadpole_result.keypoints.xy[0].cpu().numpy()

        if len(tadpole_kp) < 5:
            log.warning(f"Too few tadpole keypoints detected {img_path}")
            return MeasurementResult(
                file.name, None, "Too few tadpole keypoints detected"
            )

        labeled_kp = dict(zip(self.TADPOLE_KEYPOINTS, tadpole_kp))

        segment_lengths = [
            np.linalg.norm(tadpole_kp[a] - tadpole_kp[b])
            for a, b in self.TADPOLE_CONNECTIONS
        ]
        # --- Conversion ---

        length_mm = sum(segment_lengths) / mean_ruler_delta

        # --- Annotate image ---
        for x, y in tadpole_kp:
            cv2.circle(img, (int(x), int(y)), 24, (0, 0, 255), -1)
        for a, b in self.TADPOLE_CONNECTIONS:
            x1, y1 = tadpole_kp[a]
            x2, y2 = tadpole_kp[b]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 12)

        if not skip_scale:
            for x, y in ruler_kp:
                cv2.circle(img, (int(x), int(y)), 24, (0, 0, 255), -1)
            for a, b in self.SCALE_CONNECTIONS:
                if a < len(ruler_kp) and b < len(ruler_kp):
                    x1, y1 = ruler_kp[a]
                    x2, y2 = ruler_kp[b]
                    cv2.line(
                        img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 12
                    )

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
