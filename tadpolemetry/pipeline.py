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
    mean_ruler_delta_px: float | None
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
    a_side_tick_coords: list[tuple]
    b_side_tick_coords: list[tuple]
    side_used: str


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
    SPLINE_MODEL_CONF = 0.5

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

        a_ruler_ticks_centers = []
        b_ruler_ticks_centers = []

        if scale_result.boxes:
            ruler_ticks_xywh = [
                [float(j) for j in n.xywh[0]] for n in scale_result.boxes
            ]
            for tick in ruler_ticks_xywh:
                if tick[2] > tick[3]:
                    b_ruler_ticks_centers.append((tick[0], tick[1]))
                else:
                    a_ruler_ticks_centers.append((tick[0], tick[1]))

        else:
            raise ScaleNotDetectedError(img_path, "No scale keypoints detected.")

        mean_ruler_delta_px = 150 if skip_scale else 0

        if len(a_ruler_ticks_centers) > len(b_ruler_ticks_centers):
            mean_ruler_delta_px = self._mean_interval_from_group(a_ruler_ticks_centers)
            side_used = "TOP"
        else:
            mean_ruler_delta_px = self._mean_interval_from_group(b_ruler_ticks_centers)
            side_used = "RIGHT"

        return ScaleResult(
            mean_ruler_delta_px=mean_ruler_delta_px,
            a_side_tick_coords=a_ruler_ticks_centers,
            b_side_tick_coords=b_ruler_ticks_centers,
            side_used=side_used,
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

    def _mean_interval_from_group(self, centers: list[tuple]) -> float:

        log.debug(f"num ruler ticks: {len(centers)}")
        """Return mean px interval between adjacent ticks in a group."""
        pts = np.array(centers)

        x_spread = pts[:, 0].max() - pts[:, 0].min()
        y_spread = pts[:, 1].max() - pts[:, 1].min()
        dominant_axis = (
            0 if x_spread > y_spread else 1
        )  # 0=x (horizontal), 1=y (vertical)

        sorted_pts = pts[pts[:, dominant_axis].argsort()]
        adjacent_intervals = np.diff(sorted_pts[:, dominant_axis])

        log.debug(f"adjacent {adjacent_intervals}")

        median_interval = np.median(adjacent_intervals)

        THRESHOLD_FILTER_PCT = 0.5

        filtered_intervals = adjacent_intervals[
            (adjacent_intervals >= (1 - THRESHOLD_FILTER_PCT) * median_interval)
            & (adjacent_intervals <= (1 + THRESHOLD_FILTER_PCT) * median_interval)
        ]

        log.debug(f"filtered {filtered_intervals}")

        if len(filtered_intervals) == 0:
            log.error("No intervals after filtering. Odd!")
            raise TadpolemetryError()

        n_filtered = len(adjacent_intervals) - len(filtered_intervals)
        if n_filtered > 0:
            log.debug(f"Filtered {n_filtered} outlier tick intervals")

        return round(float(filtered_intervals.mean()), 1)

    def process(
        self, file: Path, output_dir: Path, skip_scale: bool = False
    ) -> MeasurementResult:

        img_path = str(file)

        if not Path(img_path).exists():
            return MeasurementResult(file.name, None, None, "Image file not found")

        img = cv2.imread(img_path)

        if img is None:
            return MeasurementResult(file.name, None, None, "Failed to load image")

        log.debug(f"process start for {img_path}")

        ruler_data = self._run_scale_model(img_path, skip_scale=skip_scale)
        spline_data = self._run_spline_model(img_path)

        log.debug(f"ruler delta: {ruler_data.mean_ruler_delta_px}")

        length_mm = sum(spline_data.segment_lengths) / ruler_data.mean_ruler_delta_px

        # --- Annotate image ---
        for x, y in spline_data.labeled_kp.values():
            cv2.circle(img, (int(x), int(y)), 24, (0, 0, 255), -1)
        for a, b in self.TADPOLE_CONNECTIONS:
            x1, y1 = spline_data.labeled_kp[a]
            x2, y2 = spline_data.labeled_kp[b]
            cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 12)

        for x, y in ruler_data.a_side_tick_coords:
            cv2.circle(img, (int(x), int(y)), 24, (255, 0, 255), -1)

        for x, y in ruler_data.b_side_tick_coords:
            cv2.circle(img, (int(x), int(y)), 24, (0, 255, 255), -1)

        diag_text = [
            f"Tadpole Length {round(length_mm, 2)} mm",
            f"Ruler spacing {ruler_data.mean_ruler_delta_px} px",
            f"Ruler axis: {ruler_data.side_used} side",
        ]

        line_start = 250
        for i, text in enumerate(diag_text):
            cv2.putText(
                img,
                text,
                (50, line_start + (i * 80)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                8,
                cv2.LINE_AA,
            )

        # --- Save annotated image (OPTIONAL) ---
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / file.name), img)

        # --- Show annotage dimage (OPTIONAL) ---
        # cv2.namedWindow("tadpole", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("tadpole", 800, 600)
        # cv2.imshow("tadpole", img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # input("Press enter to continue...")

        return MeasurementResult(
            file.name, length_mm, ruler_data.mean_ruler_delta_px, None
        )
