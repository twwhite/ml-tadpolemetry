from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from tadpolemetry.pipeline import MeasurementPipeline, MeasurementResult

SCALE_WEIGHTS = Path("runs/scale_model_output/best/weights/best.pt")
SPLINE_WEIGHTS = Path("runs/spline_model_output/best/weights/best.pt")


TEST_IMAGE = Path("tests/test1.jpg")
EXPECTED_LENGTH_MM = 10.9
TOLERANCE = 0.05  # 5%


@pytest.fixture(scope="module")
def pipeline():
    return MeasurementPipeline(SCALE_WEIGHTS, SPLINE_WEIGHTS)


# --- MeasurementResult ---
def test_basic_spline(pipeline, tmp_path):
    result = pipeline.process(TEST_IMAGE, tmp_path, skip_scale=True)

    assert result.success, f"Pipeline failed: {result.failure_reason}"


# --- MeasurementResult ---
def test_known_length(pipeline, tmp_path):
    result = pipeline.process(TEST_IMAGE, tmp_path)

    assert result.success, f"Pipeline failed: {result.failure_reason}"
    assert result.length_mm == pytest.approx(EXPECTED_LENGTH_MM, rel=TOLERANCE)
