import csv
import logging
import random
from datetime import datetime
from pathlib import Path

import typer

from tadpolemetry.logging import get_logger

from .pipeline import MeasurementPipeline

log = get_logger(__name__)
app = typer.Typer()

DEFAULT_SCALE_WEIGHTS = Path("runs/scale_model_output/best/weights/best.pt")
DEFAULT_SPLINE_WEIGHTS = Path("runs/spline_model_output/best/weights/best.pt")
DEFAULT_RANDOM_SAMPLE_PCT = 5
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}
DEFAULT_BINS = 10


@app.command()
def train(
    model_type: str = typer.Argument(..., help="Model to train: scale or spline"),
    config: Path = typer.Option(None, help="Path to training config YAML"),
    epochs: int = typer.Option(100),
    batch: int = typer.Option(32),
):
    from .train import train as run_train

    run_train(model_type, config, epochs, batch)

@app.command()
def copy_review(
    csv_path: Path = typer.Argument(..., help="Path to results CSV"),
    input_dir: Path = typer.Argument(..., help="Directory of original input images"),
    output_dir: Path = typer.Argument(..., help="Output directory — review/ will be created inside"),
):
    from .analyze import copy_review_images
    copy_review_images(csv_path, input_dir, output_dir)

@app.command()
def analyze(
    csv_path: Path = typer.Argument(..., help="Path to results CSV"),
    output_dir: Path = typer.Argument(..., help="Directory to save histogram"),
    bins: int = typer.Option(DEFAULT_BINS, help="Number of histogram bins"),
):
    from .analyze import plot_length_histogram, flag_outliers
    flag_outliers(csv_path)
    plot_length_histogram(csv_path, output_dir, bins)

@app.command()
def measure(
    input_dir: Path = typer.Argument(..., help="Directory of images to process"),
    output_dir: Path = typer.Argument(..., help="Directory to write results"),
    scale_weights: Path = typer.Option(
        DEFAULT_SCALE_WEIGHTS, help="Scale model weights"
    ),
    spline_weights: Path = typer.Option(
        DEFAULT_SPLINE_WEIGHTS, help="Spline model weights"
    ),
    random_sample_pct: int = typer.Option(
        DEFAULT_RANDOM_SAMPLE_PCT,
        help="Percentage of images to flag for manual validation",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose:
        logging.getLogger("tadpolemetry").setLevel(logging.DEBUG)

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = MeasurementPipeline(scale_weights, spline_weights)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"results_{timestamp}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "length_mm",
                "scale_mean_interval_px",
                "random_validate",
                "failure_reason",
            ],
        )
        writer.writeheader()

        count_flagged_for_review = 0
        for file_path in input_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                try:
                    result = pipeline.process(file_path, output_dir)
                except Exception as e:
                    typer.echo(f"Failed {file_path.name}: {e}")
                    continue

                flag_for_review = random.randrange(100) < random_sample_pct
                count_flagged_for_review += 1 if flag_for_review else 0
                writer.writerow(
                    {
                        "filename": result.filename,
                        "length_mm": result.length_mm,
                        "scale_mean_interval_px": result.mean_ruler_delta_px,
                        "failure_reason": result.failure_reason,
                        "random_validate": flag_for_review,
                    }
                )

                f.flush()

        log.debug(f"COMPLETE. Flagged {count_flagged_for_review} random samples for review")
if __name__ == "__main__":
    app()
