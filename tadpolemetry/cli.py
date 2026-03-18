import logging
from pathlib import Path

import typer

from tadpolemetry.logging import get_logger

from .pipeline import MeasurementPipeline

log = get_logger(__name__)
app = typer.Typer()

DEFAULT_SCALE_WEIGHTS = Path("runs/scale_model_output/best/weights/best.pt")
DEFAULT_SPLINE_WEIGHTS = Path("runs/spline_model_output/best/weights/best.pt")


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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    if verbose:
        logging.getLogger("tadpolemetry").setLevel(logging.DEBUG)

    pipeline = MeasurementPipeline(scale_weights, spline_weights)

    for file_path in input_dir.iterdir():
        if file_path.is_file():
            try:
                pipeline.process(file_path, output_dir)
            except Exception as e:
                typer.echo(f"Failed {file_path.name}: {e}")


if __name__ == "__main__":
    app()
