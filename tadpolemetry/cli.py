from pathlib import Path

import typer

from .pipeline import MeasurementPipeline

app = typer.Typer()

# TODO: Add canonincal "best" directory so we don't have to prescribe specific models
DEFAULT_SCALE_WEIGHTS = Path(
    "runs/scale_model_output/run_20260315_132415/weights/best.pt"
)
DEFAULT_SPLINE_WEIGHTS = Path(
    "runs/spline_model_output/initial/pose/train/weights/best.pt"
)


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
):
    pipeline = MeasurementPipeline(scale_weights, spline_weights)

    for file_path in input_dir.iterdir():
        if file_path.is_file():
            try:
                pipeline.process(file_path, output_dir)
            except Exception as e:
                typer.echo(f"Failed {file_path.name}: {e}")


if __name__ == "__main__":
    app()
