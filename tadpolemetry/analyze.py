import csv
from pathlib import Path

import matplotlib.pyplot as plt

from .logging import get_logger

log = get_logger(__name__)

DEFAULT_BINS = 10


def plot_length_histogram(
    csv_path: Path,
    output_dir: Path,
    bins: int = DEFAULT_BINS,
):
    """Read a results CSV and plot a histogram of tadpole lengths."""
    lengths = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["length_mm"]:
                try:
                    log.debug(f"LENGTH: {row['length_mm']}")
                    lengths.append(float(row["length_mm"]))
                except ValueError:
                    log.debug(f"Skipping non-numeric length value: {row['length_mm']}")

    if not lengths:
        log.warning("No valid length measurements found in CSV")
        return

    log.info(f"Plotting histogram for {len(lengths)} measurements")

    fig, ax = plt.subplots()
    ax.hist(lengths, bins=bins, edgecolor="black")
    ax.set_xlabel("Length (mm)")
    ax.set_ylabel("Count")
    ax.set_title("Tadpole Length Distribution")

    output_path = output_dir / f"{csv_path.stem}_histogram.png"
    fig.savefig(output_path)
    log.info(f"Saved histogram to {output_path}")

    plt.show()
    plt.close(fig)
