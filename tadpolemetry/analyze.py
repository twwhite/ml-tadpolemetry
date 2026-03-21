import csv
from pathlib import Path
import statistics
import matplotlib.pyplot as plt
import shutil

from .logging import get_logger

log = get_logger(__name__)

DEFAULT_BINS = 10
OUTLIER_ZSCORE_THRESHOLD = 2.0


def copy_review_images(csv_path: Path, input_dir: Path, output_dir: Path):
    """Copy images flagged for review to output_dir/review/."""
    review_dir = output_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            needs_review = (
                row.get("random_validate", "False") == "True"
                or row.get("review_outlier", "False") == "True"
            )
            if needs_review:
                src = input_dir / row["filename"]
                if src.exists():
                    shutil.copy2(src, review_dir / row["filename"])
                    copied += 1
                else:
                    log.warning(f"Image not found for review copy: {src}")

    log.info(f"Copied {copied} images to {review_dir}")

def _zscore_outliers(values: list[float], threshold: float) -> list[bool]:
    """Return a bool list — True where value is a zscore outlier."""
    if len(values) < 2:
        return [False] * len(values)
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0:
        return [False] * len(values)
    return [abs((v - mean) / stdev) > threshold for v in values]


def flag_outliers(csv_path: Path, zscore_threshold: float = OUTLIER_ZSCORE_THRESHOLD):
    """Read results CSV, flag outliers, overwrite with review_outlier column added."""
    log.debug("Starting flag outliers ...")
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not rows:
        log.warning("No rows found in CSV")
        return

    # Extract numeric columns, keeping track of which rows have valid values
    lengths = []
    ruler_spacings = []
    for row in rows:
        try:
            lengths.append(float(row["length_mm"]) if row["length_mm"] else None)
        except ValueError:
            lengths.append(None)
        try:
            ruler_spacings.append(float(row["scale_mean_interval_px"]) if row["scale_mean_interval_px"] else None)
        except ValueError:
            ruler_spacings.append(None)

    # Compute zscores only on valid values
    valid_lengths = [v for v in lengths if v is not None]
    valid_spacings = [v for v in ruler_spacings if v is not None]

    length_outliers = _zscore_outliers(valid_lengths, zscore_threshold)
    spacing_outliers = _zscore_outliers(valid_spacings, zscore_threshold)

    # Map back to original row indices
    length_outlier_map = {}
    spacing_outlier_map = {}
    li = si = 0
    for i, row in enumerate(rows):
        if lengths[i] is not None:
            length_outlier_map[i] = length_outliers[li]
            li += 1
        if ruler_spacings[i] is not None:
            spacing_outlier_map[i] = spacing_outliers[si]
            si += 1

    # Write back with review_outlier column
    new_fieldnames = list(fieldnames) + ["review_outlier"] if "review_outlier" not in fieldnames else list(fieldnames)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            is_outlier = length_outlier_map.get(i, False) or spacing_outlier_map.get(i, False)
            row["review_outlier"] = is_outlier
            writer.writerow(row)

    n_outliers = sum(length_outlier_map.get(i, False) or spacing_outlier_map.get(i, False) for i in range(len(rows)))
    log.info(f"Flagged {n_outliers}/{len(rows)} images as outliers")


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

    # Optionally show plot
    # plt.show()
    # plt.close(fig)
