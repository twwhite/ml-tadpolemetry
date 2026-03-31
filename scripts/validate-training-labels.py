"""Check YOLO pose label files for keypoints outside bounding boxes."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

KEYPOINT_NAMES = [
    "pos_rostrum",
    "pos_tailbase",
    "pos_tailbase_third",
    "pos_tailtip_third",
    "pos_tailtip",
]


def keypoint_r2(keypoints: list[tuple[float, float]]) -> float:
    """R² of middle keypoints relative to the rostrum-to-tailtip baseline."""
    pts = np.array(keypoints)
    rostrum, tailtip = pts[0], pts[-1]
    axis = tailtip - rostrum
    length = np.linalg.norm(axis)
    if length == 0:
        return 1.0
    axis_unit = axis / length

    residuals = []
    for pt in pts:
        proj = rostrum + np.dot(pt - rostrum, axis_unit) * axis_unit
        residuals.append(np.linalg.norm(pt - proj) ** 2)

    ss_res = sum(residuals)
    ss_tot = sum(np.linalg.norm(pt - pts.mean(axis=0)) ** 2 for pt in pts)
    return float(1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0)


def check_keypoint_sequence(path: Path) -> list[str]:
    """Check that keypoints progress monotonically along the dominant axis."""
    violations = []

    with open(path) as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue

            _, cx, cy, w, h, *kp_values = [float(v) for v in values]

            keypoints = [
                (kp_values[i], kp_values[i + 1]) for i in range(0, len(kp_values), 3)
            ]

            xs = [kp[0] for kp in keypoints]
            ys = [kp[1] for kp in keypoints]

            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            dominant = ys if y_range > x_range else xs

            increasing = all(
                dominant[i] <= dominant[i + 1] for i in range(len(dominant) - 1)
            )
            decreasing = all(
                dominant[i] >= dominant[i + 1] for i in range(len(dominant) - 1)
            )

            if not increasing and not decreasing:
                violations.append(
                    f"  Non-monotonic keypoints along dominant axis: "
                    f"{'y' if y_range > x_range else 'x'} values = "
                    f"{[round(v, 3) for v in dominant]}"
                )

    return violations


def check_label_file(path: Path) -> list[str]:
    """Return list of violation strings for a label file, empty if clean."""
    violations = []

    with open(path) as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue

            _, cx, cy, w, h, *kp_values = [float(v) for v in values]

            # Bounding box edges in normalized coords
            x1 = cx - w / 2
            x2 = cx + w / 2
            y1 = cy - h / 2
            y2 = cy + h / 2

            # Each keypoint is x, y, visibility
            keypoints = [
                (kp_values[i], kp_values[i + 1]) for i in range(0, len(kp_values), 3)
            ]

            for i, (kx, ky) in enumerate(keypoints):
                if not (x1 <= kx <= x2 and y1 <= ky <= y2):
                    name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kp{i}"
                    violations.append(
                        f"  {name}: ({kx:.4f}, {ky:.4f}) outside box "
                        f"[{x1:.4f}-{x2:.4f}, {y1:.4f}-{y2:.4f}]"
                    )

    return violations


def main():
    labels_dir = Path(sys.argv[1])

    if not labels_dir.exists():
        print(f"Directory not found: {labels_dir}")
        sys.exit(1)

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"Checking {len(label_files)} label files in {labels_dir}\n")

    flagged = 0
    r2_scores = []

    for path in label_files:
        violations = check_label_file(path)
        if violations:
            flagged += 1
            print(f"FLAGGED: {path.name}")
            for v in violations:
                print(v)
            print()

        sequence_violations = check_keypoint_sequence(path)
        if sequence_violations:
            flagged += 1
            print(f"FLAGGED (sequence): {path.name}")
            for v in sequence_violations:
                print(v)
            print()

        # R² linearity
        with open(path) as f:
            for line in f:
                values = line.strip().split()
                if not values:
                    continue
                _, cx, cy, w, h, *kp_values = [float(v) for v in values]
                keypoints = [
                    (kp_values[i], kp_values[i + 1])
                    for i in range(0, len(kp_values), 3)
                ]
                r2 = keypoint_r2(keypoints)
                r2_scores.append((r2, path.name))

    print(f"{flagged}/{len(label_files)} files have keypoints outside bounding box\n")

    if r2_scores:
        r2_values = [r for r, _ in r2_scores]
        print("--- Spine linearity (R²) ---")
        print(f"  mean:   {np.mean(r2_values):.4f}")
        print(f"  median: {np.median(r2_values):.4f}")
        print(f"  min:    {np.min(r2_values):.4f}")
        print("\n  Most curved (lowest R²):")
        for r2, name in sorted(r2_scores)[:10]:
            print(f"    {r2:.4f}  {name}")

        buckets = [(0.0, 0.90), (0.90, 0.95), (0.95, 0.99), (0.99, 1.01)]
        print("\n  Distribution by R² bucket:")
        for lo, hi in buckets:
            count = sum(1 for r in r2_values if lo <= r < hi)
            label = (
                f"<{hi:.2f}"
                if lo == 0
                else (f">={lo:.2f}" if hi > 1.0 else f"{lo:.2f}-{hi:.2f}")
            )
            print(f"    {label:12s}  {count:4d}  ({100 * count / len(r2_values):.1f}%)")
        plot_r2_distribution(r2_scores, labels_dir)


def load_keypoints(path: Path) -> list[tuple[float, float]] | None:
    with open(path) as f:
        for line in f:
            values = line.strip().split()
            if not values:
                continue
            _, cx, cy, w, h, *kp_values = [float(v) for v in values]
            return [
                (kp_values[i], kp_values[i + 1]) for i in range(0, len(kp_values), 3)
            ]
    return None


def plot_r2_distribution(r2_scores: list[tuple[float, str]], labels_dir: Path):
    r2_values = [r for r, _ in r2_scores]
    sorted_scores = sorted(r2_scores)

    # Pick 6 examples spread across the R² range
    n = len(sorted_scores)
    indices = [int(i * (n - 1) / 5) for i in range(6)]
    examples = [sorted_scores[i] for i in indices]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Spine Linearity (R²) Distribution", fontsize=13)

    # Histogram
    ax = axes[0, 0]
    ax.hist(r2_values, bins=30, color="steelblue", edgecolor="white")
    ax.axvline(
        np.mean(r2_values),
        color="red",
        linestyle="--",
        label=f"mean={np.mean(r2_values):.3f}",
    )
    ax.set_xlabel("R²")
    ax.set_ylabel("Count")
    ax.set_title("R² Histogram")
    ax.legend(fontsize=8)

    # Blank second slot in top row (layout balance)
    axes[0, 1].axis("off")
    axes[0, 2].axis("off")
    axes[0, 3].axis("off")

    # Example spine plots across bottom row + overflow to top right
    example_axes = [
        axes[1, 0],
        axes[1, 1],
        axes[1, 2],
        axes[1, 3],
        axes[0, 1],
        axes[0, 2],
    ]

    for ax, (r2, name) in zip(example_axes, examples):
        kps = load_keypoints(labels_dir / name)
        if kps is None:
            ax.axis("off")
            continue
        xs, ys = zip(*kps)
        # Normalize to 0-1 for display
        ax.plot(xs, ys, "o-", color="steelblue", markersize=5)
        ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(f"R²={r2:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 3].axis("off")
    plt.tight_layout()
    out = labels_dir / "r2_distribution.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
