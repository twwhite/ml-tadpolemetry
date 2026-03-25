"""Check YOLO pose label files for keypoints outside bounding boxes."""

import sys
from pathlib import Path

KEYPOINT_NAMES = [
    "pos_rostrum",
    "pos_tailbase",
    "pos_tailbase_third",
    "pos_tailtip_third",
    "pos_tailtip",
]


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
    for path in label_files:
        violations = check_label_file(path)
        if violations:
            flagged += 1
            print(f"FLAGGED: {path.name}")
            for v in violations:
                print(v)
            print()

    print(f"{flagged}/{len(label_files)} files have keypoints outside bounding box")


if __name__ == "__main__":
    main()
