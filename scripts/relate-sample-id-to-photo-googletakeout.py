#!/usr/bin/env python3
"""
Extract 'title' and 'description' fields from Google Photos supplemental
metadata JSON files in a flat directory and write them to a CSV.
"""

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract title/description from Google Photos metadata JSONs."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing the .json metadata files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        help="Path for the output CSV file.",
        default=Path("./data/inference/input_config/sample_identification_lookup.csv")
    )
    return parser.parse_args()


def extract_record(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "filename": json_path.name,
        "title": data.get("title"),          # None if missing
        "description": data.get("description"),  # None if missing
    }


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Error: {args.input_dir} is not a directory.")

    json_files = sorted(args.input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json files found in {args.input_dir}.")

    records = []
    errors = []
    for path in json_files:
        try:
            records.append(extract_record(path))
        except (json.JSONDecodeError, OSError) as exc:
            errors.append((path.name, str(exc)))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "title", "description"])
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} records to {args.output_csv}")
    if errors:
        print(f"\nSkipped {len(errors)} files due to errors:")
        for name, msg in errors:
            print(f"  {name}: {msg}")


if __name__ == "__main__":
    main()
