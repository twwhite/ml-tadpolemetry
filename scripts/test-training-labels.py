from pathlib import Path

train_labels = Path("data/training/training_data/spline_model/train/labels")

for f in train_labels.glob("*.txt"):
    with open(f) as fp:
        content = fp.read()
        values = content.strip().split()
        if len(values) != 20:
            print(f"{f.name}: {len(values)} values")
            print(f"  raw: {repr(content)}")
