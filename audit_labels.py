
from pathlib import Path
import cv2
import yaml

BASE = Path(__file__).resolve().parent
DATA_YAML = BASE / "yolo_dataset_repaired_8class" / "data.yaml"

with open(DATA_YAML, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data["names"]
if isinstance(names, dict):
    names = {int(k): v for k, v in names.items()}
else:
    names = dict(enumerate(names))

image_dir = BASE / "yolo_dataset_repaired_8class" / "images" / "val"
label_dir = BASE / "yolo_dataset_repaired_8class" / "labels" / "val"

# Inspect a few images first.
image_names = [
    "img000071.jpg",
    "img000065.jpg",
    "img000059.jpg",
]

out_dir = BASE / "label_audit"
out_dir.mkdir(exist_ok=True)

for image_name in image_names:
    image_path = image_dir / image_name
    label_path = label_dir / (Path(image_name).stem + ".txt")

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        continue

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read: {image_path}")
        continue

    h, w = image.shape[:2]

    if not label_path.exists():
        print(f"No label file: {label_path}")
        continue

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            print(f"Invalid annotation in {label_path}: {line.strip()}")
            continue

        cls, xc, yc, bw, bh = parts
        cls = int(cls)
        xc, yc, bw, bh = map(float, (xc, yc, bw, bh))

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        name = names.get(cls, str(cls))
        cv2.putText(
            image, name, (max(0, x1), max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    output = out_dir / image_name
    cv2.imwrite(str(output), image)
    print(f"Saved: {output}")

print("\nOpen the images in the label_audit folder.")
