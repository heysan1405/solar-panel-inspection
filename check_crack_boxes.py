
from pathlib import Path
from collections import Counter

LABEL_DIR = (
    Path(__file__).resolve().parent
    / "yolo_dataset_repaired_8class"
    / "labels"
)

class_names = {
    0: "crack",
    1: "finger",
    2: "black_core",
    3: "thick_line",
    4: "star_crack",
    5: "horizontal_dislocation",
    6: "vertical_dislocation",
    7: "short_circuit",
}

for split in ("train", "val"):
    folder = LABEL_DIR / split
    if not folder.exists():
        print(f"Missing folder: {folder}")
        continue

    counts = Counter()
    crack_boxes = []

    for file in folder.glob("*.txt"):
        for line_number, line in enumerate(
            file.read_text(encoding="utf-8").splitlines(), 1
        ):
            parts = line.split()
            if len(parts) != 5:
                continue

            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])

            counts[cls] += 1

            if cls == 0:
                crack_boxes.append(
                    (file.name, line_number, xc, yc, w, h)
                )

    print(f"\n===== {split.upper()} =====")
    print("Class counts:")
    for cls, name in class_names.items():
        print(f"  {name:25s}: {counts[cls]}")

    print(f"\nTotal crack boxes: {len(crack_boxes)}")
    print("Tall crack boxes (height > 0.50):")

    tall = sorted(
        (box for box in crack_boxes if box[5] > 0.50),
        key=lambda box: box[5],
        reverse=True,
    )

    for filename, line, xc, yc, w, h in tall[:30]:
        print(
            f"  {filename}: line={line}, "
            f"width={w:.3f}, height={h:.3f}, "
            f"center=({xc:.3f}, {yc:.3f})"
        )

    print(f"Total tall crack boxes: {len(tall)}")

    
