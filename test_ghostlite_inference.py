
from pathlib import Path
import torch
from custom_yolo import load_custom_yolo

BASE = Path(__file__).resolve().parent

WEIGHTS = (
    BASE / "PVEL_Detection_Repaired"
    / "ghostlite_additional50" / "weights" / "best.pt"
)

if not WEIGHTS.exists():
    raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS}")

device = "0" if torch.cuda.is_available() else "cpu"

model = load_custom_yolo(BASE / "models" / "yolo26_ghostlite.yaml")
model.load(str(WEIGHTS))

print("Checkpoint:", WEIGHTS)
print("Device:", "CUDA GPU" if device == "0" else "CPU")
print("Classes:", model.names)

image = input("Enter the full path to a solar-cell EL image: ").strip().strip('"')

if not Path(image).is_file():
    raise FileNotFoundError(f"Image not found: {image}")

results = model.predict(
    source=image,
    imgsz=640,
    conf=0.25,
    device=device,
    save=True,
    project=str(BASE / "inspection_test"),
    name="ghostlite",
)

for result in results:
    print("\nDetections:")
    if result.boxes is None or len(result.boxes) == 0:
        print("No detections above the confidence threshold.")
        continue

    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        coordinates = box.xyxy[0].tolist()

        print(
            f"{model.names[class_id]} | "
            f"confidence={confidence:.3f} | "
            f"box={coordinates}"
        )

print("\nAnnotated image saved under inspection_test/ghostlite/")
