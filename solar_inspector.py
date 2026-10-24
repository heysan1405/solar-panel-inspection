
from pathlib import Path
import base64
import json
import requests
import torch
import yaml

from custom_yolo import load_custom_yolo

BASE = Path(__file__).resolve().parent

DATA_YAML = BASE / "yolo_dataset_repaired_8class" / "data.yaml"
MODEL_YAML = BASE / "models" / "yolo26_ghostlite.yaml"
WEIGHTS = (
    BASE / "PVEL_Detection_Repaired"
    / "ghostlite" / "weights" / "best.pt"
)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:4b-instruct"

# Load the authoritative class names from the repaired dataset YAML.
with open(DATA_YAML, "r", encoding="utf-8") as f:
    dataset = yaml.safe_load(f)

names = dataset["names"]
if isinstance(names, list):
    names = {i: name for i, name in enumerate(names)}
else:
    names = {int(k): v for k, v in names.items()}

if len(names) != 8:
    raise ValueError(f"Expected 8 classes, found {len(names)}")

if not MODEL_YAML.exists() or not WEIGHTS.exists():
    raise FileNotFoundError("GhostLite model YAML or checkpoint is missing.")

device = "0" if torch.cuda.is_available() else "cpu"

print("Loading GhostLite...")
detector = load_custom_yolo(MODEL_YAML)
detector.load(str(WEIGHTS))

print("Dataset class names:", names)
print("Device:", "CUDA GPU" if device == "0" else "CPU")


def inspect_image(image_path):
    image_path = Path(image_path).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. Detect defects with YOLO26-GhostLite.
    results = detector.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.25,
        device=device,
        save=True,
        project=str(BASE / "inspection_test"),
        name="combined",
        exist_ok=True,
        verbose=False,
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append({
                "class_id": class_id,
                "class_name": names.get(class_id, f"unknown_{class_id}"),
                "confidence": round(float(box.conf.item()), 4),
                "box_xyxy": [
                    round(float(v), 2)
                    for v in box.xyxy[0].tolist()
                ],
            })

    print("\nYOLO detections:")
    print(json.dumps(detections, indent=2))

    # 2. Encode the original image for Ollama's vision API.
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    # 3. Ask Qwen to interpret the actual detector findings.
    prompt = f"""
You are an assistant for preliminary solar-cell electroluminescence
(EL) defect inspection.

The following are the actual detections produced by a trained YOLO
defect detector. Do not invent additional detector findings.

DETECTIONS:
{json.dumps(detections, indent=2)}

Explain:
1. Which defect classes the detector reported.
2. The confidence scores and locations, in plain language.
3. What these results may indicate.
4. Whether further inspection is advisable.

If the detections list is empty, say that no defects were detected
above the current threshold; do not conclude the cell is defect-free.

Distinguish model predictions from verified physical defects.
Do not claim that a panel is safe, unsafe, or definitely needs repair
based only on one image. If evidence is insufficient, say so.
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": [{
                "role": "user",
                "content": prompt,
                "images": [image_b64],
            }],
            "stream": False,
        },
        timeout=300,
    )
    response.raise_for_status()

    report = response.json()["message"]["content"]

    print("\nQwen3-VL inspection report:\n")
    print(report)

    return {
        "image": str(image_path),
        "detections": detections,
        "report": report,
    }


if __name__ == "__main__":
    image = input("Enter the full path to a solar-cell EL image: ")
    output = inspect_image(image.strip().strip('"'))

    output_path = BASE / "inspection_test" / "combined_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print(f"\nStructured report saved to: {output_path}")
