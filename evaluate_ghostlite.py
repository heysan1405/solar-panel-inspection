
from pathlib import Path
import json
import yaml
import torch
from custom_yolo import load_custom_yolo

BASE = Path(__file__).resolve().parent

DATA_YAML = BASE / "yolo_dataset_repaired_8class" / "data.yaml"
MODEL_YAML = BASE / "models" / "yolo26_ghostlite.yaml"
WEIGHTS = (
    BASE / "PVEL_Detection_Repaired"
    / "ghostlite" / "weights" / "best.pt"
)

OUTPUT_DIR = BASE / "evaluation_results"
RUN_NAME = "ghostlite_original_validation"

def main():
    for path in (DATA_YAML, MODEL_YAML, WEIGHTS):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}

    print("Loading original GhostLite checkpoint...")
    model = load_custom_yolo(MODEL_YAML)
    model.load(str(WEIGHTS))

    # Ensure results use the actual repaired dataset class names.
    model.model.names = names

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Weights: {WEIGHTS}")
    print("\nRunning validation. This may take a few minutes...\n")

    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=8,
        device=device,
        plots=True,
        project=str(OUTPUT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        verbose=True,
    )

    box = metrics.box

    report = {
        "weights": str(WEIGHTS),
        "dataset": str(DATA_YAML),
        "imgsz": 640,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "per_class": {},
    }

    print("\n========== OVERALL RESULTS ==========")
    print(f"Precision:  {box.mp:.4f} ({box.mp * 100:.2f}%)")
    print(f"Recall:     {box.mr:.4f} ({box.mr * 100:.2f}%)")
    print(f"mAP50:      {box.map50:.4f} ({box.map50 * 100:.2f}%)")
    print(f"mAP50-95:   {box.map:.4f} ({box.map * 100:.2f}%)")

    print("\n========== PER-CLASS RESULTS ==========")
    for position, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        precision, recall, ap50, ap = box.class_result(position)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        class_name = names.get(class_id, str(class_id))

        report["per_class"][class_name] = {
            "class_id": class_id,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        }

        print(
            f"{class_name:28s} "
            f"P={precision:.3f}  "
            f"R={recall:.3f}  "
            f"F1={f1:.3f}  "
            f"mAP50={ap50:.3f}  "
            f"mAP50-95={ap:.3f}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "ghostlite_original_validation.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nFull report saved to:\n{report_path}")
    print(f"Validation plots and outputs are under:\n{OUTPUT_DIR / RUN_NAME}")

if __name__ == "__main__":
    main()
