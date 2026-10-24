
from pathlib import Path
from custom_yolo import load_custom_yolo

BASE_DIR = Path(__file__).resolve().parent
yaml_path = BASE_DIR / "models" / "yolo26_ghost.yaml"

print("Checking YAML:", yaml_path)
print("File exists:", yaml_path.is_file())

if not yaml_path.is_file():
    raise FileNotFoundError(
        f"YAML not found: {yaml_path}\n"
        "Check the models folder and filename."
    )

model = load_custom_yolo(yaml_path)
model.info()

print("YOLO26-Ghost model constructed successfully.")
