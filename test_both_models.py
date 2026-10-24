
from pathlib import Path
import torch
from custom_yolo import load_custom_yolo

BASE = Path(__file__).resolve().parent
CONFIGS = [
    BASE / "models" / "yolo26_ghost.yaml",
    BASE / "models" / "yolo26_ghostlite.yaml",
]

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Device:", device)

for config in CONFIGS:
    print(f"\nTesting: {config.name}")

    model = load_custom_yolo(config)
    model.model = model.model.to(device).eval()

    x = torch.zeros(1, 3, 640, 640, device=device)

    with torch.no_grad():
        output = model.model(x)

    print("Forward pass: SUCCESS")
    print("Output type:", type(output).__name__)

    del model, x, output
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

print("\nBoth models passed the forward test.")
