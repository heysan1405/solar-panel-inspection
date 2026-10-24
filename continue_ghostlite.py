from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics.nn.modules.head import Detect

from custom_yolo import load_custom_yolo
from train_ghostlite import (
    DATASET_YAML,
    EXPECTED_NAMES,
    MODEL_YAML,
    RUN_PROJECT,
    load_and_validate_dataset,
    run_forward_backward_check,
    validate_model_graph,
)


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = (
    RUN_PROJECT / "ghostlite" / "weights" / "best.pt"
).resolve()
RUN_NAME = "ghostlite_additional50"
RUN_DIRECTORY = RUN_PROJECT / RUN_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune the trained YOLO26-GhostLite best checkpoint for 50 more epochs."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate settings and checkpoint without starting training.",
    )
    args = parser.parse_args()

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Trained checkpoint does not exist: {CHECKPOINT}")
    load_and_validate_dataset(DATASET_YAML)

    model = load_custom_yolo(CHECKPOINT)
    network = model.model
    if int(network.yaml.get("nc", -1)) != len(EXPECTED_NAMES):
        raise ValueError(
            f"Checkpoint model has nc={network.yaml.get('nc')}; "
            f"expected {len(EXPECTED_NAMES)}."
        )
    if not isinstance(network.model[-1], Detect):
        raise TypeError("The checkpoint does not load with a detection head.")
    validate_model_graph(model)

    checkpoint_args: dict[str, Any] = getattr(network, "args", {})
    checkpoint_data = checkpoint_args.get("data") if hasattr(checkpoint_args, "get") else None
    if checkpoint_data and Path(checkpoint_data).resolve() != DATASET_YAML.resolve():
        raise ValueError(
            "Checkpoint training data path differs from the repaired dataset: "
            f"{checkpoint_data}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Fine-tuning requires the requested device cuda:0.")

    print(f"Starting weights: {CHECKPOINT}")
    print(f"Checkpoint source run is preserved: {RUN_PROJECT / 'ghostlite'}")
    print(f"New run directory: {RUN_DIRECTORY}")
    print(
        "The saved checkpoint has no optimizer state; this is a 50-epoch "
        "fine-tune from best.pt with a newly initialized optimizer, not an "
        "optimizer-state resume."
    )

    if args.check_only:
        smoke_test_model = load_custom_yolo(MODEL_YAML)
        smoke_test_model.load(CHECKPOINT)
        run_forward_backward_check(smoke_test_model)
        print("Checkpoint and training smoke checks passed; no training was started.")
        return

    if RUN_DIRECTORY.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing continuation run: {RUN_DIRECTORY}"
        )

    model.train(
        data=str(DATASET_YAML),
        epochs=50,
        imgsz=640,
        device=0,
        batch=8,
        workers=2,
        patience=20,
        save=True,
        val=True,
        plots=True,
        project=str(RUN_PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        resume=False,
    )


if __name__ == "__main__":
    main()
