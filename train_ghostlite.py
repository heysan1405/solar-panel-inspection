from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules.head import Detect

from custom_yolo import load_custom_yolo
from modules.ghost_blocks import GhostBottleneck, GhostConv


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_YAML = PROJECT_ROOT / "models" / "yolo26_ghostlite.yaml"
DATASET_YAML = PROJECT_ROOT / "yolo_dataset_repaired_8class" / "data.yaml"
RUN_PROJECT = PROJECT_ROOT / "PVEL_Detection_Repaired"
RUN_NAME = "ghostlite"
EXPECTED_NAMES = {
    0: "crack",
    1: "finger",
    2: "black_core",
    3: "thick_line",
    4: "star_crack",
    5: "horizontal_dislocation",
    6: "vertical_dislocation",
    7: "short_circuit",
}


def load_and_validate_dataset(dataset_yaml: Path) -> dict[str, Any]:
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML does not exist: {dataset_yaml}")

    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {dataset_yaml}")
    if config.get("nc") != len(EXPECTED_NAMES):
        raise ValueError(
            f"Expected nc={len(EXPECTED_NAMES)} in {dataset_yaml}, got {config.get('nc')!r}"
        )

    raw_names = config.get("names")
    if isinstance(raw_names, list):
        names = {index: name for index, name in enumerate(raw_names)}
    elif isinstance(raw_names, dict):
        names = {int(index): name for index, name in raw_names.items()}
    else:
        raise ValueError(f"Dataset YAML must define class names as a list or mapping")
    if names != EXPECTED_NAMES:
        raise ValueError(
            f"Dataset classes do not match the expected IDs and names: {names!r}"
        )

    dataset_root_value = config.get("path")
    dataset_root = Path(dataset_root_value) if dataset_root_value else dataset_yaml.parent
    if not dataset_root.is_absolute():
        dataset_root = dataset_yaml.parent / dataset_root
    dataset_root = dataset_root.resolve()

    for split in ("train", "val"):
        split_value = config.get(split)
        if not isinstance(split_value, str) or not split_value.strip():
            raise ValueError(f"Dataset YAML must define a {split!r} path")
        split_path = Path(split_value)
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Resolved {split} image directory does not exist: {split_path.resolve()}"
            )

    print(f"Dataset YAML: {dataset_yaml}")
    print(f"Dataset root: {dataset_root}")
    print(f"Verified {len(names)} class IDs (0-7): {names}")
    print("Verified train and validation image directories.")
    return config


def validate_model_graph(model: Any) -> None:
    network = model.model
    detection_heads = [module for module in network.modules() if isinstance(module, Detect)]
    if len(detection_heads) != 1 or network.model[-1] is not detection_heads[0]:
        raise ValueError("Expected one Detect head at the final model layer")

    head = detection_heads[0]
    if head.nc != len(EXPECTED_NAMES) or head.nl != 3:
        raise ValueError(
            f"Unexpected Detect head configuration: nc={head.nc}, nl={head.nl}"
        )
    if len(head.f) != head.nl or any(layer_index not in network.save for layer_index in head.f):
        raise ValueError(
            f"Detect head sources {head.f} are not all saved model connections: {network.save}"
        )
    if not all(float(stride) > 0 for stride in head.stride):
        raise ValueError(f"Detect head has invalid strides: {head.stride}")

    custom_layers = [
        module
        for module in network.modules()
        if isinstance(module, (GhostConv, GhostBottleneck))
    ]
    if not any(isinstance(module, GhostConv) for module in custom_layers):
        raise ValueError("Parsed model contains no registered custom GhostConv layers")
    if not any(isinstance(module, GhostBottleneck) for module in custom_layers):
        raise ValueError("Parsed model contains no registered custom GhostBottleneck layers")

    for index, layer in enumerate(network.model):
        sources = layer.f if isinstance(layer.f, list) else [layer.f]
        for source in sources:
            valid_source = source == -1 or (
                isinstance(source, int) and 0 <= source < index
            )
            if not valid_source:
                raise ValueError(
                    f"Invalid source connection {source!r} at model layer {index}"
                )

    print(
        f"Verified Detect head connections {head.f}, strides "
        f"{[float(stride) for stride in head.stride]}, "
        f"and registered GhostConv/GhostBottleneck layers."
    )


def run_forward_backward_check(model: Any) -> None:
    network = model.model
    check_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network.to(check_device)
    network.args = get_cfg(overrides={"task": "detect"})
    network.train()
    network.zero_grad(set_to_none=True)

    generator = torch.Generator(device="cpu").manual_seed(1234)
    images = torch.rand(
        (2, 3, 640, 640),
        dtype=torch.float32,
        generator=generator,
    ).to(check_device)
    batch = {
        "img": images,
        "batch_idx": torch.tensor([0], dtype=torch.int64, device=check_device),
        "cls": torch.tensor([[0.0]], dtype=torch.float32, device=check_device),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.25, 0.25]], dtype=torch.float32, device=check_device
        ),
    }
    loss_components, _ = network(batch)
    if not torch.isfinite(loss_components).all():
        raise RuntimeError(
            f"Training smoke-test loss is not finite: {loss_components.detach().tolist()}"
        )
    loss_components.sum().backward()

    missing_gradients = [
        name
        for name, parameter in network.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    non_finite_gradients = [
        name
        for name, parameter in network.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    if missing_gradients or non_finite_gradients:
        raise RuntimeError(
            "Training smoke-test gradients are incomplete or non-finite: "
            f"missing={missing_gradients[:10]}, non_finite={non_finite_gradients[:10]}"
        )

    print(
        f"Forward/backward smoke test passed on {check_device}: "
        f"loss components={loss_components.detach().cpu().tolist()}, "
        f"all {sum(parameter.requires_grad for parameter in network.parameters())} "
        "trainable parameters received finite gradients."
    )
    network.zero_grad(set_to_none=True)
    network.cpu()


def validate_reusable_run_directory(run_directory: Path) -> None:
    if not run_directory.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_directory}")

    files = [path for path in run_directory.rglob("*") if path.is_file()]
    args_path = run_directory / "args.yaml"
    if files != [args_path]:
        raise FileExistsError(
            "Refusing to reuse the run directory because it contains files other than "
            f"args.yaml: {files}"
        )

    args = yaml.safe_load(args_path.read_text(encoding="utf-8-sig"))
    expected = {
        "task": "detect",
        "model": str(MODEL_YAML),
        "data": str(DATASET_YAML),
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "workers": 2,
        "patience": 20,
        "project": str(RUN_PROJECT),
        "name": RUN_NAME,
        "save": True,
        "val": True,
        "plots": True,
        "resume": False,
    }
    mismatches = {
        key: (args.get(key), value)
        for key, value in expected.items()
        if args.get(key) != value
    }
    if args.get("device") not in ("0", 0, "cuda:0"):
        mismatches["device"] = (args.get("device"), "0")
    if args.get("save_dir") != str(run_directory):
        mismatches["save_dir"] = (args.get("save_dir"), str(run_directory))
    if mismatches:
        raise ValueError(
            "Refusing to reuse args.yaml because its training settings differ "
            f"from this script: {mismatches}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO26-GhostLite on the repaired eight-class PVEL-AD dataset."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate configuration and run a synthetic forward/backward smoke test without training.",
    )
    parser.add_argument(
        "--reuse-empty-run",
        action="store_true",
        help=(
            "Explicitly reuse the target run directory only if it contains the matching "
            "args.yaml and no result/checkpoint files."
        ),
    )
    args = parser.parse_args()

    if not MODEL_YAML.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {MODEL_YAML}")
    load_and_validate_dataset(DATASET_YAML)

    # load_custom_yolo registers the custom Ghost modules before Ultralytics parses the YAML.
    model = load_custom_yolo(MODEL_YAML)
    model_class_count = int(model.model.yaml.get("nc", -1))
    if model_class_count != len(EXPECTED_NAMES):
        raise ValueError(
            f"Model YAML defines {model_class_count} classes; expected {len(EXPECTED_NAMES)}"
        )
    validate_model_graph(model)
    run_forward_backward_check(model)
    print(f"Loaded custom detection model from: {MODEL_YAML}")

    run_directory = RUN_PROJECT / RUN_NAME
    reuse_existing_run = False
    if run_directory.exists():
        if not args.reuse_empty_run:
            raise FileExistsError(
                f"Refusing to overwrite an existing run directory: {run_directory}. "
                "Use --reuse-empty-run only if it contains no experiment artifacts."
            )
        validate_reusable_run_directory(run_directory)
        reuse_existing_run = True

    if args.check_only:
        if reuse_existing_run:
            print(f"Verified safe empty run-directory reuse: {run_directory}")
        else:
            print(f"Run-directory target (not created): {run_directory}")
        print("Configuration check passed; training was not started.")
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "Training requires the requested device cuda:0, but CUDA is not available."
        )

    model.train(
        data=str(DATASET_YAML),
        epochs=100,
        imgsz=640,
        device="cuda:0",
        batch=8,
        workers=2,
        patience=20,
        save=True,
        val=True,
        plots=True,
        project=str(RUN_PROJECT),
        name=RUN_NAME,
        exist_ok=reuse_existing_run,
        resume=False,
    )


if __name__ == "__main__":
    main()
