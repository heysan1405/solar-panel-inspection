
from ultralytics import YOLO
from ultralytics.nn import tasks

from modules.ghost_blocks import (
    GhostConv as CustomGhostConv,
    GhostBottleneck as CustomGhostBottleneck,
)

# Register the custom classes for YAML parsing.
# Ultralytics' parser uses these names when resolving model modules.
tasks.GhostConv = CustomGhostConv
tasks.GhostBottleneck = CustomGhostBottleneck


def load_custom_yolo(yaml_path):
    """Load a custom YOLO YAML after registering Ghost modules."""
    return YOLO(str(yaml_path), task="detect")
