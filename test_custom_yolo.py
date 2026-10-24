
from ultralytics.nn import tasks
from modules.ghost_blocks import GhostConv, GhostBottleneck

import custom_yolo

print("GhostConv registered:", tasks.GhostConv is GhostConv)
print("GhostBottleneck registered:", tasks.GhostBottleneck is GhostBottleneck)

print("Custom loader imported successfully.")
