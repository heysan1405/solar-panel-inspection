
import torch
from modules.ghost_blocks import GhostConv, GhostBottleneck

x = torch.randn(1, 64, 80, 80)

ghost_conv = GhostConv(64, 128, 3, 1)
y = ghost_conv(x)
print("GhostConv output:", y.shape)

ghost_bottleneck = GhostBottleneck(64, 128, stride=2)
z = ghost_bottleneck(x)
print("GhostBottleneck output:", z.shape)
