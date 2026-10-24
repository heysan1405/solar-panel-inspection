
import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv


class GhostConv(nn.Module):
    """Ghost convolution with a primary and cheap feature-generation branch."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()

        c_primary = c2 // 2
        c_cheap = c2 - c_primary

        self.primary = Conv(c1, c_primary, k, s, g=1, act=act)
        self.cheap = Conv(
            c_primary, c_cheap, 5, 1,
            g=c_primary, act=act
        )

    def forward(self, x):
        y = self.primary(x)
        z = self.cheap(y)
        return torch.cat((y, z), dim=1)


class GhostBottleneck(nn.Module):
    """Lightweight Ghost bottleneck with optional spatial downsampling."""

    def __init__(self, c1, c2, stride=1):
        super().__init__()

        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")

        self.ghost1 = GhostConv(c1, c2, 1, 1)

        if stride == 2:
            self.dw = nn.Sequential(
                nn.Conv2d(
                    c2, c2, 3, stride=2, padding=1,
                    groups=c2, bias=False
                ),
                nn.BatchNorm2d(c2),
            )
        else:
            self.dw = nn.Identity()

        self.ghost2 = GhostConv(c2, c2, 1, 1, act=False)

        if stride == 1 and c1 == c2:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    c1, c1, 3, stride=stride,
                    padding=1, groups=c1, bias=False
                ),
                nn.BatchNorm2d(c1),
                nn.Conv2d(c1, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
            )

    def forward(self, x):
        y = self.ghost1(x)
        y = self.dw(y)
        y = self.ghost2(y)
        return y + self.shortcut(x)
