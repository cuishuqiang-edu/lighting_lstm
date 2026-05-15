"""
2D U-Net for full-grid lightning prediction.

Input:  (B, 149, H, W)   — 143 forecast channels + 6 past lightning (log1p)
Output: (B, 12, H, W)    — per-hour lightning log-probability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLightningModel


class DoubleConv(nn.Module):
    """3×3 Conv → BN → ReLU → 3×3 Conv → BN → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """MaxPool → DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsample → Concat → DoubleConv."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle spatial size mismatch from odd dimensions
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(BaseLightningModel):
    model_name = 'unet'

    def __init__(self, n_features, L, H, **kwargs):
        """
        Args:
            n_features: Input channels (149 in grid mode)
            L: Forecast window length (for compat, unused by U-Net)
            H: Output channels (12)
        """
        super().__init__(n_features, L, H)

        # Encoder
        self.inc = DoubleConv(n_features, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)

        # Output projection
        self.out = nn.Conv2d(64, H, kernel_size=1)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            raise RuntimeError(
                'U-Net requires grid-mode input (single tensor). '
                'Use --grid_mode in train.py.'
            )

        # Encoder
        x1 = self.inc(x)        # (B, 64, H, W)
        x2 = self.down1(x1)     # (B, 128, H/2, W/2)
        x3 = self.down2(x2)     # (B, 256, H/4, W/4)
        x4 = self.down3(x3)     # (B, 512, H/8, W/8)

        # Bottleneck
        x5 = self.bottleneck(x4)  # (B, 1024, H/16, W/16)

        # Decoder with skip connections
        x = self.up1(x5, x4)    # (B, 512, H/8, W/8)
        x = self.up2(x, x3)     # (B, 256, H/4, W/4)
        x = self.up3(x, x2)     # (B, 128, H/2, W/2)
        x = self.up4(x, x1)     # (B, 64, H, W)

        return self.out(x)       # (B, 12, H, W)
