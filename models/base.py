"""
Base model interface for lightning prediction.

All models must:
  - Inherit from BaseLightningModel
  - Accept (n_features, L, H, **kwargs) in __init__
  - Return (batch, H) from forward(x) where x is (X_fc, y_past) tuple
"""
import torch.nn as nn


class BaseLightningModel(nn.Module):
    """Base class for all lightning prediction models."""

    model_name = 'base'

    def __init__(self, n_features: int, L: int, H: int, **kwargs):
        super().__init__()
        self.n_features = n_features
        self.L = L
        self.H = H

    def forward(self, x):
        """
        x: tuple (X_fc, y_past)
          X_fc:  (batch, 13, 11)   GRAPES forecast trajectory
          y_past: (batch, 6)       past ADTD lightning (log1p)
        Returns: (batch, H)
        """
        raise NotImplementedError

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        return f'{self.model_name} | params={n:,} | L={self.L} H={self.H}'
