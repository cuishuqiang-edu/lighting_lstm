"""
LSTM baseline.
"""
import torch
import torch.nn as nn
from .base import BaseLightningModel


class LSTMPredictor(BaseLightningModel):
    model_name = 'lstm'

    def __init__(self, n_features, L, H, d_model=128, n_layers=2,
                 dropout=0.1, past_hours=6, **kwargs):
        super().__init__(n_features, L, H)
        self.past_hours = past_hours
        self.lstm = nn.LSTM(
            input_size=n_features + past_hours,
            hidden_size=d_model,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.proj = nn.Linear(d_model, H)

    def forward(self, x):
        X_fc, y_past = x
        # Expand y_past: (B, 6) → (B, 13, 6)
        yp = y_past.unsqueeze(1).expand(-1, X_fc.size(1), -1)
        x = torch.cat([X_fc, yp], dim=-1)  # (B, 13, 11+6)
        _, (h_n, _) = self.lstm(x)
        return self.proj(h_n[-1])
