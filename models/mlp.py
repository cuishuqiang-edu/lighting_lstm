"""
MLP baseline.
"""
import torch
import torch.nn as nn
from .base import BaseLightningModel


class MLP(BaseLightningModel):
    model_name = 'mlp'

    def __init__(self, n_features, L, H, d_model=512, n_layers=3,
                 dropout=0.1, past_hours=6, **kwargs):
        super().__init__(n_features, L, H)
        input_dim = L * n_features + past_hours
        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else d_model
            out_dim = d_model if i < n_layers - 1 else H
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        X_fc, y_past = x
        x = X_fc.reshape(X_fc.size(0), -1)
        return self.net(torch.cat([x, y_past], dim=1))
