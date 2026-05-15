"""
Transformer encoder baseline.
"""
import math
import torch
import torch.nn as nn
from .base import BaseLightningModel


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerPredictor(BaseLightningModel):
    model_name = 'transformer'

    def __init__(self, n_features, L, H, d_model=256, nhead=4,
                 n_layers=4, dropout=0.1, past_hours=6, **kwargs):
        super().__init__(n_features, L, H)
        self.past_hours = past_hours
        in_dim = n_features + past_hours
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=L)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model * L, H)

    def forward(self, x):
        X_fc, y_past = x
        yp = y_past.unsqueeze(1).expand(-1, X_fc.size(1), -1)
        x = torch.cat([X_fc, yp], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.output_proj(x.reshape(x.size(0), -1))
