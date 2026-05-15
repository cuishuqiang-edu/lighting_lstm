"""
Lightning prediction models.

Includes:
  - MLP: Simple feed-forward baseline
  - LSTM: Recurrent baseline
  - PatchTSTWrapper: TSLib PatchTST integration
  - iTransformerWrapper: TSLib iTransformer integration
  - create_model: Factory function

Usage:
    from models import create_model
    model = create_model('lstm', n_features=11, L=168, H=24, d_model=128)
"""
import torch
import torch.nn as nn
import math


class MLP(nn.Module):
    """
    Multi-Layer Perceptron baseline.

    Flattens input (L, n_features) → predicts H steps directly.

    Args:
        n_features: Number of input features
        L: Input window length
        H: Output window length
        d_model: Hidden dimension
        n_layers: Number of hidden layers
    """
    def __init__(self, n_features: int, L: int, H: int,
                 d_model: int = 512, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.flatten = nn.Flatten()
        input_dim = L * n_features

        layers = []
        for i in range(n_layers):
            in_dim = input_dim if i == 0 else d_model
            out_dim = d_model if i < n_layers - 1 else H
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_layers - 1 else nn.Identity(),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, n_features)
        B, L, F = x.shape
        x = x.reshape(B, L * F)  # (batch, L*n_features)
        return self.net(x)        # (batch, H)


class LSTMPredictor(nn.Module):
    """
    LSTM-based lightning predictor.

    Args:
        n_features: Number of input features
        L: Input window length
        H: Output window length
        d_model: Hidden dimension
        n_layers: Number of LSTM layers
        dropout: Dropout rate
    """
    def __init__(self, n_features: int, L: int, H: int,
                 d_model: int = 128, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.L = L
        self.H = H
        self.n_features = n_features

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.proj = nn.Linear(d_model, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, n_features)
        _, (h_n, _) = self.lstm(x)    # h_n: (n_layers, batch, d_model)
        last_hidden = h_n[-1]          # (batch, d_model)
        return self.proj(last_hidden)  # (batch, H)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer models."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerPredictor(nn.Module):
    """
    Standalone Transformer encoder for lightning prediction.

    Args:
        n_features: Number of input features
        L: Input window length
        H: Output window length
        d_model: Model dimension
        nhead: Number of attention heads
        n_layers: Number of transformer encoder layers
    """
    def __init__(self, n_features: int, L: int, H: int,
                 d_model: int = 256, nhead: int = 8, n_layers: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=L)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model * L, H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, n_features)
        x = self.input_proj(x)        # (batch, L, d_model)
        x = self.pos_enc(x)           # (batch, L, d_model)
        x = self.encoder(x)           # (batch, L, d_model)
        B, L, D = x.shape
        x = x.reshape(B, L * D)       # (batch, L*d_model)
        return self.output_proj(x)    # (batch, H)


# ===== TSLib Model Wrappers =====

class TSLibWrapper(nn.Module):
    """
    Generic wrapper for TSLib models (PatchTST, iTransformer, Autoformer, etc.).

    TSLib models expect:
      - x: (batch, n_features, L)  [注意: 维度顺序是 (B, F, L)]
      - Returns: (batch, H)

    This wrapper handles dimension permutation and TSLib model construction.

    Args:
        model_class: TSLib model class (e.g. Model from PatchTST.py)
        n_features: Number of input features
        L: Input length
        H: Output length
        model_kwargs: Additional kwargs passed to the TSLib model
    """
    def __init__(self, model_class, n_features: int, L: int, H: int,
                 **model_kwargs):
        super().__init__()
        self.L = L
        self.n_features = n_features

        # Common TSLib config
        default_kwargs = dict(
            enc_in=n_features,          # encoder input size
            seq_len=L,                  # input sequence length
            pred_len=H,                 # prediction length
            individual=True,           # each feature has its own head
        )
        # Merge defaults with user overrides
        for k, v in default_kwargs.items():
            model_kwargs.setdefault(k, v)

        self.model = model_class(**model_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, L, n_features)
        x = x.permute(0, 2, 1)         # → (batch, n_features, L)  TSLib convention
        return self.model(x)            # (batch, H)


def create_patchtst(n_features: int, L: int, H: int, **kwargs):
    """Create a TSLib PatchTST model."""
    try:
        import sys
        sys.path.insert(0, 'Time-Series-Library')
        from models.PatchTST import Model
        return TSLibWrapper(Model, n_features, L, H, **kwargs)
    except ImportError as e:
        raise ImportError(
            f"Cannot import PatchTST from TSLib: {e}\n"
            "Make sure Time-Series-Library is in the Python path."
        )


def create_itransformer(n_features: int, L: int, H: int, **kwargs):
    """Create a TSLib iTransformer model."""
    try:
        import sys
        sys.path.insert(0, 'Time-Series-Library')
        from models.iTransformer import Model
        return TSLibWrapper(Model, n_features, L, H, **kwargs)
    except ImportError as e:
        raise ImportError(
            f"Cannot import iTransformer from TSLib: {e}\n"
            "Make sure Time-Series-Library is in the Python path."
        )


def create_autoformer(n_features: int, L: int, H: int, **kwargs):
    """Create a TSLib Autoformer model."""
    try:
        import sys
        sys.path.insert(0, 'Time-Series-Library')
        from models.Autoformer import Model
        return TSLibWrapper(Model, n_features, L, H, **kwargs)
    except ImportError as e:
        raise ImportError(f"Cannot import Autoformer: {e}")


def create_timesnet(n_features: int, L: int, H: int, **kwargs):
    """Create a TSLib TimesNet model."""
    try:
        import sys
        sys.path.insert(0, 'Time-Series-Library')
        from models.TimesNet import Model
        return TSLibWrapper(Model, n_features, L, H, **kwargs)
    except ImportError as e:
        raise ImportError(f"Cannot import TimesNet: {e}")


# ===== Factory =====

MODEL_REGISTRY = {
    'mlp': MLP,
    'lstm': LSTMPredictor,
    'transformer': TransformerPredictor,
    'patchtst': create_patchtst,
    'itransformer': create_itransformer,
    'autoformer': create_autoformer,
    'timesnet': create_timesnet,
}


def create_model(name: str, n_features: int, L: int, H: int, **kwargs):
    """
    Create a model by name.

    Args:
        name: Model name ('mlp', 'lstm', 'transformer', 'patchtst', etc.)
        n_features: Number of input features
        L: Input window length (hours)
        H: Output window length (hours)
        **kwargs: Model-specific keyword arguments

    Returns:
        nn.Module

    Example:
        model = create_model('lstm', n_features=11, L=168, H=24, d_model=128)
        model = create_model('patchtst', n_features=11, L=168, H=24, d_model=256)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}"
        )

    creator = MODEL_REGISTRY[name]
    if name in ('mlp', 'lstm', 'transformer'):
        return creator(n_features=n_features, L=L, H=H, **kwargs)
    else:
        # TSLib models
        return creator(n_features=n_features, L=L, H=H, **kwargs)


if __name__ == '__main__':
    # Test models
    B, F, L, H = 4, 11, 168, 24
    x = torch.randn(B, L, F)

    for name in ['mlp', 'lstm', 'transformer']:
        model = create_model(name, n_features=F, L=L, H=H)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'{name:12s} | output {list(out.shape)} | params {n_params:,}')

    # Test TSLib wrapper (requires TSLib)
    try:
        model = create_model('patchtst', n_features=F, L=L, H=H, d_model=128)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'patchtst     | output {list(out.shape)} | params {n_params:,}')
    except ImportError as e:
        print(f'patchtst     | SKIP: {e}')
