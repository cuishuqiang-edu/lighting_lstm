"""
TSLib model wrapper.

Wraps PatchTST, iTransformer, Autoformer, TimesNet for the forecast setup.
forward 接受 (X_fc, y_past) 元组, 内部拼接后传给 TSLib 模型.
"""
import sys, os
import torch
import torch.nn as nn
from .base import BaseLightningModel

_tslib_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'Time-Series-Library')
if os.path.isdir(_tslib_path) and _tslib_path not in sys.path:
    sys.path.insert(0, _tslib_path)


def _make_config(**kwargs):
    class Config:
        pass
    cfg = Config()

    defaults = dict(
        task_name='long_term_forecast',
        seq_len=13, pred_len=12,
        enc_in=17, c_out=1, dec_in=17,
        d_model=256, n_heads=8, e_layers=3, d_layers=3,
        d_ff=512, dropout=0.1, activation='gelu',
        factor=3, embed='timeF', freq='h',
        label_len=6, moving_avg=4,
        num_class=0,
        output_attention=False, individual=True,
        top_k=5, num_kernels=6,
        patch_len=4, stride=2,
    )
    for k, v in defaults.items():
        setattr(cfg, k, kwargs.pop(k, v))
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


class TSLibWrapper(BaseLightningModel):
    """
    Generic wrapper for TSLib models.

    TSLib forward: (x_enc, x_mark_enc, x_dec, x_mark_dec)
      x_enc:  (B, L, F)
      x_mark_enc: (B, L, 4)  — dummy zeros
      x_dec:  (B, L+H, F)    — decoder input
      x_mark_dec: (B, L+H, 4)
      → (B, H, c_out)
    """
    def __init__(self, model_class, n_features, L, H, past_hours=6, **kwargs):
        super().__init__(n_features, L, H)
        self.model_name = kwargs.pop('model_name', 'tslib')
        self.past_hours = past_hours
        in_dim = n_features + past_hours
        self.label_len = kwargs.get('label_len', 6)

        cfg = _make_config(
            seq_len=L, pred_len=H,
            enc_in=in_dim, c_out=1, dec_in=in_dim,
            **kwargs,
        )
        self.model = model_class(cfg)

    def forward(self, x):
        X_fc, y_past = x
        yp = y_past.unsqueeze(1).expand(-1, X_fc.size(1), -1)
        x_enc = torch.cat([X_fc, yp], dim=-1)  # (B, L, 17)

        B, L, F = x_enc.shape
        device, dt = x_enc.device, x_enc.dtype

        # Dummy time features
        x_mark_enc = torch.zeros(B, L, 4, device=device, dtype=dt)

        # Decoder input
        dec_len = self.label_len + self.H
        x_dec = torch.zeros(B, dec_len, F, device=device, dtype=dt)
        if self.label_len > 0 and L >= self.label_len:
            x_dec[:, :self.label_len] = x_enc[:, -self.label_len:]
        x_dec[:, self.label_len:] = x_enc[:, -1:]  # repeat last for prediction
        x_mark_dec = torch.zeros(B, dec_len, 4, device=device, dtype=dt)

        out = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return out[:, -self.H:, 0]  # (B, 12)


# ===== Factory functions =====

def _import_model(module_name):
    import importlib.util
    path = os.path.join(os.path.dirname(_tslib_path), 'Time-Series-Library',
                        'models', f'{module_name}.py')
    if not os.path.exists(path):
        path = os.path.join(_tslib_path, 'models', f'{module_name}.py')
    if not os.path.exists(path):
        raise ImportError(f'TSLib model not found: {path}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Model


def create_patchtst(n_features, L, H, **kwargs):
    Model = _import_model('PatchTST')
    return TSLibWrapper(Model, n_features, L, H,
                        model_name='patchtst', **kwargs)


def create_itransformer(n_features, L, H, **kwargs):
    Model = _import_model('iTransformer')
    return TSLibWrapper(Model, n_features, L, H,
                        model_name='itransformer', **kwargs)


def create_autoformer(n_features, L, H, **kwargs):
    Model = _import_model('Autoformer')
    return TSLibWrapper(Model, n_features, L, H,
                        model_name='autoformer', **kwargs)


def create_timesnet(n_features, L, H, **kwargs):
    Model = _import_model('TimesNet')
    return TSLibWrapper(Model, n_features, L, H,
                        model_name='timesnet', **kwargs)


TSLIB_MODELS = {
    'patchtst': create_patchtst,
    'itransformer': create_itransformer,
    'autoformer': create_autoformer,
    'timesnet': create_timesnet,
}
