"""
Configuration management for LSTF-for-Lightning.

Loads YAML config files and provides model-specific presets.

Usage:
    from config import load_config, get_model_config

    # Load from YAML file
    config = load_config('configs/patchtst.yaml')

    # Or use presets
    config = get_model_config('patchtst', H=24)
"""
import os
import yaml
from typing import Any, Dict


# ===== Model Presets =====

MODEL_PRESETS = {
    'mlp': {
        'model': 'mlp',
        'd_model': 512,
        'n_layers': 3,
        'dropout': 0.2,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'description': 'Simple MLP baseline (flatten + FC layers)',
    },
    'lstm': {
        'model': 'lstm',
        'd_model': 128,
        'n_layers': 2,
        'dropout': 0.2,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 256,
        'description': 'LSTM baseline (2-layer, 128 hidden)',
    },
    'transformer': {
        'model': 'transformer',
        'd_model': 256,
        'nhead': 4,
        'n_layers': 4,
        'dropout': 0.1,
        'lr': 5e-4,
        'batch_size': 128,
        'description': 'Standard Transformer encoder',
    },
    'patchtst': {
        'model': 'patchtst',
        'd_model': 256,
        'nhead': 8,
        'n_layers': 3,
        'dropout': 0.2,
        'd_ff': 512,
        'patch_len': 24,
        'stride': 12,
        'lr': 1e-4,
        'batch_size': 128,
        'weight_decay': 1e-5,
        'description': 'PatchTST (TSLib) - best for long sequence forecasting',
    },
    'patchtst_large': {
        'model': 'patchtst',
        'd_model': 512,
        'nhead': 8,
        'n_layers': 6,
        'dropout': 0.2,
        'd_ff': 1024,
        'patch_len': 24,
        'stride': 12,
        'lr': 5e-5,
        'batch_size': 64,
        'weight_decay': 1e-5,
        'description': 'PatchTST large (6 layers, 512 dim)',
    },
    'itransformer': {
        'model': 'itransformer',
        'd_model': 256,
        'nhead': 8,
        'n_layers': 3,
        'dropout': 0.1,
        'd_ff': 512,
        'lr': 1e-4,
        'batch_size': 128,
        'weight_decay': 1e-5,
        'description': 'iTransformer (TSLib) - variable-centric attention',
    },
    'autoformer': {
        'model': 'autoformer',
        'd_model': 256,
        'nhead': 4,
        'n_layers': 3,
        'dropout': 0.1,
        'd_ff': 512,
        'moving_avg': 24,
        'factor': 3,
        'lr': 1e-4,
        'batch_size': 128,
        'description': 'Autoformer with seasonal-trend decomposition',
    },
    'timesnet': {
        'model': 'timesnet',
        'd_model': 256,
        'nhead': 4,
        'n_layers': 3,
        'dropout': 0.1,
        'd_ff': 512,
        'top_k': 5,
        'lr': 1e-4,
        'batch_size': 128,
        'description': 'TimesNet - 2D temporal variation',
    },
}


# ===== Default Config =====

DEFAULT_CONFIG = {
    # Data
    'cell_index': 'Data/processed/dataset/cell_index.csv',
    'dataset_dir': 'Data/processed/dataset',
    'n_features': 11,
    'L': 168,
    'H': 24,
    'stride': 24,

    # Training
    'epochs': 50,
    'batch_size': 256,
    'lr': 1e-3,
    'lr_min': 1e-6,
    'weight_decay': 1e-4,
    'clip_grad': 1.0,
    'patience': 10,
    'lr_scheduler': 'cosine',
    'log_interval': 50,

    # Data loading
    'max_train_samples': 100000,
    'max_val_samples': 20000,
    'max_test_samples': 20000,
    'num_workers': 0,

    # Misc
    'seed': 42,
    'save_dir': 'checkpoints/run',
}


def load_config(path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Merges with defaults: YAML values override defaults.

    Args:
        path: Path to YAML config file

    Returns:
        dict of configuration parameters
    """
    config = DEFAULT_CONFIG.copy()

    if not os.path.exists(path):
        raise FileNotFoundError(f'Config file not found: {path}')

    with open(path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    if yaml_config:
        config.update(yaml_config)

    return config


def get_model_config(model_name: str, **overrides) -> Dict[str, Any]:
    """
    Get configuration for a specific model with overrides.

    Args:
        model_name: Model name ('mlp', 'lstm', 'patchtst', etc.)
        **overrides: Override specific parameters

    Returns:
        Full config dict ready for training

    Example:
        config = get_model_config('patchtst', H=24, epochs=100)
        trainer = Trainer(model, train_loader, val_loader, test_loader, config)
    """
    if model_name not in MODEL_PRESETS:
        raise ValueError(f"Unknown model preset: {model_name}. "
                         f"Available: {list(MODEL_PRESETS.keys())}")

    config = DEFAULT_CONFIG.copy()
    config.update(MODEL_PRESETS[model_name])
    config.update(overrides)
    return config


def save_config(config: Dict[str, Any], path: str):
    """Save config to YAML file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f'[Config] Saved {path}')


# ===== Create Example Configs =====

def create_example_configs(output_dir='configs'):
    """Generate example YAML configs for all models."""
    os.makedirs(output_dir, exist_ok=True)

    for name, preset in MODEL_PRESETS.items():
        config = DEFAULT_CONFIG.copy()
        config.update(preset)
        config['save_dir'] = f'checkpoints/{name}'

        path = os.path.join(output_dir, f'{name}.yaml')
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f'  Created {path}')

    # Create an experiment plan
    experiment_plan = {
        'experiments': [
            {'name': 'mlp', 'description': 'MLP baseline'},
            {'name': 'lstm', 'description': 'LSTM baseline'},
            {'name': 'transformer', 'description': 'Transformer baseline'},
            {'name': 'patchtst', 'description': 'PatchTST (recommended)'},
            {'name': 'itransformer', 'description': 'iTransformer'},
        ],
        'note': 'Run experiments in order. Each produces test_results.json',
    }
    with open(os.path.join(output_dir, 'experiment_plan.yaml'), 'w') as f:
        yaml.dump(experiment_plan, f, default_flow_style=False, sort_keys=False)
    print(f'  Created {os.path.join(output_dir, "experiment_plan.yaml")}')


if __name__ == '__main__':
    create_example_configs()
    print('\nExample:')
    print("  from config import get_model_config")
    print("  cfg = get_model_config('patchtst', H=24, epochs=50)")
    print("  print(cfg['model'], cfg['d_model'], cfg['lr'])")
