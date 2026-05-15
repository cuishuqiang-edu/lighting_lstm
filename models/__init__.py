"""
Model registry with auto-discovery.

To add a new model:
  1. Create a file in models/ (e.g. models/cnn.py)
  2. Make your class inherit from BaseLightningModel
  3. Set class attribute `model_name = 'my_model'`
  4. It's automatically discovered — no registration needed!

Usage:
    from models import create_model, list_models, MODEL_REGISTRY

    # List all available models
    print(list_models())

    # Create a model
    model = create_model('lstm', n_features=11, L=168, H=24, d_model=128)
"""
import os
import importlib
import inspect
from .base import BaseLightningModel


# ===== Auto-discovery =====

def _discover_models():
    """
    Scan models/*.py for classes inheriting BaseLightningModel.
    Returns {model_name: class_or_factory, ...}
    """
    registry = {}
    models_dir = os.path.dirname(__file__)
    py_files = [f for f in os.listdir(models_dir)
                if f.endswith('.py') and f != '__init__.py' and f != 'base.py']

    for fname in py_files:
        module_name = f'models.{fname[:-3]}'
        try:
            mod = importlib.import_module(module_name)
        except ImportError as e:
            # TSLib models may fail if TSLib isn't installed — skip silently
            if 'Time-Series-Library' not in str(e):
                print(f'  [models] WARN: {module_name} failed: {e}')
            continue

        # Find model classes
        for name, obj in inspect.getmembers(mod):
            if (inspect.isclass(obj) and issubclass(obj, BaseLightningModel)
                    and obj is not BaseLightningModel):
                if hasattr(obj, 'model_name') and obj.model_name != 'base':
                    registry[obj.model_name] = obj

        # Find TSLib model factories (MODELS dict)
        if hasattr(mod, 'TSLIB_MODELS'):
            registry.update(mod.TSLIB_MODELS)

    return registry


MODEL_REGISTRY = _discover_models()


def list_models():
    """List all available model names."""
    return sorted(MODEL_REGISTRY.keys())


def create_model(name, n_features, L, H, **kwargs):
    """
    Create a model by name.

    Args:
        name: Model name ('mlp', 'lstm', 'patchtst', etc.)
        n_features: Number of input features
        L: Input window length (hours)
        H: Prediction window length (hours)
        **kwargs: Model-specific hyperparameters

    Returns:
        BaseLightningModel instance

    Example:
        model = create_model('lstm', n_features=11, L=168, H=24)
        model = create_model('patchtst', n_features=11, L=168, H=24,
                            d_model=256, n_heads=8)
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Available: {list_models()}"
        )

    creator = MODEL_REGISTRY[name]
    return creator(n_features=n_features, L=L, H=H, **kwargs)


# Print discovery on import
_discovered = list_models()
print(f'[models] Discovered {len(_discovered)} models: {", ".join(_discovered)}')
