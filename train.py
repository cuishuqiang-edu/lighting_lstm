"""
LSTF-for-Lightning — entry point.

用法:
  # YAML 配置 (推荐)
  python train.py --config configs/patchtst.yaml

  # 命令行参数
  python train.py --model lstm --epochs 10 --batch_size 64

  # YAML + 局部覆盖
  python train.py --config configs/lstm.yaml --epochs 100 --lr 0.0005

  # 恢复训练
  python train.py --resume checkpoints/lstm/best.pt

  # 仅测试
  python train.py --resume checkpoints/lstm/best.pt --test_only

  # 列举可用模型
  python train.py --list_models
"""
import argparse, sys
import torch
import numpy as np

from models import create_model, list_models
from dataset import create_dataloaders
from trainer import Trainer


# ===== Args =====

def parse_args():
    p = argparse.ArgumentParser(description='LSTF-for-Lightning')

    # Config file vs CLI
    p.add_argument('--config', type=str, default=None, help='YAML config')
    p.add_argument('--list_models', action='store_true', help='List available models')

    # Data
    p.add_argument('--cell_index', default='Data/processed/dataset/cell_index.csv')
    p.add_argument('--dataset_dir', default='Data/processed/dataset')
    p.add_argument('--n_features', type=int, default=11)
    p.add_argument('--L', type=int, default=13)
    p.add_argument('--H', type=int, default=12)
    p.add_argument('--max_train_samples', type=int, default=200000)
    p.add_argument('--max_val_samples', type=int, default=40000)
    p.add_argument('--max_test_samples', type=int, default=40000)
    p.add_argument('--num_workers', type=int, default=0)

    # Model
    p.add_argument('--model', type=str, default='lstm')
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--d_ff', type=int, default=None)  # TSLib FFN dim

    # Training
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_min', type=float, default=1e-6)
    p.add_argument('--pos_weight', type=float, default=10.0)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--lr_scheduler', default='cosine')
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--focal_gamma', type=float, default=2.0,
                   help='Focal Loss gamma: higher = more focus on hard examples')
    p.add_argument('--threshold', type=float, default=0.1,
                   help='Decision threshold for CSI/POD/FAR metrics')

    # Misc
    p.add_argument('--save_dir', default='checkpoints/run')
    p.add_argument('--resume', default=None)
    p.add_argument('--test_only', action='store_true')
    p.add_argument('--force_cpu', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--oversample_pos', type=int, default=1,
                   help='Oversample positive samples N times (train only)')
    p.add_argument('--spatial_patch_size', type=int, default=0,
                   help='Spatial neighborhood: 1=3x3, 2=5x5')
    p.add_argument('--grid_mode', action='store_true',
                   help='Full-grid U-Net mode (uses GridDataset)')
    p.add_argument('--grid_dataset_dir', default='Data/processed/grid_dataset',
                   help='Grid dataset directory (for --grid_mode)')
    p.add_argument('--crop_size', type=int, default=128,
                   help='Random crop size for grid training')

    args = p.parse_args()
    defaults = {a.dest: a.default for a in p._actions}
    return args, defaults


# ===== Main =====

def main():
    args, defaults = parse_args()

    # --list_models
    if args.list_models:
        print('Available models:')
        for name in list_models():
            print(f'  - {name}')
        return

    # Merge config: YAML → CLI overrides
    if args.config:
        from config import load_config
        cfg = load_config(args.config)
        for k, v in vars(args).items():
            if k != 'config' and v != defaults.get(k):
                cfg[k] = v
    else:
        cfg = vars(args)

    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))

    model_name = cfg['model']
    n_features = cfg.get('n_features', 11)
    L, H = cfg.get('L', 13), cfg.get('H', 12)   # 13 forecast steps → 12 pred steps
    force_cpu = cfg.get('force_cpu', False)
    save_dir = cfg.get('save_dir', 'checkpoints/run')

    print('=' * 50)
    print(f'Model: {model_name}    13 forecast steps → 12h prediction')
    print(f'Device: {"CUDA" if torch.cuda.is_available() and not force_cpu else "CPU"}')
    print(f'Save: {save_dir}')
    print('=' * 50)

    # Data
    print('[Data] Loading...')
    grid_mode = cfg.get('grid_mode', False)

    if grid_mode:
        from grid_dataset import create_grid_dataloaders as create_grid_loaders
        train_loader, val_loader, test_loader = create_grid_loaders(
            grid_dir=cfg.get('grid_dataset_dir', 'Data/processed/grid_dataset'),
            batch_size=cfg.get('batch_size', 256),
            crop_size=cfg.get('crop_size', 128),
            num_workers=cfg.get('num_workers', 0),
        )
        n_features = 149  # 13×11 forecast + 6 past lightning
    else:
        spatial_ps = cfg.get('spatial_patch_size', 0)
        train_loader, val_loader, test_loader = create_dataloaders(
            cell_index_path=cfg.get('cell_index'),
            dataset_dir=cfg.get('dataset_dir'),
            batch_size=cfg.get('batch_size', 256),
            max_train_samples=cfg.get('max_train_samples', 100000),
            max_val_samples=cfg.get('max_val_samples', 20000),
            max_test_samples=cfg.get('max_test_samples', 20000),
            num_workers=cfg.get('num_workers', 0),
            pin_memory=torch.cuda.is_available() and not force_cpu,
            oversample_pos=cfg.get('oversample_pos', 1),
            spatial_patch_size=spatial_ps,
        )

        # Adjust n_features for spatial patches
        if spatial_ps > 0:
            n_patch = (2 * spatial_ps + 1) ** 2
            n_features = n_features * n_patch
            print(f'[Data] Spatial patch {spatial_ps}  n_features: {cfg.get("n_features", 11)} -> {n_features}')

    # Model — filter config to only model-relevant kwargs
    print(f'[Model] Creating {model_name}...')
    _exclude = {'n_features', 'L', 'H', 'model', 'config', 'cell_index', 'dataset_dir',
                'save_dir', 'resume', 'test_only', 'force_cpu', 'seed', 'list_models',
                'max_train_samples', 'max_val_samples', 'max_test_samples', 'num_workers',
                'batch_size', 'epochs', 'log_interval', 'oversample_pos', 'spatial_patch_size',
                'grid_mode', 'grid_dataset_dir', 'crop_size', 'threshold'}
    model_kwargs = {k: v for k, v in cfg.items() if k not in _exclude}
    model = create_model(model_name, n_features=n_features, L=L, H=H, **model_kwargs)
    print(f'         {model.describe()}')

    # Train
    trainer = Trainer(model, train_loader, val_loader, test_loader, cfg)

    if cfg.get('resume'):
        trainer.load(cfg['resume'])

    if cfg.get('test_only'):
        trainer.test()
    else:
        trainer.train()

    print('Done!')


if __name__ == '__main__':
    main()
