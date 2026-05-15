"""
Training engine — model-agnostic train/val/test loop.

Usage:
    trainer = Trainer(model, train_loader, val_loader, test_loader, config)
    trainer.train()           # full training loop
    trainer.test()            # evaluate on test set
"""
import os, json, time
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from dataset import lightning_metrics


class FocalLoss(nn.Module):
    """Focal Loss for extreme class imbalance.

    Loss = -α * (1 - pt)^γ * log(pt)
    where pt = p if y=1 else 1-p.

    Args:
        gamma: Focusing parameter. Higher = more focus on hard examples.
        alpha: Weight for positive class (same role as pos_weight).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)  # p_t = probability of correct class
        alpha_t = targets * self.alpha + (1 - targets) * 1.0
        return alpha_t * (1 - pt) ** self.gamma * bce


class Trainer:
    """
    Lightning prediction trainer.

    Args:
        model: nn.Module (any model with forward returning (B, H))
        train_loader: DataLoader
        val_loader: DataLoader
        test_loader: DataLoader or None
        config: dict of training hyperparameters
    """
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available()
                                    and not config.get('force_cpu') else 'cpu')
        self.model.to(self.device)
        print(f'[Trainer] Device: {self.device}')
        if self.device.type == 'cuda':
            print(f'          GPU: {torch.cuda.get_device_name(0)}')

        # Loss — Focal Loss (designed for extreme class imbalance)
        self.pos_weight = config.get('pos_weight', 1.0)
        focal_gamma = config.get('focal_gamma', 2.0)
        self.criterion = FocalLoss(gamma=focal_gamma, alpha=self.pos_weight)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=float(config.get('lr', 1e-3)),
            weight_decay=float(config.get('weight_decay', 1e-4)),
        )

        # LR Scheduler
        sched = config.get('lr_scheduler', 'cosine')
        if sched == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=config.get('epochs', 50),
                eta_min=config.get('lr_min', 1e-6),
            )
        elif sched == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5,
                patience=config.get('patience', 10) // 2,
            )
        else:
            self.scheduler = None

        self.clip_grad = config.get('clip_grad', 1.0)
        self.log_interval = config.get('log_interval', 50)
        self.threshold = float(config.get('threshold', 0.1))

        # Checkpoint dir
        save_dir = config.get('save_dir', 'checkpoints/run')
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_dir / 'train.log'

        # State
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.early_stop = False

        self._log(f'Config: {json.dumps(config, indent=2)}')

    def _log(self, msg):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] {msg}'
        print(line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(line + '\n')

    # ===== Checkpoint =====

    def save(self, tag='latest'):
        """Save checkpoint. tag='best' or 'latest'."""
        state = {
            'epoch': self.start_epoch - 1 if tag == 'best' else self.start_epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'config': self.config,
        }
        if isinstance(self.scheduler, CosineAnnealingLR):
            state['scheduler_state'] = self.scheduler.state_dict()

        path = self.save_dir / f'{tag}.pt'
        torch.save(state, path)
        if tag == 'best':
            self._log(f'[Save] New best model (epoch {self.best_epoch}, val_loss={self.best_val_loss:.6f})')

    def load(self, path):
        """Load checkpoint and resume."""
        path = Path(path)
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state['model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])
        self.start_epoch = state.get('epoch', 0) + 1
        self.best_val_loss = state.get('best_val_loss', float('inf'))
        self.best_epoch = state.get('best_epoch', 0)
        if self.scheduler and 'scheduler_state' in state:
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.load_state_dict(state['scheduler_state'])
        self._log(f'[Load] {path} (epoch {state.get("epoch", "?")})')
        return state

    @staticmethod
    def _to_device(x, device):
        """Move tensor/tuple-of-tensors to device."""
        if isinstance(x, torch.Tensor):
            return x.to(device)
        if isinstance(x, (tuple, list)):
            return type(x)(Trainer._to_device(v, device) for v in x)
        return x

    # ===== Training =====

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        n_batches = len(self.train_loader)
        t0 = time.time()

        for i, (X, y) in enumerate(self.train_loader):
            X, y = self._to_device(X, self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(X)  # (B, H) logits
            y_bin = (y > 0).float()   # (B, H) binary labels
            loss = self.criterion(pred, y_bin).mean()  # FocalLoss: alpha+gamma 内置
            loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            total_loss += loss.item()

            if i % self.log_interval == 0 and i > 0:
                self._log(f'  E{epoch} [{i*100//n_batches}%] loss={total_loss/(i+1):.6f}')

        return {'loss': total_loss / n_batches, 'time': time.time() - t0}

    @torch.no_grad()
    def _evaluate(self, loader, prefix='Val'):
        self.model.eval()
        total_loss = 0
        preds, targets = [], []

        for X, y in loader:
            X, y = self._to_device(X, self.device), y.to(self.device)
            y_pred = self.model(X)  # (B, H) logits
            y_bin = (y > 0).float()
            loss = self.criterion(y_pred, y_bin)
            total_loss += loss.mean().item()
            preds.append(torch.sigmoid(y_pred).cpu())
            targets.append(y.cpu())

        y_pred = torch.cat(preds)
        y_true = torch.cat(targets)
        metrics = lightning_metrics(y_pred, y_true, threshold=self.threshold)
        metrics['loss'] = total_loss / len(loader)

        # Diagnostic: prediction statistics
        prob_mean = y_pred.mean().item()
        prob_max = y_pred.max().item()
        pct_positive = (y_pred > self.threshold).float().mean().item() * 100
        metrics['pred_mean'] = prob_mean
        metrics['pred_max'] = prob_max
        metrics['pred_pct'] = pct_positive

        return metrics

    # ===== Public API =====

    def train(self):
        """Full training loop with early stopping and checkpointing."""
        epochs = self.config.get('epochs', 50)
        patience = self.config.get('patience', 10)
        self._log(f'[Train] Starting: {epochs} epochs, patience={patience}')
        self._log(f'[Train] Train samples: {len(self.train_loader.dataset)}')

        for epoch in range(self.start_epoch, epochs):
            if self.early_stop:
                break

            train_m = self._train_epoch(epoch)
            val_m = self._evaluate(self.val_loader, 'Val')

            self._log(
                f'E{epoch:3d}/{epochs} | '
                f'train_loss={train_m["loss"]:.6f} | '
                f'val_loss={val_m["loss"]:.6f} | '
                f'CSI={val_m["csi"]:.4f} POD={val_m["pod"]:.3f} '
                f'FAR={val_m["far"]:.3f} HSS={val_m["hss"]:.3f} | '
                f'pred⇧{val_m["pred_pct"]:.1f}% (max={val_m["pred_max"]:.3f} th={self.threshold}) | '
                f'{train_m["time"]:.0f}s'
            )

            # LR schedule
            if isinstance(self.scheduler, CosineAnnealingLR):
                self.scheduler.step()
            elif isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_m['loss'])

            # Checkpoint
            if val_m['loss'] < self.best_val_loss:
                self.best_val_loss = val_m['loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save('best')
            else:
                self.patience_counter += 1
                self._log(f'  No improvement {self.patience_counter}/{patience} '
                          f'(best: {self.best_val_loss:.6f} @ E{self.best_epoch})')

            if self.patience_counter >= patience:
                self._log(f'[EarlyStop] Patience {patience} exhausted.')
                self.early_stop = True

            self.save('latest')

        # Test
        self._log(f'\n[Train] Done! Best val_loss={self.best_val_loss:.6f} @ E{self.best_epoch}')
        if self.test_loader:
            self.load(self.save_dir / 'best.pt')
            self.test()

    @torch.no_grad()
    def test(self):
        """Evaluate on test set and save results."""
        if not self.test_loader:
            self._log('[Test] No test data.')
            return {}

        metrics = self._evaluate(self.test_loader, 'Test')
        self._log(
            f'[Test] loss={metrics["loss"]:.6f} '
            f'MSE={metrics["mse"]:.4f} MAE={metrics["mae"]:.4f} '
            f'CSI={metrics["csi"]:.4f} POD={metrics["pod"]:.3f} '
            f'FAR={metrics["far"]:.3f} HSS={metrics["hss"]:.3f} '
            f'th={self.threshold}'
        )

        results = {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'test': {k: float(v) if isinstance(v, (torch.Tensor, np.floating)) else v
                     for k, v in metrics.items()},
        }
        with open(self.save_dir / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results
