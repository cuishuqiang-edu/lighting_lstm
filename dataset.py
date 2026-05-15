"""
PyTorch Dataset for init-time-aligned forecast data.

每个样本 = 一个 (格点, 起报时刻):
  X_fc:   (13, 11)      GRAPES F000-F012 的 11 维特征
  y_past: (6,)          过去 6h 真实雷击 (log1p)
  y:      (12,)         未来 12h 真实雷击 (log1p)

时间划分: 前 70% init_times → train, 15% → val, 15% → test
"""
import os, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple

# ── 数据形状常量 (与 build_dataset.py 一致) ──
FC_TIMESTEPS = 13
N_FEATURES = 11
PAST_HOURS = 6
H = 12
FLAT_DIM = FC_TIMESTEPS * N_FEATURES + PAST_HOURS  # 149


class ForecastGridDataset(Dataset):
    """
    返回 ((X_fc, y_past), y).

    DataLoader 会将其 batch 为:
      X = (X_fc_batch, y_past_batch), y = y_batch
    → model((X_fc_batch, y_past_batch)) 直接传入元组.

    Args:
        cell_index: DataFrame from cell_index.csv
        dataset_dir: cell_*.npz 所在目录
        split: 'train' | 'val' | 'test'
        n_init_total: 总起报数 (读 dataset_meta.json)
        train_ratio / val_ratio: 按时间比例划分
        max_samples: 最大样本数
        cache_cells: 缓存格点数 (-1 = 全部)
    """
    def __init__(
        self,
        cell_index: pd.DataFrame,
        dataset_dir: str,
        split: str = 'train',
        n_init_total: int = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        max_samples: int = 100000,
        cache_cells: int = -1,
        oversample_pos: int = 1,
        spatial_patch_size: int = 0,  # 0=单点, 1=3x3, 2=5x5
    ):
        assert split in ('train', 'val', 'test')
        self.dataset_dir = dataset_dir
        self.cache = {}
        self.cache_cells = cache_cells
        self.split = split
        self.spatial_patch_size = spatial_patch_size

        # 读取元数据获得总起报数
        if n_init_total is None:
            meta_path = os.path.join(dataset_dir, 'dataset_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    n_init_total = json.load(f)['n_init']
        if n_init_total is None:
            raise ValueError('n_init_total required or dataset_meta.json')

        # 时间划分
        n_train = int(n_init_total * train_ratio)
        n_val = int(n_init_total * val_ratio)
        if split == 'train':
            lo, hi = 0, n_train
        elif split == 'val':
            lo, hi = n_train, n_train + n_val
        else:
            lo, hi = n_train + n_val, n_init_total

        # 构建扁平索引: [(cell_id, npz_path, sample_start, n_avail), ...]
        self.index = []
        total = 0
        for _, row in cell_index.iterrows():
            cell_id = int(row['cell_id'])
            path = os.path.join(dataset_dir, row['file'])
            if not os.path.exists(path):
                continue
            # mmap 快速读取样本数
            f = np.load(path, mmap_mode='r')
            n_cell = f['y_future'].shape[0]
            del f
            n_avail = max(0, min(hi, n_cell) - lo)
            if n_avail > 0:
                take = min(n_avail, max_samples - total)
                if take > 0:
                    self.index.append((cell_id, path, lo, take))
                    total += take
                if total >= max_samples:
                    break

        n_cells = len(set(w[0] for w in self.index))
        print(f'  [{split}] {total} samples, {n_cells} cells')

        # === 正样本过采样 (仅训练集) ===
        if oversample_pos > 1 and split == 'train':
            positives = []
            seen_paths = {}
            for cell_id, path, lo, n in self.index:
                if path not in seen_paths:
                    seen_paths[path] = np.load(path)['y_future']
                yf = seen_paths[path][lo:lo + n]
                for local in range(n):
                    if (yf[local] > 0).any():
                        positives.append((cell_id, path, lo + local, 1))
            for p in seen_paths.values():
                del p
            n_pos = len(positives)
            self.index.extend(positives * (oversample_pos - 1))
            print(f'  [oversample] {n_pos} positives x{oversample_pos} -> '
                  f'{n_pos * oversample_pos} (+{n_pos * (oversample_pos - 1)} extra)')

        # === 空间邻域索引 (3x3 / 5x5 patch) ===
        if spatial_patch_size > 0:
            # 构建 (lat, lon) → (cell_id, path) 全量格点查找表
            self._cell_grid = {}
            for _, row in cell_index.iterrows():
                key = (int(row['lat_idx']), int(row['lon_idx']))
                self._cell_grid[key] = (
                    int(row['cell_id']),
                    os.path.join(dataset_dir, row['file']),
                )
            # 为每个在 index 里的 cell 预计算邻居列表
            self._neighbors = {}
            P = spatial_patch_size
            for cell_id, path, lo, n in self.index:
                row_ = cell_index[cell_id == cell_index['cell_id'].values].iloc[0]
                lat, lon = int(row_['lat_idx']), int(row_['lon_idx'])
                nbrs = []
                for di in range(-P, P + 1):
                    for dj in range(-P, P + 1):
                        key = (lat + di, lon + dj)
                        if key in self._cell_grid:
                            nbrs.append(self._cell_grid[key])
                        else:
                            nbrs.append(None)
                self._neighbors[cell_id] = nbrs
            n_pts = (2 * P + 1) ** 2
            print(f'  [spatial] patch {P}x{P} ({n_pts} pts, '
                  f'{N_FEATURES * n_pts} features/step)')

    def __len__(self) -> int:
        return sum(w[3] for w in self.index)

    def _load_cell(self, cell_id, path):
        if cell_id in self.cache:
            return self.cache[cell_id]
        d = np.load(path)
        x = d['X_fc'].astype(np.float32)
        p = d['y_past'].astype(np.float32)
        y = d['y_future'].astype(np.float32)
        if self.cache_cells < 0 or len(self.cache) < self.cache_cells:
            self.cache[cell_id] = (x, p, y)
        return (x, p, y)

    def __getitem__(self, idx):
        offset = 0
        for cell_id, path, lo, n in self.index:
            if idx < offset + n:
                local = idx - offset
                break
            offset += n
        else:
            raise IndexError(f'{idx} out of range')

        X_fc, y_past, y_future = self._load_cell(cell_id, path)
        row = lo + local

        # 空间邻域: 从邻居格点拼接特征
        if self.spatial_patch_size > 0 and cell_id in self._neighbors:
            feats = []
            for nbr_info in self._neighbors[cell_id]:
                if nbr_info is not None:
                    nbr_id, nbr_path = nbr_info
                    if nbr_id == cell_id:
                        feats.append(X_fc[row])  # 中心已加载
                    else:
                        nx, _, _ = self._load_cell(nbr_id, nbr_path)
                        feats.append(nx[row])
                else:
                    feats.append(np.zeros_like(X_fc[row]))
            X_fc_feat = np.concatenate(feats, axis=-1)  # (13, 11*n_pts)
        else:
            X_fc_feat = X_fc[row]  # (13, 11)

        return ((torch.from_numpy(X_fc_feat),
                 torch.from_numpy(y_past[row])),
                torch.from_numpy(y_future[row]))


def create_dataloaders(
    cell_index_path: str,
    dataset_dir: str,
    batch_size: int = 256,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    max_train_samples: int = 100000,
    max_val_samples: int = 20000,
    max_test_samples: int = 20000,
    num_workers: int = 0,
    pin_memory: bool = True,
    cache_cells: int = -1,
    oversample_pos: int = 1,
    spatial_patch_size: int = 0,  # 空间邻域 patch
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """创建 train/val/test DataLoader (时间划分)."""
    cell_index = pd.read_csv(cell_index_path)
    # 读元数据
    meta_path = os.path.join(dataset_dir, 'dataset_meta.json')
    n_init = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            n_init = json.load(f)['n_init']
    print(f'[Data] {len(cell_index)} cells, {n_init} init_times')

    common = dict(cell_index=cell_index, dataset_dir=dataset_dir,
                  n_init_total=n_init, train_ratio=train_ratio,
                  val_ratio=val_ratio, cache_cells=cache_cells,
                  spatial_patch_size=spatial_patch_size)

    train_ds = ForecastGridDataset(**common, split='train',
                                   max_samples=max_train_samples,
                                   oversample_pos=oversample_pos)
    val_ds = ForecastGridDataset(**common, split='val',
                                 max_samples=max_val_samples)
    test_ds = ForecastGridDataset(**common, split='test',
                                  max_samples=max_test_samples)

    def _loader(ds, shuffle):
        if len(ds) == 0:
            return None
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory)

    return (_loader(train_ds, True),
            _loader(val_ds, False),
            _loader(test_ds, False))


# ===== 评估指标 (不变) =====

def lightning_metrics(y_pred: torch.Tensor, y_true: torch.Tensor,
                      threshold: float = 0.5):
    """回归 + 分类指标."""
    mse = torch.mean((y_pred - y_true) ** 2).item()
    mae = torch.mean(torch.abs(y_pred - y_true)).item()

    pb = (y_pred > threshold)
    tb = (y_true > threshold)
    tp = (pb & tb).sum().item()
    fp = (pb & ~tb).sum().item()
    fn = (~pb & tb).sum().item()
    tn = (~pb & ~tb).sum().item()
    eps = 1e-8

    return {
        'mse': mse, 'mae': mae,
        'csi': tp / (tp + fn + fp + eps),
        'pod': tp / (tp + fn + eps),
        'far': fp / (tp + fp + eps),
        'hss': (2*(tp*tn - fn*fp)) / ((tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)+eps),
        'bias': (tp+fp) / (tp+fn+eps),
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
    }


if __name__ == '__main__':
    dl, vl, tl = create_dataloaders(
        cell_index_path='Data/processed/dataset/cell_index.csv',
        dataset_dir='Data/processed/dataset',
        batch_size=32,
        max_train_samples=500, max_val_samples=100, max_test_samples=100,
    )
    (X_fc, y_past), y = next(iter(dl))
    print(f'\nBatch:')
    print(f'  X_fc:   {X_fc.shape}')   # (B, 13, 11)
    print(f'  y_past: {y_past.shape}')  # (B, 6)
    print(f'  y:      {y.shape}')       # (B, 12)
    print(f'  y>0:    {(y>0).sum()}/{y.numel()} ({(y>0).float().mean():.1%})')
