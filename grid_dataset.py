"""
Grid dataset for full-grid U-Net training.

Each sample is a full 204×330 grid with 149 channels.
Training applies random 128×128 crops; val/test use center crop.
"""
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

CROP_SIZE = 128
H_GRID, W_GRID = 204, 330


class GridDataset(Dataset):
    """Full-grid dataset with spatial cropping."""

    def __init__(self, grid_dir, split='train', crop_size=CROP_SIZE):
        assert split in ('train', 'val', 'test')

        with open(os.path.join(grid_dir, 'meta.json')) as f:
            meta = json.load(f)

        files = sorted(
            f for f in os.listdir(grid_dir)
            if f.startswith('grid_') and f.endswith('.npz')
        )
        n = len(files)
        n_train = int(n * meta.get('train_ratio', 0.7))
        n_val = int(n * meta.get('val_ratio', 0.15))

        if split == 'train':
            self.files = files[:n_train]
        elif split == 'val':
            self.files = files[n_train:n_train + n_val]
        else:
            self.files = files[n_train + n_val:]

        self.grid_dir = grid_dir
        self.split = split
        self.crop_size = crop_size

        n_total = len(self.files)
        print(f'  [GridDataset] {split}: {n_total} files, crop={crop_size}')

    def __len__(self):
        return len(self.files)

    def _random_crop(self, x, y):
        h = w = self.crop_size
        top = np.random.randint(0, H_GRID - h + 1)
        left = np.random.randint(0, W_GRID - w + 1)
        return x[:, top:top + h, left:left + w], y[:, top:top + h, left:left + w]

    def _center_crop(self, x, y):
        h = w = self.crop_size
        top = (H_GRID - h) // 2
        left = (W_GRID - w) // 2
        return x[:, top:top + h, left:left + w], y[:, top:top + h, left:left + w]

    def __getitem__(self, idx):
        path = os.path.join(self.grid_dir, self.files[idx])
        data = np.load(path)
        x, y = data['x'], data['y']

        if self.split == 'train':
            x_c, y_c = self._random_crop(x, y)
        else:
            x_c, y_c = self._center_crop(x, y)

        # .copy() to release the memmap
        return torch.from_numpy(x_c.copy()), torch.from_numpy(y_c.copy())


def create_grid_dataloaders(grid_dir, batch_size=8, crop_size=CROP_SIZE,
                            num_workers=0):
    """Create train/val/test DataLoaders for full-grid training."""
    train_ds = GridDataset(grid_dir, 'train', crop_size)
    val_ds = GridDataset(grid_dir, 'val', crop_size)
    test_ds = GridDataset(grid_dir, 'test', crop_size)

    def _loader(ds, shuffle):
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True,
        )

    return (
        _loader(train_ds, True),
        _loader(val_ds, False),
        _loader(test_ds, False),
    )
