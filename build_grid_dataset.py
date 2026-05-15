"""
Build per-init-time full-grid NPZ files from raw GRAPES data.

Each NPZ file = one init_time:
  x: (149, 204, 330) float16 — 143 forecast channels + 6 past lightning
  y: (12, 204, 330)  float16 — 12h future lightning (log1p)

Output: Data/processed/grid_dataset/
  meta.json           — dataset metadata
  grid_0000.npz       — {x, y}
  grid_0001.npz       — ...
  ...

Usage: python build_grid_dataset.py
"""
import numpy as np
import os, pickle, json
from tqdm import tqdm

PROCESSED_DIR = os.path.join('Data', 'processed')
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'grid_dataset')

FEATURE_VARS = [
    'TMP_2m', 'RH_2m', 'PRES_surface', 'PRMSL',
    'UGRD_10m', 'VGRD_10m', 'DCAPE', 'CREF',
    'UPHL', 'APCP', 'MAX_VERTICAL_WIND_SHEAR_SPEED',
]
N_FEATURES = len(FEATURE_VARS)  # 11
FC_STEPS = 13  # F000–F012
PAST_HOURS = 6
H = 12
IN_CHANNELS = FC_STEPS * N_FEATURES + PAST_HOURS  # 149


def load_intermediate():
    print('[load] Loading intermediate files...')
    lightning_grid = np.load(os.path.join(PROCESSED_DIR, 'lightning_grid.npy'))
    with open(os.path.join(PROCESSED_DIR, 'init_index.pkl'), 'rb') as f:
        init_index = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, 'file_index.pkl'), 'rb') as f:
        file_index = pickle.load(f)
    print(f'  lightning_grid: {lightning_grid.shape}')
    print(f'  init_times:     {len(init_index)}')
    return lightning_grid, init_index, file_index


def compute_start_idx(init_dt, hour_start_dt):
    delta = init_dt - hour_start_dt
    return int(delta.total_seconds() // 3600)


def load_fc_grids(fh_paths, n_lat=204, n_lon=330):
    """Load 13 NPZ files → (13, n_lat, n_lon, 11) float32."""
    out = np.zeros((FC_STEPS, n_lat, n_lon, N_FEATURES), dtype=np.float32)
    for fh, path in enumerate(fh_paths):
        if not os.path.exists(path):
            return None
        try:
            f = np.load(path)
            for vi, var in enumerate(FEATURE_VARS):
                if var in f:
                    out[fh, :, :, vi] = f[var]
            f.close()
        except Exception:
            return None
    return out


def build_grid_dataset(init_index, file_index, lightning_grid):
    sorted_init = sorted(init_index.keys())
    sorted_valid = sorted(file_index.keys())
    hour_start = sorted_valid[0].replace(minute=0, second=0, microsecond=0)
    T_grid = lightning_grid.shape[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n_ok = 0
    for init_dt in tqdm(sorted_init, desc='Building grid files'):
        g0 = compute_start_idx(init_dt, hour_start)
        if g0 < 0 or g0 + H > T_grid:
            continue

        grids = load_fc_grids(init_index[init_dt])
        if grids is None:
            continue

        # Forecast: (13, 204, 330, 11) → (143, 204, 330)
        x_fc = grids.reshape(FC_STEPS * N_FEATURES, 204, 330)

        # Past lightning: (6, 204, 330) log1p
        if g0 >= PAST_HOURS:
            x_past = np.log1p(lightning_grid[g0 - PAST_HOURS:g0])
        else:
            x_past = np.zeros((PAST_HOURS, 204, 330), dtype=np.float32)
            x_past[PAST_HOURS - g0:] = np.log1p(lightning_grid[:g0])

        # Future lightning: (12, 204, 330) log1p
        y = np.log1p(lightning_grid[g0:g0 + H])

        # Concatenate to (149, 204, 330)
        x = np.concatenate([x_fc, x_past], axis=0).astype(np.float32)
        y = y.astype(np.float32)

        save_path = os.path.join(OUTPUT_DIR, f'grid_{n_ok:04d}.npz')
        np.savez_compressed(save_path, x=x, y=y)
        n_ok += 1

    # Save metadata
    meta = {
        'n_files': n_ok,
        'n_init_total': len(sorted_init),
        'grid_shape': [204, 330],
        'in_channels': IN_CHANNELS,
        'H': H,
        'feature_vars': FEATURE_VARS,
        'hour_start': hour_start.isoformat(),
        'first_init': sorted_init[0].isoformat(),
        'last_init': sorted_init[-1].isoformat(),
        'train_ratio': 0.7,
        'val_ratio': 0.15,
    }
    with open(os.path.join(OUTPUT_DIR, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    est_gb = n_ok * (IN_CHANNELS + H) * 204 * 330 * 4 / (1024**3)
    print(f'\nDone! {n_ok} grid files → {OUTPUT_DIR}')
    print(f'Estimated storage: {n_ok} × ~{est_gb*1024/n_ok:.0f} MB = {est_gb:.1f} GB (float32 compressed)')
    return n_ok


if __name__ == '__main__':
    lightning_grid, init_index, file_index = load_intermediate()
    build_grid_dataset(init_index, file_index, lightning_grid)
