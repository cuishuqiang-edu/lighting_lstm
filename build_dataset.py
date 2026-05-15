"""
构建预报对齐数据集.

以每个 GRAPES 起报时刻为一个样本:
  X_fc:      (13, 11)   F000~F012 的 11 维气象特征
  y_past:    (6,)       过去 6h 真实雷击 (log1p)
  y_future:  (12,)      未来 12h 真实雷击 (log1p)

输出 per-cell NPZ: Data/processed/dataset/cell_{lat}_{lon}.npz
"""
import numpy as np
import pandas as pd
import os, sys, pickle
from datetime import datetime
from tqdm import tqdm

# 所有路径相对于项目根目录，Windows/Linux 通用
PROCESSED_DIR = os.path.join('Data', 'processed')
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_VARS = [
    'TMP_2m', 'RH_2m', 'PRES_surface', 'PRMSL',
    'UGRD_10m', 'VGRD_10m', 'DCAPE', 'CREF',
    'UPHL', 'APCP', 'MAX_VERTICAL_WIND_SHEAR_SPEED',
]

MIN_FLASH_COUNT = 20
MAX_CELLS = int(os.environ.get('MAX_CELLS', '2000'))
PAST_HOURS = 6
H = 12


def load_intermediate():
    print('加载中间结果...')
    lightning_grid = np.load(os.path.join(PROCESSED_DIR, 'lightning_grid.npy'))
    with open(os.path.join(PROCESSED_DIR, 'init_index.pkl'), 'rb') as f:
        init_index = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, 'file_index.pkl'), 'rb') as f:
        file_index = pickle.load(f)
    lats = np.load(os.path.join(PROCESSED_DIR, 'lats.npy'))
    lons = np.load(os.path.join(PROCESSED_DIR, 'lons.npy'))
    return lightning_grid, init_index, file_index, lats, lons


def select_candidates(lightning_grid):
    total = lightning_grid.sum(axis=0)
    mask = total >= MIN_FLASH_COUNT
    idxs = np.argwhere(mask)
    sums = total[mask]
    order = np.argsort(sums)[::-1]
    idxs, sums = idxs[order], sums[order]
    if len(idxs) > MAX_CELLS:
        idxs, sums = idxs[:MAX_CELLS], sums[:MAX_CELLS]
    print(f'候选格点: {len(idxs)} 个 (≥{MIN_FLASH_COUNT} 次, '
          f'{sums.min():.0f}~{sums.max():.0f})')
    return idxs, sums


def compute_start_idx(init_dt, hour_start_dt):
    delta = init_dt - hour_start_dt
    return int(delta.total_seconds() // 3600)


def load_fc_grids(fh_paths, n_lat=204, n_lon=330):
    """Load 13 NPZ files → (13, n_lat, n_lon, n_features) float32."""
    n_f = len(FEATURE_VARS)
    out = np.zeros((13, n_lat, n_lon, n_f), dtype=np.float32)
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


def extract_all_cells(init_index, file_index, lightning_grid, candidates):
    """Main extraction: per init_time, load 13 files → extract all cells."""
    sorted_init = sorted(init_index.keys())
    sorted_valid = sorted(file_index.keys())
    hour_start = sorted_valid[0].replace(minute=0, second=0, microsecond=0)
    T_grid = lightning_grid.shape[0]
    n_feat = len(FEATURE_VARS)
    n_init_all = len(sorted_init)

    # Pre-allocate per-cell
    cell_data = {}
    for lat, lon in candidates:
        key = (lat, lon)
        cell_data[key] = {
            'X_fc': np.zeros((n_init_all, 13, n_feat), dtype=np.float32),
            'y_past': np.zeros((n_init_all, PAST_HOURS), dtype=np.float32),
            'y_future': np.zeros((n_init_all, H), dtype=np.float32),
            'valid': np.zeros(n_init_all, dtype=bool),
        }

    n_ok = 0
    for i, init_dt in enumerate(tqdm(sorted_init, desc='提取按起报')):
        g0 = compute_start_idx(init_dt, hour_start)
        if g0 < 0 or g0 + H > T_grid:
            continue

        grids = load_fc_grids(init_index[init_dt])
        if grids is None:
            continue
        n_ok += 1

        # Past/future lightning slices for all cells at once
        past_start = max(0, g0 - PAST_HOURS)
        past_slice = lightning_grid[past_start:g0, :, :] if g0 > 0 \
            else np.zeros((0, 204, 330), dtype=np.float32)
        future_end = min(g0 + H, T_grid)
        future_slice = lightning_grid[g0:future_end, :, :]

        for lat, lon in candidates:
            cd = cell_data[(lat, lon)]
            cd['X_fc'][i] = grids[:, lat, lon, :]

            # Past (may be zero-padded at start of dataset)
            if g0 > 0:
                raw = past_slice[:, lat, lon]
                cd['y_past'][i, PAST_HOURS - len(raw):] = np.log1p(raw)

            # Future
            raw = future_slice[:, lat, lon]
            cd['y_future'][i, :len(raw)] = np.log1p(raw)
            cd['valid'][i] = True

    print(f'  有效起报时次: {n_ok}/{n_init_all}')
    return cell_data, sorted_init, hour_start


def save_dataset(cell_data, candidates, cell_sums, sorted_init,
                 hour_start, lats, lons):
    """Save per-cell npz + cell_index.csv + stats."""
    n_feat = len(FEATURE_VARS)
    records = []

    print('保存格点数据...')
    for idx, ((lat, lon), cd) in enumerate(
            tqdm(cell_data.items(), desc='保存')):
        # Filter to only valid init_times
        valid_mask = cd['valid']
        X_fc = cd['X_fc'][valid_mask]
        y_past = cd['y_past'][valid_mask]
        y_future = cd['y_future'][valid_mask]

        fname = f'cell_{lat:03d}_{lon:03d}.npz'
        path = os.path.join(OUTPUT_DIR, fname)
        np.savez_compressed(path,
                            X_fc=X_fc,
                            y_past=y_past,
                            y_future=y_future)

        records.append({
            'cell_id': idx,
            'lat_idx': int(lat),
            'lon_idx': int(lon),
            'latitude': float(lats[lat, lon]),
            'longitude': float(lons[lat, lon]),
            'total_flashes': float(cell_sums[idx]),
            'n_samples': X_fc.shape[0],
            'file': fname,
        })

    # cell_index.csv
    cell_df = pd.DataFrame(records)
    cell_df.to_csv(os.path.join(OUTPUT_DIR, 'cell_index.csv'), index=False)
    print(f'  格点索引: {len(cell_df)} 条')

    # Serialize init_time metadata for dataset.py
    init_meta = {
        'n_init': len(sorted_init),
        'first_init': sorted_init[0].isoformat(),
        'last_init': sorted_init[-1].isoformat(),
        'hour_start': hour_start.isoformat(),
        'past_hours': PAST_HOURS,
        'H': H,
        'n_features': n_feat,
        'feature_vars': FEATURE_VARS,
    }
    import json
    with open(os.path.join(OUTPUT_DIR, 'dataset_meta.json'), 'w') as f:
        json.dump(init_meta, f, indent=2)

    total_mb = sum(r['n_samples'] for r in records) * (
        13 * n_feat + PAST_HOURS + H) * 4 / (1024**2)
    print(f'  预估内存: {total_mb:.0f} MB (float32)')
    print(f'  总样本: {sum(r["n_samples"] for r in records):,}')
    return cell_df


if __name__ == '__main__':
    lightning_grid, init_index, file_index, lats, lons = load_intermediate()
    candidates, cell_sums = select_candidates(lightning_grid)

    print('\n提取全部格点...')
    cell_data, sorted_init, hour_start = extract_all_cells(
        init_index, file_index, lightning_grid, candidates)

    print('\n保存数据集...')
    cell_df = save_dataset(cell_data, candidates, cell_sums,
                           sorted_init, hour_start, lats, lons)

    # Sample check
    print('\n校验:')
    row = cell_df.iloc[0]
    sp = np.load(os.path.join(OUTPUT_DIR, row['file']))
    print(f'  {row["file"]}:')
    print(f'    X_fc:    {sp["X_fc"].shape}')
    print(f'    y_past:  {sp["y_past"].shape}')
    print(f'    y_future:{sp["y_future"].shape}')
    print(f'    y_future range: {sp["y_future"].min():.3f} ~ {sp["y_future"].max():.3f}')
    print(f'    y_future >0: {(sp["y_future"]>0).mean():.1%}')
    sp.close()
    print('\n完成!')
