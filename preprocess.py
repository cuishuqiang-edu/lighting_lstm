"""
数据预处理：扫描 GRAPES + ADTD → 对齐到统一网格.

输出 (Data/processed/):
  lightning_grid.npy   (T, 204, 330)  逐小时雷击计数
  lats.npy / lons.npy                  (204, 330) 经纬度网格
  file_index.pkl       {valid_dt: filepath}   valid_time → 最短预报时效文件
  init_index.pkl       {init_dt: [F000_path .. F012_path]}  完备起报索引
"""

import numpy as np
import pandas as pd
import os, glob, pickle
from datetime import datetime, timedelta
from tqdm import tqdm

# 所有路径相对于项目根目录，Windows/Linux 通用
GRAPES_DIR = os.path.join('Data', 'CMA-2025')
CSV_PATH = os.path.join('Data', '2025.csv')
OUTPUT_DIR = os.path.join('Data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_VARS = [
    'TMP_2m', 'RH_2m', 'PRES_surface', 'PRMSL',
    'UGRD_10m', 'VGRD_10m', 'DCAPE', 'CREF',
    'UPHL', 'APCP', 'MAX_VERTICAL_WIND_SHEAR_SPEED',
]


def scan_grapes():
    """
    扫描所有 NPZ，建立两个索引：
      file_index: {valid_dt: shortest_fh_path}  用于 lightning 对齐
      init_index: {init_dt: [F000..F012_path]}  用于预测样本
    """
    file_index = {}     # valid_datetime → (path, fh)
    init_records = []   # (init_dt, fh, path)

    date_dirs = sorted(glob.glob(os.path.join(GRAPES_DIR, '20*')))
    print(f'找到 {len(date_dirs)} 个日期目录')

    for date_dir in tqdm(date_dirs, desc='扫描'):
        for init_dir in sorted(glob.glob(os.path.join(date_dir, '20*'))):
            if not os.path.isdir(init_dir):
                continue
            try:
                init_dt = datetime.strptime(os.path.basename(init_dir),
                                            '%Y%m%d%H%M%S')
            except ValueError:
                continue

            for npz_path in glob.glob(os.path.join(init_dir, '*.npz')):
                fname = os.path.basename(npz_path)
                parts = fname.replace('.npz', '').split('_')
                fh = int(parts[-1][1:])
                valid_dt = init_dt + timedelta(hours=fh)

                # file_index: keep shortest forecast hour per valid_time
                if valid_dt not in file_index or fh < file_index[valid_dt][1]:
                    file_index[valid_dt] = (npz_path, fh)
                init_records.append((init_dt, fh, npz_path))

    # Build init_index: only keep init_times with complete F000-F012
    init_df = pd.DataFrame(init_records, columns=['init_dt', 'fh', 'path'])
    init_index = {}
    for init_dt, grp in init_df.groupby('init_dt'):
        by_fh = grp.set_index('fh')['path']
        if all(fh in by_fh.index for fh in range(0, 13)):
            init_index[init_dt] = [by_fh[fh] for fh in range(0, 13)]

    # Sort file_index by valid_time
    sorted_file = {}
    for dt in sorted(file_index):
        sorted_file[dt] = file_index[dt][0]

    print(f'  valid_times: {len(sorted_file)}')
    print(f'  init_times (F000-F012 完备): {len(init_index)}')
    return sorted_file, init_index


def load_lightning_labels():
    """读取 ADTD CSV，过滤异常电流."""
    df = pd.read_csv(CSV_PATH)
    df['datetime'] = pd.to_datetime(df['Datetime'], format='%Y-%m-%d-%H:%M:%S')
    df = df[df['Lit_Current'].abs() < 500]
    print(f'  闪电记录: {len(df):,} 条')
    print(f'  时间范围: {df["datetime"].min()} → {df["datetime"].max()}')
    return df


def aggregate_lightning(df, lats, lons, hour_bins):
    """将 ADTD 散点聚合到 (T, 204, 330) 网格."""
    n_lat, n_lon = lats.shape
    T = len(hour_bins)
    grid = np.zeros((T, n_lat, n_lon), dtype=np.float32)

    lat_edges = np.linspace(lats.min(), lats.max(), n_lat + 1)
    lon_edges = np.linspace(lons.min(), lons.max(), n_lon + 1)

    df['hour_bin'] = pd.cut(df['datetime'], bins=hour_bins, labels=False)
    df = df.dropna(subset=['hour_bin']).astype({'hour_bin': int})

    for bin_idx in tqdm(range(T), desc='聚合雷电'):
        mask = df['hour_bin'] == bin_idx
        if mask.sum() == 0:
            continue
        batch = df[mask]
        lat_idx = np.searchsorted(lat_edges, batch['Lat'].values) - 1
        lon_idx = np.searchsorted(lon_edges, batch['Lon'].values) - 1
        valid = (lat_idx >= 0) & (lat_idx < n_lat) & \
                (lon_idx >= 0) & (lon_idx < n_lon)
        for li, lj in zip(lat_idx[valid], lon_idx[valid]):
            grid[bin_idx, li, lj] += 1.0
    return grid


# ===== 主流程 =====
if __name__ == '__main__':
    print('=' * 60)
    print('Step 1: 扫描 GRAPES 文件')
    print('=' * 60)
    file_index, init_index = scan_grapes()

    print('\n' + '=' * 60)
    print('Step 2: 加载 ADTD 闪电标签')
    print('=' * 60)
    df_lightning = load_lightning_labels()

    sorted_times = sorted(file_index.keys())
    t0, t1 = sorted_times[0], sorted_times[-1]
    hour_start = t0.replace(minute=0, second=0, microsecond=0)
    hour_bins = pd.date_range(start=hour_start, end=t1 + timedelta(hours=1), freq='h')
    print(f'  时间 bins: {len(hour_bins)} 个 ({hour_start} → {t1})')

    ref_f = np.load(next(iter(file_index.values())))
    lats, lons = ref_f['lats'], ref_f['lons']
    ref_f.close()

    print('\n' + '=' * 60)
    print('Step 3: 聚合闪电到网格')
    print('=' * 60)
    lightning_grid = aggregate_lightning(df_lightning, lats, lons, hour_bins)
    print(f'  lightning_grid: {lightning_grid.shape}')
    print(f'  有闪电时次: {(lightning_grid.sum(axis=(1,2)) > 0).mean():.1%}')

    print('\n保存中间文件...')
    np.save(os.path.join(OUTPUT_DIR, 'lightning_grid.npy'), lightning_grid)
    np.save(os.path.join(OUTPUT_DIR, 'lats.npy'), lats)
    np.save(os.path.join(OUTPUT_DIR, 'lons.npy'), lons)
    with open(os.path.join(OUTPUT_DIR, 'file_index.pkl'), 'wb') as f:
        pickle.dump(file_index, f)
    with open(os.path.join(OUTPUT_DIR, 'init_index.pkl'), 'wb') as f:
        pickle.dump(init_index, f)

    # CSV 索引方便查看
    idx_df = pd.DataFrame(sorted(file_index.items()),
                          columns=['valid_time', 'filepath'])
    idx_df.to_csv(os.path.join(OUTPUT_DIR, 'file_index.csv'), index=False)
    print(f'Done!  {len(file_index)}  valid_times,  {len(init_index)}  init_times')
