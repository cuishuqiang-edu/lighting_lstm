"""
格点时序数据集持久化

从预处理中间结果 (lightning_grid.npy + GRAPES NPZ 文件) 提取候选格点的完整时序，
保存为 per-cell npz 文件，便于后续训练加载。

输出:
  processed/dataset/
    cell_{latidx}_{lonidx}.npz   各格点的 (X, y) 时序
    cell_index.csv               格点索引表
    dataset_stats.json           数据集统计信息
"""
import numpy as np
import pandas as pd
import os, json, pickle, glob
from datetime import datetime, timedelta
from tqdm import tqdm

# ===== 配置 =====
# 所有路径相对于项目根目录，Windows/Linux 通用
GRAPES_DIR = os.path.join('Data', 'CMA-2025')
PROCESSED_DIR = os.path.join('Data', 'processed')
OUTPUT_DIR = os.path.join(PROCESSED_DIR, 'dataset')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 与预处理保持一致的变量
FEATURE_VARS = [
    'TMP_2m', 'RH_2m', 'PRES_surface', 'PRMSL',
    'UGRD_10m', 'VGRD_10m', 'DCAPE', 'CREF',
    'UPHL', 'APCP', 'MAX_VERTICAL_WIND_SHEAR_SPEED',
]

MIN_FLASH_COUNT = 20    # 最少闪电次数
MAX_CELLS = 2000        # 最多保存格点数
L = 168                 # 输入长度
H = 24                  # 预测长度


def load_intermediate():
    """加载预处理中间结果"""
    print('加载中间结果...')
    lightning_grid = np.load(os.path.join(PROCESSED_DIR, 'lightning_grid.npy'))
    with open(os.path.join(PROCESSED_DIR, 'file_index.pkl'), 'rb') as f:
        file_index = pickle.load(f)
    lats = np.load(os.path.join(PROCESSED_DIR, 'lats.npy'))
    lons = np.load(os.path.join(PROCESSED_DIR, 'lons.npy'))
    print(f'  闪电网格: {lightning_grid.shape}')
    print(f'  时次数:   {len(file_index)}')
    return lightning_grid, file_index, lats, lons


def select_candidate_cells(lightning_grid):
    """
    选择候选格点：闪电总量 ≥ MIN_FLASH_COUNT 的前 MAX_CELLS 个格点
    """
    total_flashes = lightning_grid.sum(axis=0)  # (204, 330)
    mask = total_flashes >= MIN_FLASH_COUNT
    candidate_indices = np.argwhere(mask)

    # 按闪电总量排序，取前 MAX_CELLS
    cell_sums = total_flashes[mask]
    order = np.argsort(cell_sums)[::-1]  # 从多到少
    candidate_indices = candidate_indices[order]
    cell_sums = cell_sums[order]

    if len(candidate_indices) > MAX_CELLS:
        candidate_indices = candidate_indices[:MAX_CELLS]
        cell_sums = cell_sums[:MAX_CELLS]

    print(f'候选格点: {len(candidate_indices)} 个')
    print(f'  闪电范围: {cell_sums.min():.0f} ~ {cell_sums.max():.0f} 次')
    print(f'  中位数:   {np.median(cell_sums):.0f} 次')
    return candidate_indices, cell_sums


def extract_cell_time_series(file_index, candidate_indices, lightning_grid):
    """
    遍历所有 NPZ 文件，逐格点提取时序
    返回: features_dict = {(lat, lon): np.ndarray(T, n_features)}
          labels_dict  = {(lat, lon): np.ndarray(T,)}
    """
    sorted_times = sorted(file_index.keys())
    T = len(sorted_times)
    n_lat, n_lon = 204, 330
    n_features = len(FEATURE_VARS)

    # 将候选格点索引转为 set 用于快速查找
    candidate_set = set((lat, lon) for lat, lon in candidate_indices)

    # 为每个候选格点预分配列表
    cell_feature_lists = {(lat, lon): [] for lat, lon in candidate_indices}
    cell_label_lists = {(lat, lon): [] for lat, lon in candidate_indices}

    print(f'共 {T} 个时次，逐文件读取特征...')
    for t_idx, dt in enumerate(tqdm(sorted_times, desc='提取特征', unit='时次')):
        npz_path = file_index[dt]
        try:
            f = np.load(npz_path)
        except Exception as e:
            print(f'  警告: 无法加载 {npz_path}: {e}')
            continue

        # 关键优化：每个变量只解压一次，然后逐格点索引
        var_cache = {}
        for var in FEATURE_VARS:
            if var in f:
                var_cache[var] = f[var]
        f.close()

        # 对每个候选格点，提取当前时次的特征
        for (lat, lon) in candidate_indices:
            if t_idx < lightning_grid.shape[0]:
                val = lightning_grid[t_idx, lat, lon]
            else:
                val = 0.0

            vec = [var_cache[var][lat, lon] for var in FEATURE_VARS
                   if var in var_cache]
            cell_feature_lists[(lat, lon)].append(vec)
            cell_label_lists[(lat, lon)].append(val)

    # 转为 numpy 数组
    print('转换数据为 numpy 数组...')
    cell_data = {}
    for (lat, lon) in tqdm(candidate_indices, desc='转换', unit='格点'):
        X = np.array(cell_feature_lists[(lat, lon)], dtype=np.float32)
        y = np.array(cell_label_lists[(lat, lon)], dtype=np.float32)
        cell_data[(lat, lon)] = (X, y)

    return cell_data


def standardize_features(cell_data, means=None, stds=None):
    """对特征做标准化（全局统计量）"""
    if means is None:
        # 拼接所有格点的特征计算全局均值和标准差
        all_X = np.concatenate([X for X, _ in cell_data.values()], axis=0)
        means = all_X.mean(axis=0)
        stds = all_X.std(axis=0)
        stds[stds < 1e-8] = 1.0  # 避免除零
        print(f'全局特征均值: {means.round(2).tolist()}')
        print(f'全局特征标准差: {stds.round(2).tolist()}')

    for key in cell_data:
        X, y = cell_data[key]
        X = (X - means) / stds
        cell_data[key] = (X, y)

    return cell_data, means, stds


def save_dataset(cell_data, candidate_indices, cell_sums, lats, lons, means, stds):
    """保存数据集到磁盘"""
    # 1. 保存标准化参数
    norm_info = {
        'means': means.tolist() if isinstance(means, np.ndarray) else means,
        'stds': stds.tolist() if isinstance(stds, np.ndarray) else stds,
        'feature_vars': FEATURE_VARS,
    }
    with open(os.path.join(OUTPUT_DIR, 'norm_params.json'), 'w') as f:
        json.dump(norm_info, f, indent=2)

    # 2. 保存每个格点的数据
    cell_records = []
    for idx, ((lat, lon), (X, y)) in enumerate(tqdm(cell_data.items(), desc='保存格点', unit='格点')):
        fname = f'cell_{lat:03d}_{lon:03d}.npz'
        path = os.path.join(OUTPUT_DIR, fname)
        np.savez_compressed(path, X=X, y=y)
        cell_records.append({
            'cell_id': idx,
            'lat_idx': int(lat),
            'lon_idx': int(lon),
            'latitude': float(lats[lat, lon]),
            'longitude': float(lons[lat, lon]),
            'total_flashes': float(cell_sums[idx]),
            'n_timesteps': X.shape[0],
            'file': fname,
            'file_size_bytes': os.path.getsize(path),
        })

    # 3. 保存格点索引 CSV
    cell_index_df = pd.DataFrame(cell_records)
    cell_index_df.to_csv(os.path.join(OUTPUT_DIR, 'cell_index.csv'), index=False)
    print(f'\n格点索引已保存: {len(cell_index_df)} 条')

    # 4. 数据集统计
    total_size_mb = cell_index_df['file_size_bytes'].sum() / (1024 * 1024)
    stats = {
        'n_cells': len(cell_data),
        'n_features': len(FEATURE_VARS),
        'n_timesteps': X.shape[0],
        'L': L,
        'H': H,
        'total_flashes_all_cells': int(cell_sums.sum()),
        'total_size_mb': round(total_size_mb, 1),
        'feature_vars': FEATURE_VARS,
    }
    print(f'数据集大小: {total_size_mb:.1f} MB')

    # 5. 校验样本示例
    print('\n校验样本示例:')
    print(f'  特征 X:  {X.shape}  ({X.dtype})')
    print(f'  标签 y:  {y.shape}  ({y.dtype}')
    print(f'  X 范围:  {X.min():.3f} ~ {X.max():.3f}')
    print(f'  y 范围:  {y.min():.0f} ~ {y.max():.0f}')
    print(f'  y 非零占比: {(y > 0).mean():.1%}')
    lightning_times = (y > 0).sum()
    print(f'  有闪电时次: {lightning_times}/{len(y)} ({lightning_times/len(y)*100:.1f}%)')

    stats['example_cell'] = {
        'lat': int(lat), 'lon': int(lon),
        'X_shape': list(X.shape), 'y_shape': list(y.shape),
        'y_nonzero_ratio': round(float((y > 0).mean()), 4),
    }
    with open(os.path.join(OUTPUT_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    print(f'\n数据集统计已保存: {os.path.join(OUTPUT_DIR, "dataset_stats.json")}')
    return stats


def print_summary(stats, cell_index_df, lats, lons):
    """打印摘要"""
    print('\n' + '='*60)
    print(f'数据集创建完成!')
    print('='*60)
    print(f'  格点数:     {stats["n_cells"]}')
    print(f'  特征数:     {stats["n_features"]}')
    print(f'  时步数:     {stats["n_timesteps"]}')
    print(f'  总大小:     {stats["total_size_mb"]} MB')
    print(f'  输入长度:   {L} 小时')
    print(f'  预测长度:   {H} 小时')

    # 地理位置分布
    top_provinces = cell_index_df.nlargest(10, 'total_flashes')[['latitude', 'longitude', 'total_flashes']]
    print(f'\n闪电最多的 10 个格点:')
    for _, row in top_provinces.iterrows():
        print(f'  ({row["latitude"]:.2f}°N, {row["longitude"]:.2f}°E): {row["total_flashes"]:.0f} 次')

    print(f'\n下一步:')
    print(f'  1. 训练基线模型: python baselines.py')
    print(f'  2. 配置 TSLib:   使用 Time-Series-Library 的配置文件')
    print(f'  3. 训练 PatchTST/iTransformer 等模型')


if __name__ == '__main__':
    lightning_grid, file_index, lats, lons = load_intermediate()
    candidate_indices, cell_sums = select_candidate_cells(lightning_grid)

    print('\n提取格点时序...')
    cell_data = extract_cell_time_series(file_index, candidate_indices, lightning_grid)

    print('\n标准化特征...')
    cell_data, means, stds = standardize_features(cell_data)

    print('\n保存数据集...')
    stats = save_dataset(cell_data, candidate_indices, cell_sums, lats, lons, means, stds)

    cell_index_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'cell_index.csv'))
    print_summary(stats, cell_index_df, lats, lons)
