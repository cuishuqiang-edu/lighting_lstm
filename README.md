# Lightning-LSTF — 全网格 U-Net 闪电预测

基于 LightningNet-NWP 思路，从逐格点时序预测切换到全网格 2D U-Net 方案。

## 项目结构

```
├── train.py                 # 训练入口 (支持 --grid_mode)
├── trainer.py               # 训练循环 + 评估指标
├── config.py                # YAML 配置管理
│
├── build_grid_dataset.py    # 构建全网格数据集 (NPZ)
├── grid_dataset.py          # GridDataset + DataLoader
├── dataset.py               # (旧) 逐格点数据集
│
├── models/
│   ├── __init__.py           # 模型自动注册
│   ├── base.py               # BaseLightningModel 接口
│   ├── unet.py               # 2D U-Net (核心模型)
│   ├── lstm.py               # LSTM 基线
│   ├── mlp.py                # MLP 基线
│   ├── transformer.py        # Transformer 基线
│   └── tslib_wrapper.py      # TSLib 模型封装
│
├── configs/
│   ├── unet.yaml             # U-Net 配置
│   ├── lstm.yaml, mlp.yaml, ...
│
├── requirements.txt
└── README.md
```

## 全网格方案

### 数据流

```
原始 GRAPES NPZ (13 步 × 11 变量) + 闪电网格 (T×204×330)
        ↓ build_grid_dataset.py
grid_0000.npz ~ grid_1239.npz  每个文件:
  x: (149, 204, 330) = 13×11 预报 + 6h 历史闪电
  y: (12, 204, 330)  = 未来 12h 闪电 (log1p)
        ↓ GridDataset + random crop 128×128
(B, 149, 128, 128) → UNet → (B, 12, 128, 128)
```

### 模型结构

```
Encoder:  Conv(149→64) → Down(64→128→256→512)
Decoder:  Up(512→256→128→64) → Conv1×1(64→12)
参数: ~31.5M
```

### 损失函数

Focal Loss (α=10, γ=2) — 针对闪电极度稀疏 (~0.3%) 设计。

## 训练

```bash
conda activate cui_torch
cd /home/xingtao/cui_model/cc_try

python train.py --config configs/unet.yaml
# 等价于:
python train.py --grid_mode --model unet --epochs 50 --batch_size 8 --lr 1e-3 \
  --grid_dataset_dir /mnt/sda/data/cuisq/CMA-2025/Data/processed/grid_dataset/grid_dataset
```

## 服务器部署

### 数据

| 目录 | 位置 | 大小 |
|------|------|------|
| 网格数据集 | `/mnt/sda/data/cuisq/CMA-2025/Data/processed/grid_dataset/grid_dataset/` | ~27 GB (1240 NPZ) |

### 代码

GitHub: https://github.com/cuishuqiang-edu/lighting_lstm

```bash
cd /home/xingtao/cui_model/cc_try
git pull origin master
```

### 路径映射

- 代码: `/home/xingtao/cui_model/cc_try/`
- 数据: `/mnt/sda/data/cuisq/CMA-2025/Data/processed/grid_dataset/grid_dataset/`
- Python: `cui_torch` (conda)

## 评估指标

| 指标 | 说明 |
|------|------|
| MSE / MAE | 回归误差 (log1p 空间) |
| CSI | Critical Success Index (核心) |
| POD / FAR | 命中率 / 误报率 |
| HSS | Heidke Skill Score |
