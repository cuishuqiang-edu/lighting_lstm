# LSTF-for-Lightning

基于 LSTF Transformer (PatchTST/iTransformer) 的闪电中期预测 (12-24h+)。

## 项目结构

```
d:\application\claude\
├── preprocess.py            # 数据预处理 (GRPES+ADTD 对齐)
├── extract_dataset.py       # 格点时序数据集提取
├── baselines.py             # 基线模型 (Persistence/Climatology/Zero)
│
├── dataset.py               # PyTorch DataLoader (LightningGridDataset)
├── models.py                # 模型定义 (MLP/LSTM/Transformer + TSLib 封装)
├── train.py                 # 训练框架 (checkpoint/metrics/logging)
├── config.py                # 配置管理 (YAML + 预设)
│
├── configs/                 # YAML 配置文件
│   ├── mlp.yaml             # MLP 基线
│   ├── lstm.yaml            # LSTM 基线
│   ├── transformer.yaml     # Transformer 基线
│   ├── patchtst.yaml        # PatchTST (推荐)
│   └── itransformer.yaml    # iTransformer
│
├── requirements.txt         # Python 依赖
├── server_setup.sh          # 服务器环境配置脚本
│
├── Data/
│   ├── CMA-2025/            # GRAPES 3KM 预报场 (31 GB)
│   ├── 2025.csv             # ADTD 闪电标签 (~97万条)
│   └── processed/           # 预处理输出
│       ├── lightning_grid.npy    # 闪电网格 (3899×204×330)
│       ├── dataset/              # 格点时序数据集 (233 MB)
│       │   ├── cell_*.npz        # 2000 个格点
│       │   ├── cell_index.csv    # 格点索引
│       │   └── dataset_stats.json
│       └── ...
│
└── Time-Series-Library/     # TSLib (thuml/Time-Series-Library)
```

## 数据

| 数据 | 来源 | 覆盖 | 大小 |
|------|------|------|------|
| CMA-GRAPES 3KM | CMA 区域数值模式 | 2025-05-12 → 10-21, 华北 3km | 31 GB |
| ADTD 闪电定位 | CMA 地基闪电定位网 | 2025-04-01 → 10-30, 97万条 | CSV |

**特征**: 11 个大气变量 (TMP_2m, RH_2m, DCAPE, CREF, UPHL, 风切变等)
**标签**: log(1 + 闪电次数/小时), 稀疏度 ~0.3%

## 训练框架

### 本地测试 (CPU)
```bash
conda activate lstf_lightning

# LSTM 基线 (小批量快速测试)
python train.py --model lstm --epochs 3 --batch_size 32 --max_train_samples 1000

# MLP
python train.py --model mlp --epochs 5 --batch_size 64 --max_train_samples 2000
```

### 服务器训练 (CUDA)
```bash
# 使用配置文件的完整训练
python train.py --config configs/patchtst.yaml          # PatchTST
python train.py --config configs/itransformer.yaml      # iTransformer
python train.py --config configs/lstm.yaml              # LSTM 基线

# 或命令行参数
python train.py --model patchtst --epochs 50 --batch_size 128 \
    --lr 1e-4 --d_model 256 --n_layers 3 \
    --save_dir checkpoints/patchtst

# 恢复训练
python train.py --resume checkpoints/patchtst/best.pt

# 仅测试
python train.py --resume checkpoints/patchtst/best.pt --test_only
```

### 输出
```
checkpoints/{model}/
├── best.pt              # 最佳模型 (val_loss 最低)
├── latest.pt            # 最新模型
├── train.log            # 训练日志
├── test_results.json    # 测试集结果
└── ...
```

### 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| MSE/MAE | 回归误差 (log1p 空间) | 越低越好 |
| CSI | Critical Success Index | 越高越好 |
| POD | Probability of Detection | 越高越好 |
| FAR | False Alarm Ratio | 越低越好 |
| HSS | Heidke Skill Score | 越高越好 (>0 有技巧) |

## 服务器迁移

```bash
# 1. 复制代码和数据集到服务器
rsync -avz d:\application\claude\ user@server:/home/user/lstf-lightning/

# 2. 服务器上运行配置脚本
bash server_setup.sh

# 3. 开始训练
cd /home/user/lstf-lightning
python train.py --config configs/patchtst.yaml --epochs 100
```

## 实验路线

| 阶段 | 模型 | 预期耗时 (GPU) |
|------|------|----------------|
| 基线 | MLP, LSTM | ~30 min |
| 核心 | PatchTST, iTransformer | ~2-4 h |
| 扩展 | Autoformer, TimesNet | ~2-4 h |

## 引用

设计文档: [lstf-lightning-design.md](lstf-lightning-design.md)
TSLib: https://github.com/thuml/Time-Series-Library
