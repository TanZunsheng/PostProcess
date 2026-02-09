# BrainOmni 后处理工具

将已预处理的 BIDS 格式 EEG 数据转换为 BrainOmni 编码器所需的 `.pt` 格式。

## 功能

1. **6D 坐标提取** - 从 MNE info 提取 [x, y, z, dir_x, dir_y, dir_z]
2. **坐标归一化** - 将坐标缩放到 [-1, 1] 范围
3. **滑动窗口分段** - 默认 10 秒窗口，5 秒步长 (50% 重叠)
4. **传感器类型归一化** - 虚拟参考 + 整体 Z-Score
5. **`.pt` 格式输出** - 包含 `x`, `pos`, `sensor_type`
6. **🆕 训练元数据生成** - 自动生成 `train.json`/`val.json`/`test.json`

## 安装

EEG 环境已有 mne，只需安装 torch：

```bash
conda activate EEG
pip install torch
```

## 使用方法

### 处理单个数据集

```bash
python brainomni_postprocess.py \
    --input_dir /work/2024/tanzunsheng/ProcessedData \
    --output_dir /work/2024/tanzunsheng/BrainOmniData \
    --dataset SEED-DV \
    --time_window 10 \
    --stride 5 \
    --workers 4
```

### 处理所有数据集

```bash
python brainomni_postprocess.py \
    --input_dir /work/2024/tanzunsheng/ProcessedData \
    --output_dir /work/2024/tanzunsheng/BrainOmniData \
    --workers 8
```

### 测试模式（只处理前 5 个文件）

```bash
python brainomni_postprocess.py \
    --input_dir /work/2024/tanzunsheng/ProcessedData \
    --output_dir /work/2024/tanzunsheng/BrainOmniData_test \
    --dataset SEED-DV \
    --limit 5 \
    --workers 1
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input_dir` | 预处理后的 BIDS 数据根目录 | 必填 |
| `--output_dir` | 输出目录 | 必填 |
| `--dataset` | 指定处理的数据集名称 | 全部处理 |
| `--time_window` | 时间窗口 (秒) | 10 |
| `--stride` | 滑动步长 (秒) | 5 |
| `--sfreq` | 目标采样率 (Hz) | 256 |
| `--workers` | 并行进程数 | 4 |
| `--limit` | 限制文件数（测试用） | 无限制 |
| `--no_generate_metadata` | 跳过生成训练元数据 | 默认生成 |

## 输出格式

### 数据文件 (.pt)

每个 `.pt` 文件包含：

```python
{
    'x': torch.Tensor (C, T),           # 归一化信号 (通道数, 时间点), bfloat16
    'pos': torch.Tensor (C, 6),         # 归一化坐标, bfloat16
    'sensor_type': torch.Tensor (C,),   # 类型标签 {0: EEG}, int32
}
```

其中：
- `C` = 通道数
- `T` = 2560 (10秒 × 256Hz)
- 使用 `bfloat16` 格式可节省 50% 磁盘空间

### 训练元数据 (metadata/)

处理完成后会自动生成 BrainOmni 训练所需的元数据：

```
BrainOmniData/
├── metadata/
│   ├── train.json    # 训练集 (85%)
│   ├── val.json      # 验证集 (10%)
│   └── test.json     # 测试集 (5%)
└── ...
```

每个 JSON 文件格式：
```json
[
    {"dataset": "HBN_EEG", "path": "/path/to/0_data.pt", "channels": 128, "is_eeg": true, "is_meg": false},
    ...
]
```

## 验证输出

```bash
python test_output.py /path/to/output_dir
```

## 目录结构

```
BrainOmniData/
├── metadata/                     # 🆕 BrainOmni 训练元数据
│   ├── train.json
│   ├── val.json
│   └── test.json
├── SEED-DV/
│   └── sub-10/
│       └── eeg/
│           └── sub-10_task-visual_run-1_eeg/
│               ├── 0_data.pt
│               ├── 1_data.pt
│               └── ...
├── HBN_EEG/
│   └── ...
└── processing_metadata.json      # 处理过程记录
```

## 与 BrainOmni 集成

生成的数据可直接用于 BrainOmni 训练：

```python
# 在 BrainOmni 项目中，修改 constant.py 中的路径指向你的数据
PRETRAIN_METADATA_PATH = "/work/2024/tanzunsheng/BrainOmniData/metadata"

# 或在训练脚本中指定
from brainomni.pretrain_dataset import build_brain_bucket_dataloader
dataloader = build_brain_bucket_dataloader(
    mode="train",
    metadata_path="/work/2024/tanzunsheng/BrainOmniData/metadata",
    ...
)
```

