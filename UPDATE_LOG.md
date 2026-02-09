# BrainOmniPostProcess 更新说明

## ✅ 已完成的功能更新

### 1. 文件日志功能

**功能**：
- 所有运行日志自动保存到 `log/` 文件夹
- 日志文件命名格式：`{数据集名}-{时间戳}.log`
- 例如：`SEED-DV-20260130_013345.log` 或 `HBN_EEG-20260130_020130.log`

**位置**：`/work/2024/tanzunsheng/Code/BrainOmniPostProcess/log/`

**查看日志**：
```bash
# 查看最新日志
ls -lht /work/2024/tanzunsheng/Code/BrainOmniPostProcess/log/ | head

# 查看特定数据集的日志
tail -f /work/2024/tanzunsheng/Code/BrainOmniPostProcess/log/SEED-DV-*.log

# 搜索日志内容
grep "成功" /work/2024/tanzunsheng/Code/BrainOmniPostProcess/log/*.log
```

### 2. Metadata 目录自定义

**功能**：
- 使用 `--dataset` 参数时，生成 `{dataset}-metadata/` 目录
- 不指定时，生成默认 `metadata/` 目录

**示例**：
```bash
# SEED-DV 处理 → 生成 SEED-DV-metadata/
python brainomni_postprocess.py --dataset SEED-DV ...

# HBN_EEG 处理 → 生成 HBN_EEG-metadata/
python brainomni_postprocess.py --dataset HBN_EEG ...
```

### 3. HBN_EEG 处理脚本

**文件**：`run_hbn_eeg.sh`

**配置**：
- 32 个并行进程
- 输入：`/work/2024/tanzunsheng/ProcessedData/HBN_EEG`
- 输出：`/work/2024/tanzunsheng/PENCIData/HBN_EEG`
- Metadata：`/work/2024/tanzunsheng/PENCIData/HBN_EEG-metadata/`
- 日志：`log/HBN_EEG-{timestamp}.log`

**使用方法**：
```bash
cd /work/2024/tanzunsheng/Code/BrainOmniPostProcess
./run_hbn_eeg.sh
```

**或在后台运行**：
```bash
nohup ./run_hbn_eeg.sh > hbn_output.log 2>&1 &
```

**或使用 tmux**：
```bash
tmux new -s HBN
./run_hbn_eeg.sh
# Ctrl+B, D 来 detach
```

## 文件结构

```
/work/2024/tanzunsheng/Code/BrainOmniPostProcess/
├── brainomni_postprocess.py    # 主程序（已更新）
├── run_seed_dv.sh               # SEED-DV 处理脚本
├── run_hbn_eeg.sh               # HBN_EEG 处理脚本（新增）
└── log/                         # 日志文件夹（新增）
    ├── SEED-DV-20260130_013345.log
    ├── HBN_EEG-20260130_020130.log
    └── ...

/work/2024/tanzunsheng/PENCIData/
├── SEED-DV/                    # SEED-DV 数据文件
├── SEED-DV-metadata/           # SEED-DV 元数据
│   ├── train.json
│   └── val.json
├── HBN_EEG/                    # HBN_EEG 数据文件
└── HBN_EEG-metadata/           # HBN_EEG 元数据
    ├── train.json
    └── val.json
```

## 性能配置

### SEED-DV（当前运行中）
- Workers: 4
- 文件数: 147
- 预计时间: 1-2 小时

### HBN_EEG（待处理）
- Workers: 32
- 文件数: ~数千（12个子数据集）
- 预计时间: 取决于总文件数

## 监控命令

```bash
# 监控 SEED-DV 处理（tmux 中运行）
tmux attach -t SEED

# 查看日志
tail -f log/SEED-DV-*.log

# 检查输出文件数
find /work/2024/tanzunsheng/PENCIData/SEED-DV -name "*.pt" | wc -l

# 查看 metadata 统计
cat /work/2024/tanzunsheng/PENCIData/SEED-DV-metadata/train.json | grep -o '"channels":[0-9]*' | head
```

## 注意事项

1. **日志文件会持续增长**，建议定期清理旧日志
2. **32 workers 需要足够的 CPU 和内存**，确保服务器资源充足
3. **HBN_EEG 数据量大**，处理时间可能较长，建议使用 tmux 后台运行
4. **日志同时输出到控制台和文件**，便于实时监控和事后查看
