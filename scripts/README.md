# 处理脚本使用说明

## 📁 脚本目录结构

所有处理脚本统一存放在 `scripts/` 文件夹中。

## 📊 可用脚本列表

### HBN_EEG 数据集
```bash
bash scripts/run_hbn_eeg.sh
```
- 大型数据集，128通道
- 已修复坏道插值bug

### SEED-DV 数据集
```bash
bash scripts/run_seed_dv.sh
```
- 61通道（不含Cz参考电极）
- 已验证

### Brennan_Hale2019 数据集
```bash
bash scripts/run_brennan_hale2019.sh
```
- 单一数据集，28个文件

### Broderick2018 数据集（4个子数据集）

**方式1：分别运行**（推荐，可并行）
```bash
# CocktailParty 子数据集
bash scripts/run_broderick2018_cocktailparty.sh

# NaturalSpeech 子数据集
bash scripts/run_broderick2018_naturalspeech.sh

# NaturalSpeechReverse 子数据集
bash scripts/run_broderick2018_naturalspeechreverse.sh

# SpeechInNoise 子数据集
bash scripts/run_broderick2018_speechinnoise.sh
```

**方式2：一键运行全部**（已弃用，使用方式1）
```bash
bash scripts/run_broderick2018.sh  # 循环处理4个子数据集
```

## ⚙️ 统一参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| TIME_WINDOW | 10秒 | 时间窗口 |
| STRIDE | 5秒 | 滑动步长 |
| SFREQ | 256Hz | 目标采样率 |
| WORKERS | 8 | 并行进程数 |

## 📂 输出目录

所有数据集输出到对应目录：
- HBN_EEG → `/work/2024/tanzunsheng/PENCIData/HBN_EEG/`
- SEED-DV → `/work/2024/tanzunsheng/PENCIData/SEED-DV/`
- Brennan_Hale2019 → `/work/2024/tanzunsheng/PENCIData/Brennan_Hale2019/`
- **Broderick2018** → `/work/2024/tanzunsheng/PENCIData/Broderick2018/`
  - 4个子数据集都输出到此目录下的各自子文件夹

## 💡 并行运行建议

可以在不同tmux会话中同时运行多个脚本：

```bash
# Session 1
tmux new -s broderick_cp
bash scripts/run_broderick2018_cocktailparty.sh

# Session 2
tmux new -s broderick_ns
bash scripts/run_broderick2018_naturalspeech.sh

# Session 3
tmux new -s broderick_nsr
bash scripts/run_broderick2018_naturalspeechreverse.sh

# Session 4
tmux new -s broderick_sin
bash scripts/run_broderick2018_speechinnoise.sh
```

**建议**：同时运行2-3个脚本，避免过度负载NFS。

## 🔍 监控运行状态

```bash
# 查看日志
tail -f log/Broderick2018_*.log

# 检查tmux会话
tmux ls

# 进入tmux会话
tmux attach -t <session-name>
```

## ✅ 验证输出

处理完成后检查：
```bash
# 查看生成的文件
find /work/2024/tanzunsheng/PENCIData/Broderick2018 -name "*.pt" | wc -l

# 查看各子数据集
ls -lh /work/2024/tanzunsheng/PENCIData/Broderick2018/
```
