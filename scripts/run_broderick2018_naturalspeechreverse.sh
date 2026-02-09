#!/bin/bash
# Broderick2018_NaturalSpeechReverse 数据集后处理脚本

set -e

# 环境配置
export CONDA_ENV="EEG"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# 路径配置
SCRIPT_DIR="/work/2024/tanzunsheng/Code/BrainOmniPostProcess"
INPUT_DIR="/work/2024/tanzunsheng/ProcessedData/Broderick2018"  # 父目录
OUTPUT_DIR="/work/2024/tanzunsheng/PENCIData/Broderick2018"
DATASET="Broderick2018_NaturalSpeechReverse"

# 参数配置
TIME_WINDOW=10        # 时间窗口（秒）
STRIDE=5              # 滑动步长（秒）
SFREQ=256             # 目标采样率（Hz）
WORKERS=8             # 并行进程数

# 激活环境
echo "激活 Conda 环境: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# 显示配置信息
echo "==========================================="
echo "Broderick2018_NaturalSpeechReverse 数据集后处理"
echo "==========================================="
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "时间窗口: ${TIME_WINDOW}s"
echo "滑动步长: ${STRIDE}s"
echo "采样率: ${SFREQ}Hz"
echo "并行进程: $WORKERS"
echo "==========================================="
echo ""

# 运行处理
cd "$SCRIPT_DIR"
python brainomni_postprocess.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --time_window $TIME_WINDOW \
    --stride $STRIDE \
    --sfreq $SFREQ \
    --workers $WORKERS

# 检查结果
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✓ Broderick2018_NaturalSpeechReverse 处理完成！"
    echo "==========================================="
    echo "输出位置: $OUTPUT_DIR/$DATASET"
else
    echo ""
    echo "==========================================="
    echo "✗ 处理失败（退出码: $EXIT_CODE）"
    echo "==========================================="
fi

exit $EXIT_CODE
