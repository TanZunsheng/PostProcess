#!/bin/bash
# BrainOmniPostProcess 处理 SEED-DV 数据集
# 将预处理后的 BIDS 数据转换为 BrainOmni 训练格式

set -e

# 环境配置
export CONDA_ENV="EEG"

# 路径配置
SCRIPT_DIR="/work/2024/tanzunsheng/Code/BrainOmniPostProcess"
INPUT_DIR="/work/2024/tanzunsheng/ProcessedData"
OUTPUT_DIR="/work/2024/tanzunsheng/PENCIData"
DATASET="SEED-DV"

# 参数配置
TIME_WINDOW=10        # 时间窗口（秒）
STRIDE=5              # 滑动步长（秒）
SFREQ=256             # 目标采样率（Hz）
WORKERS=16             # 并行进程数

# 激活环境
echo "激活 Conda 环境: $CONDA_ENV"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV

# 显示配置信息
echo "=========================================="
echo "BrainOmni 后处理 - SEED-DV 数据集"
echo "=========================================="
echo "输入目录: $INPUT_DIR/$DATASET"
echo "输出目录: $OUTPUT_DIR/$DATASET"
echo "时间窗口: ${TIME_WINDOW}s"
echo "滑动步长: ${STRIDE}s"
echo "采样率: ${SFREQ}Hz"
echo "并行进程: $WORKERS"
echo "=========================================="
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
    echo "=========================================="
    echo "✓ BrainOmni 后处理完成！"
    echo "=========================================="
    echo "输出位置: $OUTPUT_DIR"
    echo ""
    echo "生成的文件："
    echo "  - 数据: $OUTPUT_DIR/$DATASET/**/*_data.pt"
    echo "  - 元数据: $OUTPUT_DIR/metadata/{train,val,test}.json"
    echo "  - 日志: $OUTPUT_DIR/processing_metadata.json"
else
    echo ""
    echo "=========================================="
    echo "✗ BrainOmni 后处理失败（退出码: $EXIT_CODE）"
    echo "=========================================="
fi

exit $EXIT_CODE
