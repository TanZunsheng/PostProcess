#!/bin/bash
# 一键运行三个THINGS相关数据集处理脚本
# 依次运行: Grootswagers2019, THINGS-EEG, ThingsEEG
# 注意：不使用 set -e，以便某个数据集失败后继续处理其他数据集

SCRIPT_DIR="/work/2024/tanzunsheng/Code/BrainOmniPostProcess"

echo "==========================================="
echo "THINGS系列数据集批处理"
echo "==========================================="
echo "将依次处理以下数据集："
echo "  1. Grootswagers2019"
echo "  2. THINGS-EEG"
echo "  3. ThingsEEG"
echo "==========================================="
echo ""

cd "$SCRIPT_DIR"

# 统计结果
TOTAL=3
SUCCESS=0
FAILED_DATASETS=()

# 1. Grootswagers2019
echo "==========================================="
echo "[1/3] 开始处理 Grootswagers2019"
echo "==========================================="
if bash scripts/run_grootswagers2019.sh; then
    echo "✓ Grootswagers2019 处理完成"
    ((SUCCESS++))
else
    echo "✗ Grootswagers2019 处理失败"
    FAILED_DATASETS+=("Grootswagers2019")
fi

echo ""

# 2. THINGS-EEG
echo "==========================================="
echo "[2/3] 开始处理 THINGS-EEG"
echo "==========================================="
if bash scripts/run_things_eeg.sh; then
    echo "✓ THINGS-EEG 处理完成"
    ((SUCCESS++))
else
    echo "✗ THINGS-EEG 处理失败"
    FAILED_DATASETS+=("THINGS-EEG")
fi

echo ""

# 3. ThingsEEG
echo "==========================================="
echo "[3/3] 开始处理 ThingsEEG"
echo "==========================================="
if bash scripts/run_thingseeg.sh; then
    echo "✓ ThingsEEG 处理完成"
    ((SUCCESS++))
else
    echo "✗ ThingsEEG 处理失败"
    FAILED_DATASETS+=("ThingsEEG")
fi

echo ""

# 汇总结果
echo "==========================================="
echo "处理完成汇总"
echo "==========================================="
echo "总计: $TOTAL 个数据集"
echo "成功: $SUCCESS 个"
echo "失败: $((TOTAL - SUCCESS)) 个"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "失败的数据集:"
    for FAILED in "${FAILED_DATASETS[@]}"; do
        echo "  - $FAILED"
    done
    echo ""
    echo "✗ 部分数据集处理失败"
    exit 1
else
    echo ""
    echo "✓ 所有数据集处理成功！"
    echo "输出位置: /work/2024/tanzunsheng/PENCIData"
    exit 0
fi
