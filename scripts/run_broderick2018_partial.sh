#!/bin/bash
# Broderick2018 全部数据集处理脚本
# 依次运行全部4个子数据集
# 注意：不使用 set -e，以便某个子数据集失败后继续处理其他数据集

SCRIPT_DIR="/work/2024/tanzunsheng/Code/BrainOmniPostProcess"

echo "=========================================="
echo "Broderick2018 全部数据集处理"
echo "=========================================="
echo "将依次处理以下子数据集："
echo "  1. CocktailParty"
echo "  2. NaturalSpeech"
echo "  3. NaturalSpeechReverse"
echo "  4. SpeechInNoise"
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"

# 统计结果
TOTAL=4
SUCCESS=0
FAILED_DATASETS=()

# 1. CocktailParty
echo "=========================================="
echo "[1/3] 开始处理 CocktailParty"
echo "=========================================="
if bash scripts/run_broderick2018_cocktailparty.sh; then
    echo "✓ CocktailParty 处理完成"
    ((SUCCESS++))
else
    echo "✗ CocktailParty 处理失败"
    FAILED_DATASETS+=("CocktailParty")
fi

echo ""

# 2. NaturalSpeech
echo "=========================================="
echo "[2/3] 开始处理 NaturalSpeech"
echo "=========================================="
if bash scripts/run_broderick2018_naturalspeech.sh; then
    echo "✓ NaturalSpeech 处理完成"
    ((SUCCESS++))
else
    echo "✗ NaturalSpeech 处理失败"
    FAILED_DATASETS+=("NaturalSpeech")
fi

echo ""

# 3. NaturalSpeechReverse
echo "=========================================="
echo "[3/3] 开始处理 NaturalSpeechReverse"
echo "=========================================="
if bash scripts/run_broderick2018_naturalspeechreverse.sh; then
    echo "✓ NaturalSpeechReverse 处理完成"
    ((SUCCESS++))
else
    echo "✗ NaturalSpeechReverse 处理失败"
    FAILED_DATASETS+=("NaturalSpeechReverse")
fi

echo ""

# 4. SpeechInNoise
echo "=========================================="
echo "[4/4] 开始处理 SpeechInNoise"
echo "=========================================="
if bash scripts/run_broderick2018_speechinnoise.sh; then
    echo "✓ SpeechInNoise 处理完成"
    ((SUCCESS++))
else
    echo "✗ SpeechInNoise 处理失败"
    FAILED_DATASETS+=("SpeechInNoise")
fi

echo ""

# 汇总结果
echo "=========================================="
echo "处理完成汇总"
echo "=========================================="
echo "总计: $TOTAL 个子数据集"
echo "成功: $SUCCESS 个"
echo "失败: $((TOTAL - SUCCESS)) 个"

if [ ${#FAILED_DATASETS[@]} -gt 0 ]; then
    echo ""
    echo "失败的子数据集:"
    for FAILED in "${FAILED_DATASETS[@]}"; do
        echo "  - $FAILED"
    done
    echo ""
    echo "✗ 部分数据集处理失败"
    exit 1
else
    echo ""
    echo "✓ 所有数据集处理成功！"
    echo "输出位置: /work/2024/tanzunsheng/PENCIData/Broderick2018"
    exit 0
fi
