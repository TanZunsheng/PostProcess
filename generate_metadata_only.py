#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立运行的元数据生成脚本
用于为已处理的 .pt 文件生成 BrainOmni 训练元数据
"""

import sys
import argparse
from pathlib import Path

# 导入主脚本中的函数
from brainomni_postprocess import generate_brainomni_metadata

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成 BrainOmni 训练元数据')
    parser.add_argument('output_dir', type=str, help='BrainOmniData 目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.97, help='训练集比例 (默认 0.97)')
    parser.add_argument('--val_ratio', type=float, default=0.03, help='验证集比例 (默认 0.03)')
    parser.add_argument('--test_ratio', type=float, default=0.00, help='测试集比例 (默认 0.00)')
    parser.add_argument('--workers', '-w', type=int, default=8, help='并行进程数 (默认 8)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"错误: 目录不存在: {output_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print("生成 BrainOmni 训练元数据")
    print("=" * 60)
    print(f"数据目录: {output_dir}")
    print(f"训练/验证/测试: {args.train_ratio*100:.0f}% / {args.val_ratio*100:.0f}% / {args.test_ratio*100:.0f}%")
    print(f"并行进程: {args.workers}")
    print("=" * 60)
    
    generate_brainomni_metadata(
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_workers=args.workers
    )
    
    print("\n✓ 完成! 元数据已保存到:")
    print(f"  {output_dir}/metadata/train.json")
    print(f"  {output_dir}/metadata/val.json")
    if args.test_ratio > 0:
        print(f"  {output_dir}/metadata/test.json")

