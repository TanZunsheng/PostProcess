#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速生成 BrainOmni 训练元数据（不读取所有文件内容）

策略：
1. 扫描所有 .pt 文件路径
2. 按目录分组，每个目录只采样读取 1 个文件获取通道数
3. 假设同一目录下的文件通道数相同（来自同一录音）
4. 速度提升 100-1000 倍
"""

import sys
import json
import time
import random
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_metadata_fast(
    output_dir: Path,
    train_ratio: float = 0.97,
    val_ratio: float = 0.03,
    test_ratio: float = 0.00,
    seed: int = 42
):
    """
    快速生成元数据（只读取少量样本文件）
    """
    random.seed(seed)
    
    logger.info("=" * 60)
    logger.info("快速生成 BrainOmni 训练元数据")
    logger.info("=" * 60)
    
    # 1. 扫描所有 .pt 文件
    logger.info("扫描 .pt 文件...")
    start_time = time.time()
    pt_files = list(output_dir.rglob('*_data.pt'))
    total_files = len(pt_files)
    scan_time = time.time() - start_time
    logger.info(f"找到 {total_files:,} 个 .pt 文件 (耗时 {scan_time:.1f}s)")
    
    if total_files == 0:
        logger.warning("未找到任何 .pt 文件")
        return
    
    # 2. 按父目录分组（同一目录下的文件来自同一录音，通道数相同）
    logger.info("按目录分组...")
    dir_files = defaultdict(list)
    for pt_file in pt_files:
        dir_files[pt_file.parent].append(pt_file)
    
    n_dirs = len(dir_files)
    logger.info(f"共 {n_dirs:,} 个目录（每个目录采样 1 个文件获取通道数）")
    
    # 3. 每个目录采样一个文件获取通道数
    logger.info("采样读取通道数...")
    dir_channels = {}
    failed_dirs = []
    
    for i, (dir_path, files) in enumerate(dir_files.items()):
        if (i + 1) % 1000 == 0 or i == 0:
            logger.info(f"采样进度: {i+1:,}/{n_dirs:,} ({(i+1)/n_dirs*100:.1f}%)")
        
        # 随机选一个文件采样
        sample_file = random.choice(files)
        try:
            data = torch.load(sample_file, weights_only=True)
            n_channels = data['x'].shape[0]
            sensor_type = data['sensor_type'].numpy()
            is_eeg = bool((sensor_type == 0).all())
            is_meg = not is_eeg
            dir_channels[dir_path] = {
                'channels': n_channels,
                'is_eeg': is_eeg,
                'is_meg': is_meg
            }
        except Exception as e:
            logger.warning(f"采样失败 {sample_file}: {e}")
            failed_dirs.append(dir_path)
    
    logger.info(f"采样完成: 成功 {len(dir_channels):,} / 失败 {len(failed_dirs)}")
    
    # 4. 构建完整元数据列表
    logger.info("构建元数据列表...")
    metadata_list = []
    
    for dir_path, files in dir_files.items():
        if dir_path not in dir_channels:
            continue
        
        ch_info = dir_channels[dir_path]
        
        # 提取数据集名称（从目录路径）
        parts = dir_path.parts
        try:
            idx = parts.index(output_dir.name)
            dataset_name = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        except ValueError:
            dataset_name = "unknown"
        
        for pt_file in files:
            metadata_list.append({
                "dataset": dataset_name,
                "path": str(pt_file),
                "channels": ch_info['channels'],
                "is_eeg": ch_info['is_eeg'],
                "is_meg": ch_info['is_meg']
            })
    
    total_success = len(metadata_list)
    logger.info(f"元数据总数: {total_success:,}")
    
    # 5. 按数据集分组
    datasets = defaultdict(list)
    for m in metadata_list:
        datasets[m["dataset"]].append(m)
    
    # 6. 划分数据集
    logger.info(f"划分数据集 (train={train_ratio*100:.0f}%, val={val_ratio*100:.0f}%, test={test_ratio*100:.0f}%)...")
    train_list = []
    val_list = []
    test_list = []
    
    for ds_name, ds_data in datasets.items():
        # 按通道数分组
        channel_groups = defaultdict(list)
        for m in ds_data:
            channel_groups[m["channels"]].append(m)
        
        for ch, items in channel_groups.items():
            random.shuffle(items)
            n = len(items)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            
            train_list.extend(items[:n_train])
            val_list.extend(items[n_train:n_train + n_val])
            if test_ratio > 0:
                test_list.extend(items[n_train + n_val:])
    
    random.shuffle(train_list)
    random.shuffle(val_list)
    if test_ratio > 0:
        random.shuffle(test_list)
    
    # 7. 保存元数据
    logger.info("保存元数据文件...")
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [("train", train_list), ("val", val_list)]
    if test_ratio > 0:
        splits.append(("test", test_list))
    
    for name, data_list in splits:
        filepath = metadata_dir / f"{name}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False)  # 不用 indent，节省空间
        logger.info(f"  ✓ {name}.json: {len(data_list):,} 条记录")
    
    # 8. 统计
    logger.info("=" * 60)
    logger.info("元数据统计")
    logger.info("=" * 60)
    logger.info(f"训练集: {len(train_list):,} ({len(train_list)/total_success*100:.1f}%)")
    logger.info(f"验证集: {len(val_list):,} ({len(val_list)/total_success*100:.1f}%)")
    if test_ratio > 0:
        logger.info(f"测试集: {len(test_list):,} ({len(test_list)/total_success*100:.1f}%)")
    logger.info(f"数据集数量: {len(datasets)}")
    logger.info("-" * 60)
    for ds_name, ds_data in datasets.items():
        channels = set(m["channels"] for m in ds_data)
        logger.info(f"  {ds_name}: {len(ds_data):,} 片段, 通道数: {sorted(channels)}")
    logger.info("=" * 60)
    
    total_time = time.time() - start_time
    logger.info(f"总耗时: {total_time:.1f} 秒")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='快速生成 BrainOmni 训练元数据')
    parser.add_argument('output_dir', type=str, help='BrainOmniData 目录路径')
    parser.add_argument('--train_ratio', type=float, default=0.97, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.03, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.00, help='测试集比例')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        print(f"错误: 目录不存在: {output_dir}")
        sys.exit(1)
    
    generate_metadata_fast(
        output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
