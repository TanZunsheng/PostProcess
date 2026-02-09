#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将现有的 float32 格式 PT 文件原地转换为 bfloat16

优势：
1. 节省 50% 磁盘空间
2. 加载速度更快
3. 与 BrainOmni 训练格式一致

使用方法：
    # 先试运行（只统计，不转换）
    python convert_to_bfloat16.py /path/to/BrainOmniData --dry-run
    
    # 实际转换
    python convert_to_bfloat16.py /path/to/BrainOmniData --workers 16
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool
from typing import Dict

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_single_file(args_tuple):
    """
    转换单个 PT 文件
    
    返回: (success, filepath, original_size, new_size, error)
    """
    pt_file, dry_run = args_tuple
    
    try:
        # 记录原始文件大小
        original_size = pt_file.stat().st_size
        
        if dry_run:
            # 干运行模式：只检查不转换
            return True, str(pt_file), original_size, original_size // 2, None
        
        # 加载数据
        data = torch.load(pt_file, weights_only=True)
        
        # 检查当前数据类型
        current_dtype_x = data['x'].dtype
        current_dtype_pos = data['pos'].dtype
        
        # 如果已经是 bfloat16，跳过
        if current_dtype_x == torch.bfloat16 and current_dtype_pos == torch.bfloat16:
            return True, str(pt_file), original_size, original_size, None
        
        # 转换数据类型
        data['x'] = data['x'].to(torch.bfloat16)
        data['pos'] = data['pos'].to(torch.bfloat16)
        # sensor_type 保持 int32/int64 不变
        
        # 保存（覆盖原文件）
        torch.save(data, pt_file)
        
        # 获取新文件大小
        new_size = pt_file.stat().st_size
        
        return True, str(pt_file), original_size, new_size, None
        
    except Exception as e:
        return False, str(pt_file), 0, 0, str(e)


def format_size(bytes_size):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def convert_dataset(
    data_dir: Path,
    workers: int = 8,
    dry_run: bool = False
):
    """
    批量转换数据集
    """
    logger.info("=" * 60)
    if dry_run:
        logger.info("干运行模式（只统计，不转换）")
    else:
        logger.info("开始转换 PT 文件: float32 -> bfloat16")
    logger.info("=" * 60)
    
    # 1. 扫描所有 PT 文件
    logger.info(f"扫描目录: {data_dir}")
    start_time = time.time()
    pt_files = list(data_dir.rglob('*_data.pt'))
    scan_time = time.time() - start_time
    
    total_files = len(pt_files)
    logger.info(f"找到 {total_files:,} 个 PT 文件 (耗时 {scan_time:.1f}s)")
    
    if total_files == 0:
        logger.warning("未找到任何 PT 文件")
        return
    
    # 2. 并行转换
    logger.info(f"使用 {workers} 个进程并行处理...")
    logger.info("-" * 60)
    
    completed = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0
    total_original_size = 0
    total_new_size = 0
    
    start_time = time.time()
    
    # 准备参数
    args_list = [(f, dry_run) for f in pt_files]
    
    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(convert_single_file, args_list, chunksize=100):
            success, filepath, orig_size, new_size, error = result
            completed += 1
            
            if success:
                total_original_size += orig_size
                total_new_size += new_size
                
                if orig_size == new_size and not dry_run:
                    skipped_count += 1  # 已经是 bfloat16
                else:
                    success_count += 1
            else:
                failed_count += 1
                logger.error(f"转换失败: {filepath}: {error}")
            
            # 更新进度 - 每个文件都更新
            elapsed = time.time() - start_time
            rate = completed / elapsed if elapsed > 0 else 0
            remaining = (total_files - completed) / rate if rate > 0 else 0
            progress = completed / total_files * 100
            
            saved_size = total_original_size - total_new_size
            saved_pct = (saved_size / total_original_size * 100) if total_original_size > 0 else 0
            
            # 使用 \r 覆盖当前行实现实时更新
            print(
                f"\r进度: {completed:,}/{total_files:,} ({progress:.1f}%) | "
                f"速度: {rate:.0f} 文件/秒 | "
                f"剩余: {remaining/60:.1f} 分钟 | "
                f"已节省: {format_size(saved_size)} ({saved_pct:.1f}%)",
                end='', flush=True
            )
    
    # 打印换行，结束进度条
    print()
    
    # 3. 统计结果
    total_time = time.time() - start_time
    saved_size = total_original_size - total_new_size
    saved_pct = (saved_size / total_original_size * 100) if total_original_size > 0 else 0
    
    logger.info("=" * 60)
    logger.info("转换完成统计")
    logger.info("=" * 60)
    logger.info(f"总文件数: {total_files:,}")
    logger.info(f"成功转换: {success_count:,}")
    if skipped_count > 0:
        logger.info(f"已是 bfloat16（跳过）: {skipped_count:,}")
    logger.info(f"失败: {failed_count:,}")
    logger.info("-" * 60)
    logger.info(f"原始大小: {format_size(total_original_size)}")
    logger.info(f"转换后大小: {format_size(total_new_size)}")
    logger.info(f"节省空间: {format_size(saved_size)} ({saved_pct:.1f}%)")
    logger.info("-" * 60)
    logger.info(f"总耗时: {total_time/60:.1f} 分钟")
    logger.info(f"平均速度: {total_files/total_time:.0f} 文件/秒")
    logger.info("=" * 60)
    
    if dry_run:
        logger.info("✓ 干运行完成，未实际修改文件")
        logger.info(f"预计可节省: {format_size(saved_size)}")
    else:
        logger.info("✓ 转换完成！")


def main():
    parser = argparse.ArgumentParser(
        description='将 float32 格式的 PT 文件转换为 bfloat16',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 先试运行，查看预计节省空间
  python convert_to_bfloat16.py /path/to/BrainOmniData --dry-run
  
  # 实际转换（推荐使用多进程）
  python convert_to_bfloat16.py /path/to/BrainOmniData --workers 16
  
注意:
  - 此脚本会直接覆盖原文件
  - 建议先用 --dry-run 测试
  - 对于 160 万文件，使用 16 个进程大约需要 2-4 小时
        """
    )
    parser.add_argument(
        'data_dir',
        type=str,
        help='BrainOmniData 目录路径'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=8,
        help='并行进程数（默认 8，建议 16-32）'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='干运行模式：只统计不转换'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"目录不存在: {data_dir}")
        sys.exit(1)
    
    # 确认操作
    if not args.dry_run:
        logger.warning("=" * 60)
        logger.warning("警告: 此操作将直接覆盖原文件！")
        logger.warning("建议先运行 --dry-run 查看预估结果")
        logger.warning("=" * 60)
        response = input("确认继续? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            logger.info("操作已取消")
            sys.exit(0)
    
    convert_dataset(
        data_dir=data_dir,
        workers=args.workers,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
