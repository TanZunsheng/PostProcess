#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证 BrainOmni 后处理输出的脚本

检查输出的 .pt 文件是否符合 BrainOmni 编码器的要求。

使用方法：
    python test_output.py /path/to/output_dir
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np


def validate_pt_file(file_path: Path) -> dict:
    """
    验证单个 .pt 文件
    """
    result = {
        'file': str(file_path),
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    try:
        data = torch.load(file_path)
        
        # 检查必需的键
        required_keys = ['x', 'pos', 'sensor_type']
        for key in required_keys:
            if key not in data:
                result['errors'].append(f"缺少必需的键: {key}")
                result['valid'] = False
        
        if not result['valid']:
            return result
        
        x = data['x']
        pos = data['pos']
        sensor_type = data['sensor_type']
        
        # 检查形状
        n_channels = x.shape[0]
        n_samples = x.shape[1]
        
        result['stats']['n_channels'] = n_channels
        result['stats']['n_samples'] = n_samples
        result['stats']['duration_sec'] = n_samples / 256
        
        # 检查 pos 形状
        if pos.shape != (n_channels, 6):
            result['errors'].append(f"pos 形状错误: 期望 ({n_channels}, 6), 实际 {pos.shape}")
            result['valid'] = False
        
        # 检查 sensor_type 形状
        if sensor_type.shape != (n_channels,):
            result['errors'].append(f"sensor_type 形状错误: 期望 ({n_channels},), 实际 {sensor_type.shape}")
            result['valid'] = False
        
        # 检查信号统计
        x_mean = x.mean().item()
        x_std = x.std().item()
        result['stats']['x_mean'] = x_mean
        result['stats']['x_std'] = x_std
        
        if abs(x_mean) > 0.1:
            result['warnings'].append(f"信号均值偏离 0: {x_mean:.4f}")
        
        if abs(x_std - 1.0) > 0.5:
            result['warnings'].append(f"信号标准差偏离 1: {x_std:.4f}")
        
        # 检查坐标范围
        pos_xyz = pos[:, :3]
        pos_min = pos_xyz.min().item()
        pos_max = pos_xyz.max().item()
        result['stats']['pos_min'] = pos_min
        result['stats']['pos_max'] = pos_max
        
        if pos_min < -2 or pos_max > 2:
            result['warnings'].append(f"坐标可能未归一化: [{pos_min:.3f}, {pos_max:.3f}]")
        
        # 检查 NaN
        if torch.isnan(x).any():
            result['errors'].append("信号包含 NaN")
            result['valid'] = False
        
        if torch.isnan(pos).any():
            result['errors'].append("坐标包含 NaN")
            result['valid'] = False
        
    except Exception as e:
        result['errors'].append(f"加载文件失败: {e}")
        result['valid'] = False
    
    return result


def main():
    parser = argparse.ArgumentParser(description='验证 BrainOmni 后处理输出')
    parser.add_argument('output_dir', type=str, help='输出目录')
    parser.add_argument('--limit', '-l', type=int, default=10, help='检查的文件数量限制')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"错误: 目录不存在: {output_dir}")
        sys.exit(1)
    
    # 查找 .pt 文件
    pt_files = list(output_dir.rglob('*_data.pt'))
    print(f"找到 {len(pt_files)} 个 .pt 文件")
    
    if len(pt_files) == 0:
        print("没有找到任何 .pt 文件")
        sys.exit(1)
    
    # 检查前 N 个文件
    files_to_check = pt_files[:args.limit]
    print(f"检查前 {len(files_to_check)} 个文件...")
    print("=" * 60)
    
    valid_count = 0
    warning_count = 0
    error_count = 0
    
    for f in files_to_check:
        result = validate_pt_file(f)
        
        status = "✅" if result['valid'] else "❌"
        if result['warnings']:
            status = "⚠️"
            warning_count += 1
        
        if result['valid']:
            valid_count += 1
        else:
            error_count += 1
        
        if args.verbose or not result['valid'] or result['warnings']:
            print(f"\n{status} {f.name}")
            
            if result['stats']:
                stats = result['stats']
                print(f"   形状: ({stats.get('n_channels', '?')}, {stats.get('n_samples', '?')})")
                print(f"   时长: {stats.get('duration_sec', '?'):.1f} 秒")
                print(f"   均值: {stats.get('x_mean', '?'):.6f}")
                print(f"   标准差: {stats.get('x_std', '?'):.6f}")
                print(f"   坐标范围: [{stats.get('pos_min', '?'):.3f}, {stats.get('pos_max', '?'):.3f}]")
            
            for err in result['errors']:
                print(f"   ❌ {err}")
            
            for warn in result['warnings']:
                print(f"   ⚠️ {warn}")
    
    print("\n" + "=" * 60)
    print("验证结果汇总:")
    print(f"  ✅ 有效: {valid_count}")
    print(f"  ⚠️ 警告: {warning_count}")
    print(f"  ❌ 错误: {error_count}")
    
    # 显示一个样例
    if valid_count > 0:
        print("\n" + "=" * 60)
        print("样例数据加载代码:")
        print("=" * 60)
        sample_file = pt_files[0]
        print(f"""
import torch

# 加载数据
data = torch.load('{sample_file}')

print(f"信号形状: {{data['x'].shape}}")      # (C, T)
print(f"坐标形状: {{data['pos'].shape}}")    # (C, 6)
print(f"类型: {{data['sensor_type']}}")       # (C,)
""")


if __name__ == '__main__':
    main()
