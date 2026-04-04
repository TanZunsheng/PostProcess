#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BrainOmni 后处理脚本

将已预处理的 BIDS 格式 EEG 数据转换为 BrainOmni 编码器所需的 .pt 格式。

功能：
    1. 读取 BIDS 格式的预处理数据 (.fif, .vhdr)
    2. 提取 6D 电极坐标 (x, y, z, dir_x, dir_y, dir_z)
    3. 归一化坐标到 [-1, 1] 范围
    4. 滑动窗口分段 (10s/5s stride)
    5. 以可配置方式准备片段信号
    6. 保存为 .pt 文件

使用方法：
    conda activate EEG
    python brainomni_postprocess.py --input_dir /path/to/ProcessedData --output_dir /path/to/output

作者：自动生成
日期：2025-12-29
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import time

import numpy as np
import torch
import mne

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# BrainOmni 核心常量
# ============================================
SAMPLE_RATE = 256  # 目标采样率 (Hz)
LOW = 0.1  # 高通截止频率 (Hz) - 仅用于检查
HIGH = 45  # 低通截止频率 (Hz) - 仅用于检查
DEFAULT_TIME = 10  # 每段时长 (秒)
DEFAULT_STRIDE = 5  # 滑动步长 (秒)
DEFAULT_SEGMENT_NORMALIZATION = "none"
DEFAULT_SEGMENT_MIN_ACTIVITY_RATIO = 1e-2
DEFAULT_SEGMENT_MIN_ACTIVITY_ABS_EEG = 5e-7  # 0.5 uV，适合当前真实 EEG 数据

# 传感器类型映射
SENSOR_TYPE_DICT = {
    "EEG": 0,
    "MAG": 1,
    "GRAD": 2,
}


# ============================================
# 核心处理函数 (从 BrainOmni 移植)
# ============================================


def extract_pos_sensor_type(info) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 MNE info 对象中提取 6D 电极坐标和传感器类型

    参数：
        info: MNE info 对象

    返回：
        pos: (n_channels, 6) 数组 - [x, y, z, dir_x, dir_y, dir_z]
        sensor_type: (n_channels,) 数组 - {0: EEG, 1: MAG, 2: GRAD}
    """
    pos = []
    sensor_type = []

    for ch in info["chs"]:
        kind = int(ch["kind"])

        if kind == 2:  # EEG
            # EEG 电极：位置 + 零方向向量
            loc = ch["loc"][:3]
            if np.isnan(loc).any() or np.allclose(loc, 0):
                # 如果没有坐标，跳过警告，后续会处理
                pass
            pos.append(np.hstack([loc, np.array([0.0, 0.0, 0.0])]))
            sensor_type.append(SENSOR_TYPE_DICT["EEG"])

        elif kind == 1:  # MEG
            xyz = ch["loc"][:3]
            coil_type = str(ch["coil_type"])

            # 确定方向向量索引
            dir_idx = 3
            if "PLANAR" in coil_type:
                dir_idx = 1
            direction = ch["loc"][3 * dir_idx : 3 * (dir_idx + 1)]
            pos.append(np.hstack([xyz, direction]))

            if "MAG" in coil_type:
                sensor_type.append(SENSOR_TYPE_DICT["MAG"])
            else:
                sensor_type.append(SENSOR_TYPE_DICT["GRAD"])
        else:
            # 其他类型通道，按 EEG 处理
            loc = ch["loc"][:3]
            pos.append(np.hstack([loc, np.array([0.0, 0.0, 0.0])]))
            sensor_type.append(SENSOR_TYPE_DICT["EEG"])

    pos = np.stack(pos).astype(np.float32)
    sensor_type = np.array(sensor_type).astype(np.int32)

    return pos, sensor_type


def get_sensor_type_mask(
    sensor_type: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    获取各传感器类型的布尔掩码
    """
    eeg_mask = sensor_type == SENSOR_TYPE_DICT["EEG"]
    mag_mask = sensor_type == SENSOR_TYPE_DICT["MAG"]
    grad_mask = sensor_type == SENSOR_TYPE_DICT["GRAD"]
    meg_mask = mag_mask | grad_mask
    return eeg_mask, mag_mask, grad_mask, meg_mask


def normalize_pos(
    pos: np.ndarray, eeg_mask: np.ndarray, meg_mask: np.ndarray
) -> np.ndarray:
    """
    归一化电极坐标到 [-1, 1] 范围

    EEG 和 MEG 分别独立归一化
    """
    pos = pos.copy()

    if eeg_mask.any():
        # 中心化
        eeg_mean = np.mean(pos[eeg_mask, :3], axis=0, keepdims=True)
        pos[eeg_mask, :3] -= eeg_mean

        # 标准化
        eeg_scale = np.sqrt(3 * np.mean(np.sum(pos[eeg_mask, :3] ** 2, axis=1)))
        if eeg_scale > 1e-10:
            pos[eeg_mask, :3] /= eeg_scale

    if meg_mask.any():
        # 中心化
        meg_mean = np.mean(pos[meg_mask, :3], axis=0, keepdims=True)
        pos[meg_mask, :3] -= meg_mean

        # 标准化
        meg_scale = np.sqrt(3 * np.mean(np.sum(pos[meg_mask, :3] ** 2, axis=1)))
        if meg_scale > 1e-10:
            pos[meg_mask, :3] /= meg_scale

    return pos


def sensortype_wise_normalize(
    data: np.ndarray, eeg_mask: np.ndarray, mag_mask: np.ndarray, grad_mask: np.ndarray
) -> np.ndarray:
    """
    传感器类型级别的信号归一化

    关键设计：
    1. 虚拟参考：每个时间点减去所有通道的平均值
    2. 整体 Z-Score：使用整体标准差，保留通道间的幅度关系
    """
    result = data.copy()

    if eeg_mask.any():
        eeg_data = result[eeg_mask, :]
        # 虚拟参考（平均参考）
        eeg_mean = np.mean(eeg_data, axis=0, keepdims=True)
        eeg_data = eeg_data - eeg_mean
        # 整体标准化
        eeg_std = np.std(eeg_data) + 1e-5
        result[eeg_mask, :] = eeg_data / eeg_std

    if mag_mask.any():
        mag_data = result[mag_mask, :]
        mag_mean = np.mean(mag_data, axis=0, keepdims=True)
        mag_data = mag_data - mag_mean
        mag_std = np.std(mag_data) + 1e-13
        result[mag_mask, :] = mag_data / mag_std

    if grad_mask.any():
        grad_data = result[grad_mask, :]
        grad_mean = np.mean(grad_data, axis=0, keepdims=True)
        grad_data = grad_data - grad_mean
        grad_std = np.std(grad_data) + 1e-13
        result[grad_mask, :] = grad_data / grad_std

    return result.astype(np.float32)


def prepare_segment_signal(
    seg_data: np.ndarray,
    eeg_mask: np.ndarray,
    mag_mask: np.ndarray,
    grad_mask: np.ndarray,
    segment_normalization: str = DEFAULT_SEGMENT_NORMALIZATION,
) -> np.ndarray:
    """
    根据配置准备单个片段的传感器信号。

    `none` 保留预处理后的幅值；`sensortype_zscore` 保留旧版 BrainOmni
    的逐段归一化行为，便于兼容旧数据流。
    """
    if segment_normalization == "none":
        return seg_data.astype(np.float32, copy=True)
    if segment_normalization == "sensortype_zscore":
        return sensortype_wise_normalize(seg_data, eeg_mask, mag_mask, grad_mask)
    raise ValueError(f"不支持的 segment_normalization: {segment_normalization}")


def _compute_masked_segment_activity(
    seg_data: np.ndarray,
    mask: np.ndarray,
) -> Optional[float]:
    """
    计算单类传感器在当前片段中的活动度。

    活动度定义为“按通道去均值后的整体 RMS”，这样可以稳定识别全 0、
    常数片段和近似静默片段，而不会被整体偏置误导。
    """
    if not mask.any():
        return None
    sensor_data = seg_data[mask].astype(np.float64, copy=False)
    sensor_data = sensor_data - np.mean(sensor_data, axis=1, keepdims=True)
    return float(np.sqrt(np.mean(sensor_data**2)))


def compute_segment_sensor_activities(
    seg_data: np.ndarray,
    eeg_mask: np.ndarray,
    mag_mask: np.ndarray,
    grad_mask: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    计算片段在各传感器类型上的活动度。
    """
    return {
        "EEG": _compute_masked_segment_activity(seg_data, eeg_mask),
        "MAG": _compute_masked_segment_activity(seg_data, mag_mask),
        "GRAD": _compute_masked_segment_activity(seg_data, grad_mask),
    }


def _segment_is_low_activity(
    activities: Dict[str, Optional[float]],
    thresholds: Dict[str, float],
) -> bool:
    """
    判断片段是否在所有可用传感器组上都低于活动度阈值。

    只要任一实际存在的传感器组活动度达标，就保留该片段。
    """
    has_available_group = False
    for sensor_name, threshold in thresholds.items():
        activity = activities.get(sensor_name)
        if activity is None:
            continue
        has_available_group = True
        if activity >= threshold:
            return False
    return has_available_group


def accept_segment(seg_data: np.ndarray, pos: np.ndarray) -> bool:
    """
    检查数据段是否有效（无 NaN）
    """
    return not (np.isnan(seg_data).any() or np.isnan(pos).any())


def split_to_segments(
    data: np.ndarray,
    pos: np.ndarray,
    sensor_type: np.ndarray,
    eeg_mask: np.ndarray,
    mag_mask: np.ndarray,
    grad_mask: np.ndarray,
    sample_rate: int,
    time_window: int,
    stride: int,
    segment_normalization: str = DEFAULT_SEGMENT_NORMALIZATION,
    segment_min_activity_ratio: float = DEFAULT_SEGMENT_MIN_ACTIVITY_RATIO,
    segment_min_activity_abs_eeg: float = DEFAULT_SEGMENT_MIN_ACTIVITY_ABS_EEG,
    return_filter_stats: bool = False,
) -> Union[List[Dict], Tuple[List[Dict], Dict]]:
    """
    滑动窗口分段并准备片段信号

    参数：
        data: (n_channels, n_samples) 原始数据
        pos: (n_channels, 6) 归一化后的电极坐标
        sensor_type: (n_channels,) 传感器类型
        eeg_mask, mag_mask, grad_mask: 布尔掩码
        sample_rate: 采样率
        time_window: 时间窗口 (秒)
        stride: 滑动步长 (秒)

    返回：
        segments: 包含 x, pos, sensor_type 的字典列表
    """
    segment_candidates = []
    segment_samples = int(time_window * sample_rate)
    stride_samples = int(stride * sample_rate)
    activity_values = {"EEG": [], "MAG": [], "GRAD": []}
    total_windows = 0

    start = 0
    end = segment_samples

    while end <= data.shape[1]:
        total_windows += 1
        seg_data = data[:, start:end]

        seg_prepared = prepare_segment_signal(
            seg_data,
            eeg_mask,
            mag_mask,
            grad_mask,
            segment_normalization=segment_normalization,
        )

        if accept_segment(seg_prepared, pos):
            is_exact_zero = not np.any(seg_data)
            activities = compute_segment_sensor_activities(
                seg_data,
                eeg_mask,
                mag_mask,
                grad_mask,
            )
            if not is_exact_zero:
                for sensor_name, value in activities.items():
                    if value is not None:
                        activity_values[sensor_name].append(value)
            segment = {
                "x": torch.from_numpy(seg_prepared).to(torch.bfloat16),
                "pos": torch.from_numpy(pos.copy()).to(torch.bfloat16),
                "sensor_type": torch.from_numpy(sensor_type.copy()),
            }
            segment_candidates.append(
                {
                    "segment": segment,
                    "is_exact_zero": is_exact_zero,
                    "activities": activities,
                }
            )

        start += stride_samples
        end += stride_samples

    activity_references = {}
    activity_thresholds = {}
    for sensor_name, values in activity_values.items():
        if not values:
            continue
        reference = float(np.median(np.asarray(values, dtype=np.float64)))
        abs_threshold = (
            segment_min_activity_abs_eeg if sensor_name == "EEG" else 0.0
        )
        threshold = max(abs_threshold, segment_min_activity_ratio * reference)
        activity_references[sensor_name] = reference
        activity_thresholds[sensor_name] = threshold

    filtered_exact_zero_segments = 0
    filtered_low_activity_segments = 0
    kept_segments = []
    for candidate in segment_candidates:
        if candidate["is_exact_zero"]:
            filtered_exact_zero_segments += 1
            continue
        if activity_thresholds and _segment_is_low_activity(
            candidate["activities"], activity_thresholds
        ):
            filtered_low_activity_segments += 1
            continue
        kept_segments.append(candidate["segment"])

    filter_stats = {
        "total_windows": total_windows,
        "valid_windows": len(segment_candidates),
        "kept_segments": len(kept_segments),
        "filtered_exact_zero_segments": filtered_exact_zero_segments,
        "filtered_low_activity_segments": filtered_low_activity_segments,
        "activity_references": activity_references,
        "activity_thresholds": activity_thresholds,
    }

    if return_filter_stats:
        return kept_segments, filter_stats
    return kept_segments


# ============================================
# 文件处理函数
# ============================================


def find_bids_files(
    input_dir: Path, extensions: List[str] = [".fif", ".vhdr"]
) -> List[Path]:
    """
    递归查找 BIDS 目录下的预处理文件
    """
    files = []
    for ext in extensions:
        files.extend(input_dir.rglob(f"*{ext}"))

    # 过滤：只保留 derivatives/preprocessing 目录下的文件
    files = [f for f in files if "derivatives" in str(f) and "preprocessing" in str(f)]

    return sorted(files)


def process_single_file(
    file_path: Path,
    output_dir: Path,
    time_window: int = DEFAULT_TIME,
    stride: int = DEFAULT_STRIDE,
    target_sfreq: int = SAMPLE_RATE,
    segment_normalization: str = DEFAULT_SEGMENT_NORMALIZATION,
    segment_min_activity_ratio: float = DEFAULT_SEGMENT_MIN_ACTIVITY_RATIO,
    segment_min_activity_abs_eeg: float = DEFAULT_SEGMENT_MIN_ACTIVITY_ABS_EEG,
) -> Dict:
    """
    处理单个 BIDS 文件

    参数：
        file_path: 输入文件路径
        output_dir: 输出目录
        time_window: 时间窗口 (秒)
        stride: 滑动步长 (秒)
        target_sfreq: 目标采样率

    返回：
        处理结果信息
    """
    result = {
        "file": str(file_path),
        "success": False,
        "n_segments": 0,
        "error": None,
        "filtered_exact_zero_segments": 0,
        "filtered_low_activity_segments": 0,
    }

    # 断点续传：检查输出目录是否已存在
    try:
        # 动态找出相对于 subject 的路径（保留 derivatives/preprocessing/）
        parts = file_path.parts
        if "derivatives" in parts:
            idx = parts.index("derivatives")
            relative_path = Path(*parts[idx:])  # derivatives/preprocessing/sub-XX/...
        elif "preprocessing" in parts:
            idx = parts.index("preprocessing")
            relative_path = Path(*parts[idx:])  # preprocessing/sub-XX/...
        else:
            # 回退逻辑
            relative_path = file_path.relative_to(file_path.parents[4])

        output_subdir = output_dir / relative_path.parent / file_path.stem

        if output_subdir.exists():
            existing_pts = list(output_subdir.glob("*_data.pt"))
            if len(existing_pts) > 0:
                logger.info(
                    f"⏭️  跳过（已处理）: {file_path} | 已有 {len(existing_pts)} 个片段"
                )
                return {
                    "file": str(file_path),
                    "success": True,
                    "n_segments": len(existing_pts),
                    "skipped": True,
                    "output_dir": str(output_subdir),
                }
    except Exception as e:
        # 如果检查失败，继续正常处理
        logger.warning(f"检查已存在文件时出错: {e}，继续处理")

    try:
        logger.info(f"处理文件: {file_path}")

        # 1. 读取数据
        # 检查是否是 BIDS 格式（在 derivatives/preprocessing 目录下）
        is_bids = "derivatives" in str(file_path) and "preprocessing" in str(file_path)

        if is_bids:
            # 尝试解析 BIDS 文件名中的 subject 和 session
            fname = file_path.stem.replace("_eeg", "")
            fname_parts = fname.split("_")

            subject = None
            session = None
            for part in fname_parts:
                if part.startswith("sub-"):
                    subject = part.replace("sub-", "")
                elif part.startswith("ses-"):
                    session = part.replace("ses-", "")

            # 使用标准读取（绕过 MNE-BIDS 严格验证，如 split 文件的 acq_time 问题）
            if file_path.suffix == ".vhdr":
                raw = mne.io.read_raw_brainvision(
                    str(file_path), preload=True, verbose="ERROR"
                )
            elif file_path.suffix == ".fif":
                raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose="ERROR")
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")

            # 手动从 electrodes.tsv 加载坐标
            try:
                import pandas as pd

                # 直接从文件所在的 eeg/ 目录查找 electrodes.tsv
                # 避免 parents[N] 层级计算错误（BIDS结构深度因数据集而异）
                eeg_dir = file_path.parent  # .../sub-XX/ses-YY/eeg/ 或 .../sub-XX/eeg/

                electrodes_file = None
                # 优先匹配含 session 的命名格式（如 ThingsEEG: sub-01_ses-01_space-CapTrak_electrodes.tsv）
                # 再回落到不含 session 的格式（如 Brennan_Hale2019: sub-S11_space-CapTrak_electrodes.tsv）
                candidate_names = []
                if session:
                    candidate_names += [
                        f"sub-{subject}_ses-{session}_space-CapTrak_electrodes.tsv",
                        f"sub-{subject}_ses-{session}_electrodes.tsv",
                    ]
                candidate_names += [
                    f"sub-{subject}_space-CapTrak_electrodes.tsv",
                    f"sub-{subject}_electrodes.tsv",
                ]
                for name in candidate_names:
                    electrode_path = eeg_dir / name
                    if electrode_path.exists():
                        electrodes_file = electrode_path
                        break

                if electrodes_file:
                    df = pd.read_csv(electrodes_file, sep="\t")
                    ch_pos = {}
                    for _, row in df.iterrows():
                        ch_name = row["name"]
                        if ch_name in raw.ch_names:
                            ch_pos[ch_name] = np.array([row["x"], row["y"], row["z"]])

                    if ch_pos:
                        montage = mne.channels.make_dig_montage(
                            ch_pos=ch_pos, coord_frame="head"
                        )
                        import warnings

                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore", message="Not setting position.*misc channel"
                            )
                            raw.set_montage(montage)
                        logger.info(
                            f"  BIDS 格式，从 electrodes.tsv 成功加载 {len(ch_pos)} 个通道坐标 ({electrodes_file.name})"
                        )
                else:
                    logger.warning(f"  未找到 electrodes.tsv，已搜索目录: {eeg_dir}")
            except Exception as e:
                logger.warning(f"  尝试从 electrodes.tsv 加载坐标失败: {e}")

        else:
            # 非 BIDS 格式，使用标准读取
            if file_path.suffix == ".vhdr":
                raw = mne.io.read_raw_brainvision(
                    str(file_path), preload=True, verbose="ERROR"
                )
            elif file_path.suffix == ".fif":
                raw = mne.io.read_raw_fif(str(file_path), preload=True, verbose="ERROR")
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")

        # 2. 只选择 EEG 通道
        # 注意：预处理已经做了坏道插值(interpolate_bads)，
        # 虽然raw.info['bads']仍有通道列表，但数据已修复，应该保留
        # 不包含misc类型（如Cz参考电极），符合标准EEG建模实践
        picks = mne.pick_types(raw.info, eeg=True, meg=True, exclude=[])
        if len(picks) == 0:
            raise ValueError("没有找到 EEG/MEG 通道")
        raw.pick(picks)

        # 3. 检查并重采样（如果需要）
        current_sfreq = raw.info["sfreq"]
        if abs(current_sfreq - target_sfreq) > 0.1:
            logger.info(f"  重采样: {current_sfreq} Hz -> {target_sfreq} Hz")
            raw.resample(target_sfreq, verbose="ERROR")

        # 4. 设置 montage（如果没有有效坐标）
        # 注意：如果是通过 MNE-BIDS 读取的，electrodes.tsv 中的坐标已经加载，不要覆盖
        if not is_bids:
            # 检查是否已有有效坐标
            has_valid_coords = False
            if raw.get_montage() is not None:
                dig = raw.get_montage().get_positions()
                if dig and "ch_pos" in dig and len(dig["ch_pos"]) > 0:
                    # 检查坐标是否有效（非零非NaN）
                    coords = list(dig["ch_pos"].values())
                    if len(coords) > 0:
                        first_coord = coords[0]
                        if not (
                            np.isnan(first_coord).any() or np.allclose(first_coord, 0)
                        ):
                            has_valid_coords = True

            if not has_valid_coords:
                # 尝试不同的标准 montage
                n_channels = len(raw.ch_names)
                ch_names_lower = [ch.lower() for ch in raw.ch_names]

                # 检测电极命名系统
                if any(
                    ch.startswith("e") and ch[1:].isdigit() for ch in ch_names_lower
                ):
                    # GSN HydroCel 系统 (E1, E2, ..., E128)
                    try:
                        logger.info("  尝试设置 GSN-HydroCel-128 montage")
                        montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
                        raw.set_montage(montage, on_missing="ignore")
                        has_valid_coords = True
                    except Exception as e:
                        logger.warning(f"  GSN-HydroCel-128 montage 失败: {e}")

                if not has_valid_coords:
                    # 尝试标准 10-20 系统
                    try:
                        logger.info("  尝试设置 standard_1020 montage")
                        montage = mne.channels.make_standard_montage("standard_1020")
                        raw.set_montage(montage, on_missing="ignore")
                        has_valid_coords = True
                    except Exception as e:
                        logger.warning(f"  standard_1020 montage 失败: {e}")

                if not has_valid_coords:
                    # 生成球面均匀分布的虚拟坐标
                    logger.warning("  无法设置标准 montage，生成虚拟球面坐标")
                    n_ch = len(raw.ch_names)
                    # 创建球面上均匀分布的点
                    indices = np.arange(0, n_ch, dtype=float) + 0.5
                    phi = np.arccos(1 - 2 * indices / n_ch)
                    theta = np.pi * (1 + 5**0.5) * indices
                    x = np.cos(theta) * np.sin(phi) * 0.1
                    y = np.sin(theta) * np.sin(phi) * 0.1
                    z = np.cos(phi) * 0.1

                    # 创建 montage
                    ch_pos = {
                        raw.ch_names[i]: np.array([x[i], y[i], z[i]])
                        for i in range(n_ch)
                    }
                    montage = mne.channels.make_dig_montage(
                        ch_pos=ch_pos, coord_frame="head"
                    )
                    raw.set_montage(montage)
        else:
            logger.info("  BIDS 格式，使用 electrodes.tsv 中的坐标，跳过 montage 设置")

        # 定义需要显式排除的非EEG物理通道（与 PENCI 导联场保持一致）
        non_eeg_channels = {"Cz"}

        # 4.5 检查并移除没有有效坐标的通道（如 CB1, CB2 等不在 montage 中的通道）
        invalid_channels = []
        montage_check = raw.get_montage()

        if montage_check is not None:
            montage_pos_check = montage_check.get_positions()
            if montage_pos_check and "ch_pos" in montage_pos_check:
                # 从 montage 检查坐标
                ch_pos_dict = montage_pos_check["ch_pos"]
                for ch_name in raw.ch_names:
                    if ch_name in non_eeg_channels:
                        invalid_channels.append(ch_name)
                    elif ch_name not in ch_pos_dict:
                        invalid_channels.append(ch_name)
                    else:
                        coord = ch_pos_dict[ch_name]
                        if np.isnan(coord).any() or np.allclose(coord, 0, atol=1e-10):
                            invalid_channels.append(ch_name)
            else:
                # 回退到检查 info
                for ch in raw.info["chs"]:
                    ch_name = ch["ch_name"]
                    loc = ch["loc"][:3]
                    if (
                        ch_name in non_eeg_channels
                        or np.isnan(loc).any()
                        or np.allclose(loc, 0, atol=1e-10)
                    ):
                        if ch_name not in invalid_channels:
                            invalid_channels.append(ch_name)
        else:
            # 没有 montage，检查 info
            for ch in raw.info["chs"]:
                ch_name = ch["ch_name"]
                loc = ch["loc"][:3]
                if (
                    ch_name in non_eeg_channels
                    or np.isnan(loc).any()
                    or np.allclose(loc, 0, atol=1e-10)
                ):
                    if ch_name not in invalid_channels:
                        invalid_channels.append(ch_name)

        if invalid_channels:
            logger.warning(
                f"  移除无效坐标通道 ({len(invalid_channels)}个): {invalid_channels}"
            )
            # 保留有效通道
            valid_picks = [
                i for i, ch in enumerate(raw.ch_names) if ch not in invalid_channels
            ]
            if len(valid_picks) < 10:
                raise ValueError(f"有效通道太少 ({len(valid_picks)}个)，跳过此文件")
            raw.pick(valid_picks)

        # 5. 提取坐标和传感器类型
        pos, sensor_type = extract_pos_sensor_type(raw.info)
        eeg_mask, mag_mask, grad_mask, meg_mask = get_sensor_type_mask(sensor_type)

        # 6. 归一化坐标
        pos_normalized = normalize_pos(pos, eeg_mask, meg_mask)

        # 7. 获取数据并分段
        data = raw.get_data()
        segments, segment_filter_stats = split_to_segments(
            data,
            pos_normalized,
            sensor_type,
            eeg_mask,
            mag_mask,
            grad_mask,
            target_sfreq,
            time_window,
            stride,
            segment_normalization=segment_normalization,
            segment_min_activity_ratio=segment_min_activity_ratio,
            segment_min_activity_abs_eeg=segment_min_activity_abs_eeg,
            return_filter_stats=True,
        )

        if len(segments) == 0:
            filtered_total = (
                segment_filter_stats["filtered_exact_zero_segments"]
                + segment_filter_stats["filtered_low_activity_segments"]
            )
            if segment_filter_stats["valid_windows"] > 0 and filtered_total > 0:
                raise ValueError("所有有效片段都因全 0 / 近零规则被过滤")
            raise ValueError("数据太短，无法生成任何片段")

        filtered_total = (
            segment_filter_stats["filtered_exact_zero_segments"]
            + segment_filter_stats["filtered_low_activity_segments"]
        )
        if filtered_total > 0:
            logger.info(
                "  片段过滤: 保留 %d / %d，剔除全 0 %d，剔除近零 %d",
                segment_filter_stats["kept_segments"],
                segment_filter_stats["valid_windows"],
                segment_filter_stats["filtered_exact_zero_segments"],
                segment_filter_stats["filtered_low_activity_segments"],
            )
            threshold_text = []
            for sensor_name in ("EEG", "MAG", "GRAD"):
                ref = segment_filter_stats["activity_references"].get(sensor_name)
                thr = segment_filter_stats["activity_thresholds"].get(sensor_name)
                if ref is None or thr is None:
                    continue
                threshold_text.append(
                    f"{sensor_name}: median={ref:.3e}, threshold={thr:.3e}"
                )
            if threshold_text:
                logger.info("  活动度阈值: %s", " | ".join(threshold_text))

        # 8. 创建输出目录
        # 保持 BIDS 结构（包含 derivatives/preprocessing/）
        parts = file_path.parts
        if "derivatives" in parts:
            idx = parts.index("derivatives")
            relative_path = Path(
                *parts[idx:]
            )  # derivatives/preprocessing/sub-XX/ses-YY/eeg/...
        elif "preprocessing" in parts:
            idx = parts.index("preprocessing")
            relative_path = Path(*parts[idx:])  # preprocessing/sub-XX/eeg/...
        else:
            # 回退逻辑
            relative_path = file_path.relative_to(file_path.parents[4])

        output_subdir = output_dir / relative_path.parent / file_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        # 9. 保存片段
        for i, segment in enumerate(segments):
            output_path = output_subdir / f"{i}_data.pt"
            torch.save(segment, output_path)

        result["success"] = True
        result["n_segments"] = len(segments)
        result["output_dir"] = str(output_subdir)
        result["n_channels"] = data.shape[0]
        result["duration_sec"] = data.shape[1] / target_sfreq
        result["filtered_exact_zero_segments"] = segment_filter_stats[
            "filtered_exact_zero_segments"
        ]
        result["filtered_low_activity_segments"] = segment_filter_stats[
            "filtered_low_activity_segments"
        ]

        logger.info(f"  成功生成 {len(segments)} 个片段")

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"  处理失败: {e}")

    return result


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    time_window: int = DEFAULT_TIME,
    stride: int = DEFAULT_STRIDE,
    target_sfreq: int = SAMPLE_RATE,
    max_workers: int = 4,
    limit: Optional[int] = None,
    segment_normalization: str = DEFAULT_SEGMENT_NORMALIZATION,
    segment_min_activity_ratio: float = DEFAULT_SEGMENT_MIN_ACTIVITY_RATIO,
    segment_min_activity_abs_eeg: float = DEFAULT_SEGMENT_MIN_ACTIVITY_ABS_EEG,
) -> List[Dict]:
    """
    处理整个数据集

    参数：
        input_dir: 输入目录
        output_dir: 输出目录
        time_window: 时间窗口 (秒)
        stride: 滑动步长 (秒)
        target_sfreq: 目标采样率
        max_workers: 并行工作数
        limit: 限制处理文件数（用于测试）

    返回：
        所有文件的处理结果
    """
    logger.info(f"扫描输入目录: {input_dir}")
    files = find_bids_files(input_dir)

    if limit:
        files = files[:limit]

    logger.info(f"找到 {len(files)} 个文件")

    if len(files) == 0:
        logger.warning("未找到任何文件")
        return []

    results = []

    # 使用进程池并行处理
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_file,
                    f,
                    output_dir,
                    time_window,
                    stride,
                    target_sfreq,
                    segment_normalization,
                    segment_min_activity_ratio,
                    segment_min_activity_abs_eeg,
                ): f
                for f in files
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
    else:
        for f in files:
            result = process_single_file(
                f,
                output_dir,
                time_window,
                stride,
                target_sfreq,
                segment_normalization,
                segment_min_activity_ratio,
                segment_min_activity_abs_eeg,
            )
            results.append(result)

    return results


def generate_metadata(
    results: List[Dict],
    output_dir: Path,
    time_window: int = DEFAULT_TIME,
    stride: int = DEFAULT_STRIDE,
    target_sfreq: int = SAMPLE_RATE,
    segment_normalization: str = DEFAULT_SEGMENT_NORMALIZATION,
    segment_min_activity_ratio: float = DEFAULT_SEGMENT_MIN_ACTIVITY_RATIO,
    segment_min_activity_abs_eeg: float = DEFAULT_SEGMENT_MIN_ACTIVITY_ABS_EEG,
):
    """
    生成元数据文件
    """
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    skipped = [r for r in successful if r.get("skipped", False)]
    processed = [r for r in successful if not r.get("skipped", False)]

    metadata = {
        "total_files": len(results),
        "successful": len(successful),
        "processed": len(processed),
        "skipped": len(skipped),
        "failed": len(failed),
        "total_segments": sum(r.get("n_segments", 0) for r in successful),
        "filtered_exact_zero_segments": sum(
            r.get("filtered_exact_zero_segments", 0) for r in successful
        ),
        "filtered_low_activity_segments": sum(
            r.get("filtered_low_activity_segments", 0) for r in successful
        ),
        "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "sample_rate": target_sfreq,
            "time_window": time_window,
            "stride": stride,
            "segment_normalization": segment_normalization,
            "segment_min_activity_ratio": segment_min_activity_ratio,
            "segment_min_activity_abs_eeg": segment_min_activity_abs_eeg,
        },
        "files": results,
    }

    metadata_path = output_dir / "processing_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"元数据保存到: {metadata_path}")

    # 统计信息
    logger.info("=" * 60)
    logger.info("处理完成统计:")
    logger.info(f"  总文件数: {len(results)}")
    logger.info(f"  成功: {len(successful)}")
    logger.info(f"    - 新处理: {len(processed)}")
    logger.info(f"    - 跳过（已存在）: {len(skipped)}")
    logger.info(f"  失败: {len(failed)}")
    logger.info(f"  总片段数: {metadata['total_segments']:,}")
    logger.info(f"  剔除全 0 片段数: {metadata['filtered_exact_zero_segments']:,}")
    logger.info(f"  剔除近零片段数: {metadata['filtered_low_activity_segments']:,}")
    logger.info("=" * 60)

    if failed:
        logger.warning("失败的文件:")
        for r in failed[:10]:
            logger.warning(f"  - {r['file']}: {r['error']}")
        if len(failed) > 10:
            logger.warning(f"  ... 还有 {len(failed) - 10} 个失败文件")


def _load_pt_metadata(pt_file: Path, output_dir_name: str) -> dict:
    """
    加载单个 .pt 文件的元数据（用于并行处理）
    """
    try:
        data = torch.load(pt_file, weights_only=True)
        n_channels = data["x"].shape[0]
        sensor_type = data["sensor_type"].numpy()

        # 判断是 EEG 还是 MEG
        is_eeg = bool((sensor_type == SENSOR_TYPE_DICT["EEG"]).all())
        is_meg = bool(
            (
                (sensor_type == SENSOR_TYPE_DICT["MAG"])
                | (sensor_type == SENSOR_TYPE_DICT["GRAD"])
            ).all()
        )

        # 提取数据集名称
        parts = pt_file.parts
        try:
            # PENCIData 目录的下一级是数据集名称 (如 PENCIData/ThingsEEG/sub-01/...)
            if "PENCIData" in parts:
                idx = parts.index("PENCIData")
                dataset_name = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
            # 否则尝试查找指定的输出目录名
            elif output_dir_name in parts:
                idx = parts.index(output_dir_name)
                # 输出目录通常就是数据集本身的目录，或者下一级
                dataset_name = (
                    parts[idx]
                    if parts[idx] != "BrainOmniPostProcess"
                    else (parts[idx + 1] if idx + 1 < len(parts) else "unknown")
                )
            else:
                dataset_name = "unknown"
        except ValueError:
            dataset_name = "unknown"

        return {
            "dataset": dataset_name,
            "path": str(pt_file),
            "channels": n_channels,
            "is_eeg": is_eeg,
            "is_meg": is_meg,
        }
    except Exception as e:
        logger.warning(f"读取 {pt_file} 失败: {e}")
        return None


def generate_brainomni_metadata(
    output_dir: Path,
    dataset_name: str = None,
    train_ratio: float = 0.97,
    val_ratio: float = 0.03,
    test_ratio: float = 0.00,
    seed: int = 42,
    max_workers: int = 8,
):
    """
    生成 BrainOmni 训练所需的元数据文件 (train.json, val.json, test.json)

    参数：
        output_dir: 输出目录（包含 .pt 文件）
        dataset_name: 数据集名称，用于命名 metadata 目录（如 'SEED-DV' 生成 'SEED-DV-metadata'）
        train_ratio: 训练集比例 (默认 97%)
        val_ratio: 验证集比例 (默认 3%)
        test_ratio: 测试集比例 (默认 0%)
        seed: 随机种子
        max_workers: 并行工作进程数
    """
    import random
    from functools import partial

    random.seed(seed)

    logger.info("=" * 60)
    logger.info("生成 BrainOmni 训练元数据")
    logger.info("=" * 60)

    # 查找所有 .pt 文件
    logger.info("扫描 .pt 文件...")
    pt_files = list(output_dir.rglob("*_data.pt"))
    total_files = len(pt_files)
    logger.info(f"找到 {total_files:,} 个 .pt 文件")

    if total_files == 0:
        logger.warning("未找到任何 .pt 文件，跳过元数据生成")
        return

    # 并行加载元数据
    logger.info(f"使用 {max_workers} 个进程并行读取元数据...")
    metadata_list = []
    failed_count = 0

    # 使用 multiprocessing.Pool 替代 ProcessPoolExecutor，更高效
    from multiprocessing import Pool

    load_func = partial(_load_pt_metadata, output_dir_name=output_dir.name)

    completed = 0
    start_time = time.time()
    update_interval = 100  # 每 100 个文件更新一次进度

    with Pool(processes=max_workers) as pool:
        # 使用 imap_unordered，边处理边返回结果，避免内存爆炸
        for result in pool.imap_unordered(load_func, pt_files, chunksize=100):
            completed += 1

            if result is not None:
                metadata_list.append(result)
            else:
                failed_count += 1

            # 更新进度
            if completed % update_interval == 0 or completed == total_files:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining = (total_files - completed) / rate if rate > 0 else 0

                logger.info(
                    f"进度: {completed:,}/{total_files:,} ({completed/total_files*100:.1f}%) | "
                    f"速度: {rate:.0f} 文件/秒 | "
                    f"预计剩余: {remaining/60:.1f} 分钟"
                )

                # 动态调整更新间隔，避免日志过多
                if completed == 1000 and rate > 0:
                    # 根据速度调整更新频率：每 5 秒更新一次
                    update_interval = max(100, int(rate * 5))
                    logger.info(f"调整进度更新间隔为每 {update_interval} 个文件")

    success_count = len(metadata_list)
    logger.info(f"成功: {success_count:,} | 失败: {failed_count:,}")

    if success_count == 0:
        logger.error("没有成功读取任何元数据")
        return

    # 按数据集分组
    logger.info("按数据集和通道数分组...")
    datasets = {}
    for m in metadata_list:
        ds = m["dataset"]
        if ds not in datasets:
            datasets[ds] = []
        datasets[ds].append(m)

    # 划分数据集
    logger.info(
        f"划分数据集 (train={train_ratio*100:.0f}%, val={val_ratio*100:.0f}%, test={test_ratio*100:.0f}%)..."
    )
    train_list = []
    val_list = []
    test_list = []

    for ds_name, ds_data in datasets.items():
        # 按通道数分组
        channel_groups = {}
        for m in ds_data:
            ch = m["channels"]
            if ch not in channel_groups:
                channel_groups[ch] = []
            channel_groups[ch].append(m)

        # 对每个通道组进行划分
        for ch, items in channel_groups.items():
            random.shuffle(items)
            n = len(items)
            # 重要：不要用 int() 分别截断 train/val 导致余数被丢弃。
            # 这里让 val/test 先按比例取整，train 直接吃掉剩余，保证每组总数严格为 n。
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio) if test_ratio > 0 else 0
            n_train = n - n_val - n_test

            # 兜底：防止用户传入异常比例导致负数
            if n_train < 0:
                n_train = max(n - n_val, 0)
                n_test = max(n - n_train - n_val, 0)

            train_list.extend(items[:n_train])
            val_list.extend(items[n_train : n_train + n_val])
            if test_ratio > 0:
                test_list.extend(items[n_train + n_val :])

    # 最终打乱
    random.shuffle(train_list)
    random.shuffle(val_list)
    if test_ratio > 0:
        random.shuffle(test_list)

    # 保存元数据
    logger.info("保存元数据文件...")
    # 根据是否有 dataset_name 决定目录名
    if dataset_name:
        # 统一放到数据集目录的父目录旁边：PENCIData/{ds_name}-metadata/
        # 对 Broderick 子数据集：PENCIData/Broderick2018/{subds_name}-metadata/
        metadata_dir = output_dir.parent / f"{dataset_name}-metadata"
    else:
        metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    splits = [("train", train_list), ("val", val_list)]
    if test_ratio > 0:
        splits.append(("test", test_list))

    for name, data_list in splits:
        filepath = metadata_dir / f"{name}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        logger.info(f"  ✓ {name}.json: {len(data_list):,} 条记录")

    # 统计信息
    logger.info("=" * 60)
    logger.info("元数据统计")
    logger.info("=" * 60)
    logger.info(
        f"训练集: {len(train_list):,} ({len(train_list)/success_count*100:.1f}%)"
    )
    logger.info(f"验证集: {len(val_list):,} ({len(val_list)/success_count*100:.1f}%)")
    if test_ratio > 0:
        logger.info(
            f"测试集: {len(test_list):,} ({len(test_list)/success_count*100:.1f}%)"
        )
    logger.info(f"数据集数量: {len(datasets)}")
    logger.info("-" * 60)
    for ds_name, ds_data in datasets.items():
        channels = set(m["channels"] for m in ds_data)
        logger.info(f"  {ds_name}: {len(ds_data):,} 片段, 通道数: {sorted(channels)}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="将预处理的 BIDS 数据转换为 BrainOmni 格式"
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        required=True,
        help="输入目录 (预处理后的 BIDS 数据根目录)",
    )
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="输出目录")
    parser.add_argument(
        "--time_window",
        "-t",
        type=int,
        default=DEFAULT_TIME,
        help=f"时间窗口 (秒)，默认 {DEFAULT_TIME}",
    )
    parser.add_argument(
        "--stride",
        "-s",
        type=int,
        default=DEFAULT_STRIDE,
        help=f"滑动步长 (秒)，默认 {DEFAULT_STRIDE}",
    )
    parser.add_argument(
        "--sfreq",
        type=int,
        default=SAMPLE_RATE,
        help=f"目标采样率 (Hz)，默认 {SAMPLE_RATE}",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="并行工作进程数，默认 4"
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="限制处理文件数（用于测试）"
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="指定处理的数据集名称（子目录名）"
    )
    parser.add_argument(
        "--segment_normalization",
        type=str,
        choices=["none", "sensortype_zscore"],
        default=DEFAULT_SEGMENT_NORMALIZATION,
        help="片段信号准备方式；none 保留预处理后的幅值，sensortype_zscore 兼容旧版逐段标准化",
    )
    parser.add_argument(
        "--segment_min_activity_ratio",
        type=float,
        default=DEFAULT_SEGMENT_MIN_ACTIVITY_RATIO,
        help="近零过滤的相对活动度阈值，按单文件片段活动度中位数的比例计算，默认 0.01",
    )
    parser.add_argument(
        "--segment_min_activity_abs_eeg",
        type=float,
        default=DEFAULT_SEGMENT_MIN_ACTIVITY_ABS_EEG,
        help="近零过滤的 EEG 绝对活动度阈值（单位 V，默认 5e-7，即 0.5uV）",
    )
    parser.add_argument(
        "--no_generate_metadata",
        action="store_true",
        help="跳过生成 BrainOmni 训练元数据（默认会自动生成）",
    )

    args = parser.parse_args()

    # 配置文件日志
    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)

    # 生成日志文件名：数据集名称-时间.log
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.dataset if args.dataset else "all_datasets"
    log_filename = f"{dataset_name}-{timestamp}.log"
    log_filepath = log_dir / log_filename

    # 添加文件处理器
    file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)

    logger.info(f"日志文件: {log_filepath}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存根输出目录用于生成元数据
    root_output_dir = output_dir

    # 如果指定了数据集，只处理该数据集
    if args.dataset:
        input_dir = input_dir / args.dataset
        output_dir = output_dir / args.dataset
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BrainOmni 后处理开始")
    logger.info("=" * 60)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(
        (
            "参数: time_window=%ss, stride=%ss, sfreq=%sHz, "
            "segment_normalization=%s, segment_min_activity_ratio=%s, "
            "segment_min_activity_abs_eeg=%s"
        ),
        args.time_window,
        args.stride,
        args.sfreq,
        args.segment_normalization,
        args.segment_min_activity_ratio,
        args.segment_min_activity_abs_eeg,
    )
    logger.info("=" * 60)

    start_time = time.time()

    results = process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        time_window=args.time_window,
        stride=args.stride,
        target_sfreq=args.sfreq,
        max_workers=args.workers,
        limit=args.limit,
        segment_normalization=args.segment_normalization,
        segment_min_activity_ratio=args.segment_min_activity_ratio,
        segment_min_activity_abs_eeg=args.segment_min_activity_abs_eeg,
    )

    generate_metadata(
        results,
        output_dir,
        time_window=args.time_window,
        stride=args.stride,
        target_sfreq=args.sfreq,
        segment_normalization=args.segment_normalization,
        segment_min_activity_ratio=args.segment_min_activity_ratio,
        segment_min_activity_abs_eeg=args.segment_min_activity_abs_eeg,
    )

    # 生成 BrainOmni 训练所需的元数据
    if not args.no_generate_metadata:
        # 如果指定了数据集，传递数据集名称
        generate_brainomni_metadata(output_dir, dataset_name=args.dataset)

    elapsed = time.time() - start_time
    logger.info(f"总耗时: {elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
