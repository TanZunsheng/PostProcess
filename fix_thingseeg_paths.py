"""
将 ThingsEEG 的输出路径从
  PENCIData/ThingsEEG/sub-*/...
迁移到
  PENCIData/ThingsEEG/derivatives/preprocessing/sub-*/...
并同步更新 JSON 元数据文件中的路径。
"""

import os
import re
import shutil
from pathlib import Path

THINGSEEG_DIR = Path("/work/2024/tanzunsheng/PENCIData/ThingsEEG")
METADATA_DIR  = Path("/work/2024/tanzunsheng/PENCIData/ThingsEEG-metadata")

TARGET_DIR = THINGSEEG_DIR / "derivatives" / "preprocessing"

OLD_PREFIX = "/work/2024/tanzunsheng/PENCIData/ThingsEEG/sub-"
NEW_PREFIX = "/work/2024/tanzunsheng/PENCIData/ThingsEEG/derivatives/preprocessing/sub-"


def get_subject_names_from_json():
    """从 JSON 中提取所有 sub-* 目录名，避免扫描大目录"""
    subjects = set()
    # 用正则从 JSON 文本直接提取，不解析整个 JSON（文件太大）
    pattern = re.compile(r'/PENCIData/ThingsEEG/(sub-[^/]+)/')
    for jf in METADATA_DIR.glob("*.json"):
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                for m in pattern.finditer(line):
                    subjects.add(m.group(1))
        if subjects:
            break  # 一个文件就够了
    return sorted(subjects)


def move_subject_dirs():
    print("=" * 60)
    print("步骤 1/2：移动 sub-* 目录")
    print("=" * 60)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"已创建目标目录: {TARGET_DIR}")

    subjects = get_subject_names_from_json()
    print(f"从 JSON 中提取到 {len(subjects)} 个 sub-* 目录: {subjects}")

    moved = 0
    skipped = 0
    for name in subjects:
        src = THINGSEEG_DIR / name
        dst = TARGET_DIR / name
        if dst.exists():
            print(f"  跳过（目标已存在）: {name}")
            skipped += 1
            continue
        if not src.exists():
            print(f"  警告（源不存在，跳过）: {src}")
            skipped += 1
            continue
        print(f"  移动: {name} ...", end="", flush=True)
        shutil.move(str(src), str(dst))
        print(" 完成")
        moved += 1

    print(f"\n移动完成：{moved} 个已移动，{skipped} 个已跳过")


def update_json_files():
    print("\n" + "=" * 60)
    print("步骤 2/2：更新 JSON 元数据路径")
    print("=" * 60)

    json_files = list(METADATA_DIR.glob("*.json"))
    # 同时更新数据目录下的 processing_metadata.json
    pm = THINGSEEG_DIR / "processing_metadata.json"
    if pm.exists():
        json_files.append(pm)

    for jf in json_files:
        print(f"\n  处理: {jf.name} ... ", end="", flush=True)
        text = jf.read_text(encoding="utf-8")
        count = text.count(OLD_PREFIX)
        if count == 0:
            print("无需修改")
            continue
        new_text = text.replace(OLD_PREFIX, NEW_PREFIX)
        jf.write_text(new_text, encoding="utf-8")
        print(f"替换了 {count} 处路径")

    print("\nJSON 更新完成")


if __name__ == "__main__":
    move_subject_dirs()
    update_json_files()
    print("\n全部完成！")
