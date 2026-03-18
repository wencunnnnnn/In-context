"""
生成 masks_merged 文件夹：将同一图像的多个子 mask 合并为一张。

命名规律：
  masks_merged/{id}--{prefix}_merged.png
  = masks/{id}--{prefix}_000.png | masks/{id}--{prefix}_001.png | ...

优化：先一次性列出 masks/ 目录建立索引，避免每次 glob 搜索。
"""

import json
import os
import re
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

DATA_ROOT = "/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1"
MASKS_DIR = os.path.join(DATA_ROOT, "masks")
MERGED_DIR = os.path.join(DATA_ROOT, "masks_merged")
GROUNDING_JSON = "/media/userdisk2/zhli/data/MeCoVQA/test/MeCoVQA_Grounding_test.json"

def main():
    os.makedirs(MERGED_DIR, exist_ok=True)

    # 1. 从 JSON 中提取所有 masks_merged 路径
    with open(GROUNDING_JSON, "r") as f:
        data = json.load(f)

    merged_paths = []
    for item in data:
        for conv in item.get("conversations", []):
            val = conv.get("value", "")
            m = re.search(r"<mask>(masks_merged/.*?)</mask>", val)
            if m:
                merged_paths.append(m.group(1))

    print(f"需要生成 {len(merged_paths)} 个 merged mask")

    # 2. 一次性列出 masks/ 目录，按前缀分组建立索引
    # 前缀定义：文件名中最后一个 _ 之前的部分（如 xxx--0000_000.png -> xxx--0000）
    print("正在索引 masks/ 目录...")
    prefix_to_files = defaultdict(list)
    for fname in os.listdir(MASKS_DIR):
        if not fname.endswith(".png"):
            continue
        # xxx--0000_000.png -> 前缀 xxx--0000
        base = fname[:-4]  # 去掉 .png
        last_underscore = base.rfind("_")
        if last_underscore != -1:
            prefix = base[:last_underscore]
            prefix_to_files[prefix].append(fname)
    print(f"索引完成，共 {len(prefix_to_files)} 个前缀组")

    # 3. 逐个生成 merged mask
    success = 0
    skipped = 0

    for merged_rel in tqdm(merged_paths):
        merged_abs = os.path.join(DATA_ROOT, merged_rel)
        if os.path.exists(merged_abs):
            success += 1
            continue

        # 从 masks_merged/xxx--0000_merged.png 提取前缀 xxx--0000
        basename = os.path.basename(merged_rel)  # xxx--0000_merged.png
        prefix = basename.replace("_merged.png", "")  # xxx--0000

        # 从索引中查找子 mask
        sub_fnames = sorted(prefix_to_files.get(prefix, []))

        if not sub_fnames:
            print(f"  [SKIP] 找不到子 mask，前缀: {prefix}")
            skipped += 1
            continue

        # 合并：取并集
        merged = None
        for fname in sub_fnames:
            sub_path = os.path.join(MASKS_DIR, fname)
            mask = np.array(Image.open(sub_path).convert("L"))
            if merged is None:
                merged = mask.copy()
            else:
                merged = np.maximum(merged, mask)

        # 保存
        Image.fromarray(merged).save(merged_abs)
        success += 1

    print(f"\n完成！成功: {success}, 跳过: {skipped}, 总计: {len(merged_paths)}")


if __name__ == "__main__":
    main()
