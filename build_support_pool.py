"""
Step 1: 构建三级层次化 Support Pool

完全复用 TextICL-Seg/test-three.py 的策略：
- 通过 samed2d_index.jsonl 查 SAM 元数据，获取准确的 (dataset, modality, class)
- 从训练集构建三级 support pool
- 为测试集每个样本建立 sid → (dataset, modality, class) 映射

输出：support_pool.pkl
"""

import json
import pickle
from collections import defaultdict
from tqdm import tqdm

# 数据路径
SAM_INDEX_JSONL = "/media/userdisk2/zhli/TextICL-Seg/scripts/data/processed/samed2d_index.jsonl"
TRAIN_JSONL = "/media/userdisk2/zhli/TextICL-Seg/scripts/data/processed/mecovqa_grounding_samples.jsonl"
TEST_JSONL = "/media/userdisk2/zhli/TextICL-Seg/data/processed/mecovqa_grounding_test_samples.jsonl"
OUTPUT_PATH = "/media/userdisk2/zhli/MedPLIB/support_pool.pkl"


def main():
    # ========== 1. 加载 SAMed2D 索引作为全局查找表 ==========
    sam_lookup = {}
    print("加载 SAMed2D 索引（约377万条）...")
    with open(SAM_INDEX_JSONL, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading SAM index"):
            item = json.loads(line)
            # 统一路径格式
            clean_img = item["image"].replace("./", "")
            sam_lookup[clean_img] = item

    print(f"  SAM 索引加载完成: {len(sam_lookup)} 条")

    # ========== 2. 从训练集构建三级 support pool ==========
    pool_lv1 = defaultdict(list)  # (dataset, modality, class)
    pool_lv2 = defaultdict(list)  # (modality, class)
    pool_lv3 = defaultdict(list)  # (class,)

    print("\n从训练集构建 support pool...")
    with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Building pool"):
            item = json.loads(line)
            img_path = item["image"].replace("./", "")
            mask_path = item["mask"].replace("./", "")

            # 通过 SAM 索引获取准确的 dataset/modality/class
            sam_info = sam_lookup.get(img_path)
            if not sam_info:
                continue

            try:
                clean_masks = [m.replace("./", "") for m in sam_info["masks"]]
                m_idx = clean_masks.index(mask_path)
                cls = sam_info["mask_classes"][m_idx]
                modality = sam_info["modality"]
                dataset = sam_info["dataset"]
            except (ValueError, KeyError):
                continue

            entry = {"image": img_path, "mask": mask_path}
            pool_lv1[(dataset, modality, cls)].append(entry)
            pool_lv2[(modality, cls)].append(entry)
            pool_lv3[(cls,)].append(entry)

    print(f"\nPool 统计:")
    print(f"  Lv1 (dataset, modality, class): {len(pool_lv1)} 个 key")
    print(f"  Lv2 (modality, class):          {len(pool_lv2)} 个 key")
    print(f"  Lv3 (class):                    {len(pool_lv3)} 个 key")

    # 统计总 entry 数
    total_entries = sum(len(v) for v in pool_lv1.values())
    print(f"  总 entry 数: {total_entries}")

    # ========== 3. 构建测试集 sid → (dataset, modality, class) 映射 ==========
    test_key_map = {}
    print("\n构建测试集 sid → key 映射...")
    with open(TEST_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            img_path = item["image"].replace("./", "")
            mask_path = item["mask"].replace("./", "")

            sam_info = sam_lookup.get(img_path)
            if not sam_info:
                continue

            try:
                clean_masks = [m.replace("./", "") for m in sam_info["masks"]]
                m_idx = clean_masks.index(mask_path)
                cls = sam_info["mask_classes"][m_idx]
                modality = sam_info["modality"]
                dataset = sam_info["dataset"]
            except (ValueError, KeyError):
                continue

            test_key_map[item["sid"]] = (dataset, modality, cls)

    print(f"  测试集映射完成: {len(test_key_map)} 个样本")

    # ========== 4. 统计测试集匹配覆盖率 ==========
    lv1_hit, lv2_hit, lv3_hit, no_hit = 0, 0, 0, 0
    for sid, (dataset, modality, cls) in test_key_map.items():
        if (dataset, modality, cls) in pool_lv1:
            lv1_hit += 1
        elif (modality, cls) in pool_lv2:
            lv2_hit += 1
        elif (cls,) in pool_lv3:
            lv3_hit += 1
        else:
            no_hit += 1

    total = len(test_key_map)
    print(f"\n测试集匹配覆盖率:")
    print(f"  Lv1 命中: {lv1_hit} ({lv1_hit/total*100:.1f}%)")
    print(f"  Lv2 命中: {lv2_hit} ({lv2_hit/total*100:.1f}%)")
    print(f"  Lv3 命中: {lv3_hit} ({lv3_hit/total*100:.1f}%)")
    print(f"  未命中:   {no_hit} ({no_hit/total*100:.1f}%)")

    # ========== 5. 保存 ==========
    output = {
        "pool_lv1": dict(pool_lv1),
        "pool_lv2": dict(pool_lv2),
        "pool_lv3": dict(pool_lv3),
        "test_key_map": test_key_map,
    }
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(output, f)
    print(f"\n已保存到 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
