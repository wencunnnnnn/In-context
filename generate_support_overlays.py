"""
Step 2: Overlay 示例图像生成工具

提供 create_overlay() 函数，将 support mask 半透明叠加到 support image 上。
推理时动态调用，不需要预生成。

本脚本同时提供测试功能：从 support pool 中取几个样本，生成 overlay 并保存，供目视检查。
"""

import os
import pickle
import random
import numpy as np
from PIL import Image


DATA_ROOT = "/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1"
POOL_PATH = "/media/userdisk2/zhli/MedPLIB/support_pool.pkl"
OUTPUT_DIR = "/media/userdisk2/zhli/MedPLIB/overlay_examples"


def create_overlay(image_path, mask_path, alpha=0.5, color=(0, 255, 0)):
    """
    将 mask 半透明叠加到 image 上。

    Args:
        image_path: support image 的完整路径
        mask_path:  support mask 的完整路径
        alpha:      叠加透明度 (0~1)
        color:      叠加颜色 (R, G, B)，默认绿色

    Returns:
        PIL.Image: overlay 后的 RGB 图像
    """
    # 加载图像（转 RGB）
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float32)

    # 加载 mask（转灰度，二值化）
    mask = Image.open(mask_path).convert("L")
    mask = mask.resize(img.size, Image.NEAREST)
    mask_arr = (np.array(mask) > 0).astype(np.float32)

    # 半透明叠加
    color_arr = np.array(color, dtype=np.float32)
    for c in range(3):
        img_arr[:, :, c] = np.where(
            mask_arr > 0,
            (1 - alpha) * img_arr[:, :, c] + alpha * color_arr[c],
            img_arr[:, :, c],
        )

    return Image.fromarray(img_arr.astype(np.uint8))


def test_overlay():
    """从 support pool 取样本生成 overlay 并保存，供目视检查。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(POOL_PATH, "rb") as f:
        pool_data = pickle.load(f)

    pool_lv1 = pool_data["pool_lv1"]

    # 随机取 5 个 key，每个取 1 个样本
    keys = random.sample(list(pool_lv1.keys()), min(5, len(pool_lv1)))

    for key in keys:
        entry = random.choice(pool_lv1[key])
        img_path = os.path.join(DATA_ROOT, entry["image"])
        mask_path = os.path.join(DATA_ROOT, entry["mask"])

        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            print(f"  [SKIP] 文件不存在: {key}")
            continue

        overlay = create_overlay(img_path, mask_path)

        # 保存：原图 + overlay 并排对比
        img_orig = Image.open(img_path).convert("RGB")
        comparison = Image.new("RGB", (img_orig.width * 2, img_orig.height))
        comparison.paste(img_orig, (0, 0))
        comparison.paste(overlay, (img_orig.width, 0))

        dataset, modality, cls = key
        fname = f"{modality}_{dataset}_{cls}.png"
        comparison.save(os.path.join(OUTPUT_DIR, fname))
        print(f"  已保存: {fname} (key={key})")

    print(f"\n所有示例已保存到 {OUTPUT_DIR}")


if __name__ == "__main__":
    test_overlay()
