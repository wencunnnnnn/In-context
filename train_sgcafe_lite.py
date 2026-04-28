"""
SGCAFE 轻量训练脚本：只加载 CLIP + SGCAFE，不加载 LLM 和 SAM
只用 loss_align + loss_seg_weak 训练，单卡 3-4GB 显存即可

用法:
    python train_sgcafe_lite.py \
        --vision_tower /media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14 \
        --sgcafe_ckpt /media/userdisk2/zhli/MedPLIB/checkpoints \
        --data_path /media/userdisk2/zhli/data/MeCoVQA/train/MeCoVQA-Grounding.json \
        --image_folder /media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1 \
        --support_pool_path /media/userdisk2/zhli/MedPLIB/support_pool.pkl \
        --epochs 10 --lr 3e-4 --batch_size 8
"""

import argparse
import json
import math
import os
import random
import re
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModel, CLIPImageProcessor

# SGCAFE 直接 import
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.medplib.model.sgcafe import SGCAFEModule


# ============================================================
# Dataset: 只需要 query 图 + support 图 + mask
# ============================================================
class SGCAFEDataset(Dataset):
    """轻量 dataset：只做 CLIP 预处理，不做 tokenizer/SAM 预处理"""

    def __init__(self, data_path, image_folder, support_pool_path, clip_processor):
        with open(data_path) as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.clip_processor = clip_processor

        # 加载 support pool
        import pickle
        with open(support_pool_path, 'rb') as f:
            pool = pickle.load(f)
        self.pool_lv1 = pool['pool_lv1']
        self.pool_lv2 = pool['pool_lv2']
        self.pool_lv3 = pool['pool_lv3']

        # 解析每个样本的 metadata
        self.meta_cache = {}
        self.patterns = [
            r'segments?\s+the\s+(.+?)\s+in\s+this',
            r'segmenting\s+the\s+(.+?)\s+in\s+this',
            r'segment\s+out\s+the\s+(.+?)\s+in\s+this',
            r'segment\s+the\s+(.+?)\s+in\s+this',
            r'mask\s+(?:for|of)\s+the\s+(.+?)\s+in\s+this',
        ]

    def __len__(self):
        return len(self.data)

    def _parse_meta(self, sample_id, conv_text):
        parts = sample_id.split('--')
        if len(parts) < 2:
            return None
        modality = parts[0].split('_')[0]
        dataset = parts[1]
        cls = None
        for p in self.patterns:
            m = re.search(p, conv_text, re.IGNORECASE)
            if m:
                cls = m.group(1).strip().replace(' ', '_')
                break
        if cls is None:
            return None
        return (dataset, modality, cls)

    def _get_support(self, sample_id, conv_text):
        meta = self._parse_meta(sample_id, conv_text)
        if meta is None:
            return None, None
        dataset, modality, cls = meta

        candidates = None
        if (dataset, modality, cls) in self.pool_lv1:
            candidates = self.pool_lv1[(dataset, modality, cls)]
        elif (modality, cls) in self.pool_lv2:
            candidates = self.pool_lv2[(modality, cls)]
        elif (cls,) in self.pool_lv3:
            candidates = self.pool_lv3[(cls,)]
        if not candidates:
            return None, None

        filtered = [e for e in candidates if sample_id not in e['image']]
        if not filtered:
            return None, None

        entry = random.choice(filtered)
        img_path = os.path.join(self.image_folder, entry['image'])
        mask_path = os.path.join(self.image_folder, entry['mask'])
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            return None, None

        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize((img.shape[1], img.shape[0]), Image.NEAREST)
            mask_binary = (np.array(mask) > 0).astype(np.float32)
            return img, mask_binary
        except Exception:
            return None, None

    def _make_mask_weights(self, mask_binary, num_patches=576):
        mask_resized = cv2.resize(mask_binary, (24, 24), interpolation=cv2.INTER_AREA)
        return torch.tensor(mask_resized.flatten()[:num_patches], dtype=torch.float32).clamp(0, 1)

    def __getitem__(self, idx):
        for _ in range(10):
            item = self.data[idx]
            sample_id = item['id']
            conv_text = item['conversations'][0]['value']

            # query 图
            img_path = os.path.join(self.image_folder, item['image'])
            if not os.path.exists(img_path):
                idx = random.randint(0, len(self.data) - 1)
                continue

            # query GT mask
            gpt_val = item['conversations'][1]['value'] if len(item['conversations']) > 1 else ''
            mask_match = re.search(r'<mask>(.*?)</mask>', gpt_val)
            if not mask_match:
                idx = random.randint(0, len(self.data) - 1)
                continue
            query_mask_path = os.path.join(self.image_folder, mask_match.group(1))
            if not os.path.exists(query_mask_path):
                idx = random.randint(0, len(self.data) - 1)
                continue

            # support 图 + mask
            support_rgb, support_mask = self._get_support(sample_id, conv_text)
            if support_rgb is None:
                idx = random.randint(0, len(self.data) - 1)
                continue

            try:
                # CLIP 预处理
                query_img = Image.open(img_path).convert('RGB')
                query_clip = self.clip_processor(images=query_img, return_tensors='pt')['pixel_values'][0]
                support_pil = Image.fromarray(support_rgb)
                support_clip = self.clip_processor(images=support_pil, return_tensors='pt')['pixel_values'][0]

                # mask weights (576,)
                support_mask_weights = self._make_mask_weights(support_mask)

                # query GT mask 下采样到 24x24
                query_mask_img = Image.open(query_mask_path).convert('L')
                query_mask_img = query_mask_img.resize((query_img.size[0], query_img.size[1]), Image.NEAREST)
                query_mask_np = (np.array(query_mask_img) > 0).astype(np.float32)
                query_mask_24 = cv2.resize(query_mask_np, (24, 24), interpolation=cv2.INTER_AREA)
                query_mask_flat = torch.tensor(query_mask_24.flatten(), dtype=torch.float32).clamp(0, 1)  # (576,)

                return {
                    'query_clip': query_clip,
                    'support_clip': support_clip,
                    'support_mask_weights': support_mask_weights,
                    'query_mask_flat': query_mask_flat,
                }
            except Exception:
                idx = random.randint(0, len(self.data) - 1)
                continue

        # fallback: 返回零数据
        return {
            'query_clip': torch.zeros(3, 224, 224),
            'support_clip': torch.zeros(3, 224, 224),
            'support_mask_weights': torch.zeros(576),
            'query_mask_flat': torch.zeros(576),
        }


# ============================================================
# 训练主函数
# ============================================================
def train(args):
    device = torch.device(f'cuda:{args.gpu}')

    # 1. 加载 CLIP（冻结）
    print(f"[1/3] Loading CLIP from {args.vision_tower} ...")
    clip_model = CLIPVisionModel.from_pretrained(args.vision_tower, torch_dtype=torch.bfloat16)
    clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    clip_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)

    clip_dim = clip_model.config.hidden_size  # 1024

    # 2. 加载 SGCAFE（可训练）
    print("[2/3] Initializing SGCAFE ...")
    sgcafe = SGCAFEModule(clip_dim=clip_dim, inner_dim=256, num_heads=4)

    # 尝试从 checkpoint 加载已有权重
    if args.sgcafe_ckpt:
        loaded = False
        for fname in ['pytorch_model.bin', 'model.safetensors']:
            fpath = os.path.join(args.sgcafe_ckpt, fname)
            if not os.path.exists(fpath):
                continue
            if fname.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(fpath)
            else:
                state_dict = torch.load(fpath, map_location='cpu')
            sgcafe_keys = {k.replace('model.sgcafe.', ''): v for k, v in state_dict.items() if 'sgcafe' in k}
            if sgcafe_keys:
                sgcafe.load_state_dict(sgcafe_keys, strict=False)
                print(f"  Loaded {len(sgcafe_keys)} SGCAFE keys from {fname}")
                loaded = True
                break
        if not loaded:
            print("  No SGCAFE keys found in checkpoint, using random init")
    else:
        print("  No checkpoint specified, using random init")

    sgcafe.to(device=device, dtype=torch.bfloat16).train()
    trainable_params = sum(p.numel() for p in sgcafe.parameters() if p.requires_grad)
    print(f"  SGCAFE trainable params: {trainable_params:,}")

    # 3. Dataset & DataLoader
    print("[3/3] Loading dataset ...")
    dataset = SGCAFEDataset(args.data_path, args.image_folder, args.support_pool_path, clip_processor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"  Dataset size: {len(dataset)}, batches/epoch: {len(loader)}")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(sgcafe.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(loader)
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # 训练循环
    print(f"\n{'='*50}")
    print(f"Start training: {args.epochs} epochs, lr={args.lr}")
    print(f"{'='*50}\n")

    save_dir = os.path.join('runs', 'sgcafe_lite')
    os.makedirs(save_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        sgcafe.train()
        epoch_loss_align = 0.0
        epoch_loss_weak = 0.0
        epoch_count = 0

        for batch_idx, batch in enumerate(loader):
            query_clip = batch['query_clip'].to(device=device, dtype=torch.bfloat16)
            support_clip = batch['support_clip'].to(device=device, dtype=torch.bfloat16)
            support_mask_weights = batch['support_mask_weights'].to(device=device, dtype=torch.float32)
            query_mask_flat = batch['query_mask_flat'].to(device=device, dtype=torch.float32)

            # CLIP 编码（冻结）
            with torch.no_grad():
                q_out = clip_model(query_clip).last_hidden_state[:, 1:, :]   # (B, 576, 1024)
                s_out = clip_model(support_clip).last_hidden_state[:, 1:, :]

            # SGCAFE 前向
            enhanced, attn_map, support_dropped = sgcafe(q_out, s_out, support_mask_weights)

            if support_dropped:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                # loss_align
                attn_mean = attn_map.mean(dim=1).float()
                s_mask = support_mask_weights.float()
                attn_to_fg = (attn_mean * s_mask.unsqueeze(1)).sum(-1).clamp(1e-6, 1 - 1e-6)
                loss_align = F.binary_cross_entropy(attn_to_fg, query_mask_flat)

                # loss_seg_weak
                aux_pred = sgcafe.aux_forward(enhanced).flatten(1).float()
                loss_seg_weak = F.binary_cross_entropy_with_logits(aux_pred, query_mask_flat)

                loss = args.align_weight * loss_align + args.seg_weak_weight * loss_seg_weak

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sgcafe.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            if not support_dropped:
                epoch_loss_align += loss_align.item()
                epoch_loss_weak += loss_seg_weak.item()
                epoch_count += 1

            if global_step % args.print_freq == 0:
                avg_a = epoch_loss_align / max(epoch_count, 1)
                avg_w = epoch_loss_weak / max(epoch_count, 1)
                lr_now = scheduler.get_last_lr()[0]
                print(f"[Epoch {epoch+1}/{args.epochs}] Step {global_step} | "
                      f"loss_align={avg_a:.4f} loss_weak={avg_w:.4f} lr={lr_now:.6f}")

        avg_a = epoch_loss_align / max(epoch_count, 1)
        avg_w = epoch_loss_weak / max(epoch_count, 1)
        print(f"\n--- Epoch {epoch+1} done: loss_align={avg_a:.4f} loss_weak={avg_w:.4f} ---")

        ckpt_path = os.path.join(save_dir, f'sgcafe_epoch{epoch+1}.pt')
        torch.save(sgcafe.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path}\n")

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vision_tower', type=str, required=True)
    parser.add_argument('--sgcafe_ckpt', type=str, default=None)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--support_pool_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=50)
    parser.add_argument('--align_weight', type=float, default=1.0)
    parser.add_argument('--seg_weak_weight', type=float, default=0.1)
    args = parser.parse_args()
    train(args)
