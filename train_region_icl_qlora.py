"""
QLoRA 训练脚本：Region ICL + SGCAFE
- LLM: 4bit 量化 + LoRA 微调
- SGCAFE: 全量微调
- CLIP / SAM / mm_projector / region_fea_adapter: 冻结
- 训练数据: 分割数据 + <region> prompt，让 router 学会 <region> + <SEG>
- 支持多卡训练 (torchrun)
"""

import argparse
import json
import math
import os
import random
import sys
import time
import types
from functools import partial

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import tqdm
import transformers
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.MedPLIB import MedPLIBForCausalLM
from model.medplib import conversation as conversation_lib
from datasets import LazySupervisedDataset, DataCollatorForSupervisedDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, dict_to_cuda, ADD_OTHERS_TOKENS,
                         intersectionAndUnionGPU)


def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA Region ICL + SGCAFE Training")

    # Model
    parser.add_argument("--version", required=True, type=str, help="Path to model checkpoint")
    parser.add_argument("--vision_tower", required=True, type=str)
    parser.add_argument("--vision_pretrained", required=True, type=str, help="SAM checkpoint")
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--sam_img_size", default=256, type=int)

    # Data
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--image_folder", required=True, type=str)
    parser.add_argument("--support_pool_path", required=True, type=str)

    # LoRA
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    # Training
    parser.add_argument("--epochs", default=3, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--grad_accumulation_steps", default=16, type=int)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--sgcafe_lr", default=1e-3, type=float)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--gpu", default=1, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--print_freq", default=20, type=int)
    parser.add_argument("--save_steps", default=500, type=int)
    parser.add_argument("--run_name", default="region_icl_qlora", type=str, help="Run name for save dir")

    # Loss weights
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--align_loss_weight", default=1.0, type=float)
    parser.add_argument("--seg_weak_loss_weight", default=0.1, type=float)

    # MoE (保持和原模型一致)
    parser.add_argument("--moe_enable", type=bool, default=True)
    parser.add_argument("--moe_mode", default="dense", type=str)
    parser.add_argument("--num_experts", default=2, type=int)
    parser.add_argument("--top_k_experts", default=1, type=int)
    parser.add_argument("--capacity_factor", default=1.5, type=float)
    parser.add_argument("--use_residual", type=bool, default=False)
    parser.add_argument("--router_aux_loss_coef", default=0.0, type=float)
    parser.add_argument("--eval_capacity_factor", default=2.0, type=float)
    parser.add_argument("--min_capacity", default=0, type=int)
    parser.add_argument("--ep_size", default=1, type=int)
    parser.add_argument("--expert_pretrained_path", type=str, default=None)
    parser.add_argument("--moe_layers_idx", type=str, default=None)
    parser.add_argument("--return_gating_logit", type=bool, default=False)

    # Region
    parser.add_argument("--bypass_sgcafe", action="store_true", default=False,
                        help="跳过 SGCAFE，不传 support_clip，只保留 Region ICL")
    parser.add_argument("--region_fea_adapter", action="store_true", default=True)
    parser.add_argument("--region_geo_sampler", action="store_true", default=False)
    parser.add_argument("--max_sample_point", default=512, type=int)
    parser.add_argument("--sampler_pooler_mode", default="max", type=str)

    return parser.parse_args()


def set_seed(seed=42, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True


_GLOO_GROUP = None  # 专用于 CPU 梯度 all-reduce（绕开 NCCL socket 问题）


def setup_distributed():
    """初始化分布式训练环境。
    单机多卡无 NVLink 时，NCCL 依赖 socket，容易因 ppp0/docker
    虚拟接口干扰而崩溃。解决方案：
      - NCCL 进程组：保留，用于模型前向/其他 CUDA 集合通信
      - gloo 进程组：额外创建，用于手动梯度 all-reduce（共享内存路径，极稳定）
    """
    global _GLOO_GROUP
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        # 额外建立 gloo 进程组，专门用于 CPU 梯度同步
        _GLOO_GROUP = dist.new_group(backend="gloo")
        return rank, local_rank, world_size, True
    else:
        return 0, 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def sync_trainable_grads(model, world_size):
    """All-reduce trainable gradients as a single flat tensor via gloo (CPU).

    Batching into one call avoids N round-trips for large params like text_hidden_fcs.
    """
    params = [p for p in model.parameters()
              if p.requires_grad and p.dtype in (torch.float32, torch.float16, torch.bfloat16)]
    if not params:
        return

    grads = []
    for p in params:
        g = p.grad.data if p.grad is not None else torch.zeros_like(p.data)
        grads.append(g.reshape(-1).float().cpu())

    flat = torch.cat(grads).contiguous()
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=_GLOO_GROUP)
    flat.div_(world_size)

    offset = 0
    for p in params:
        n = p.numel()
        chunk = flat[offset:offset + n].reshape(p.shape).to(dtype=p.data.dtype)
        if p.grad is None:
            p.grad = chunk.to(device=p.device)
        else:
            p.grad.data.copy_(chunk.to(device=p.device, dtype=p.grad.dtype))
        offset += n


def train(args):
    rank, local_rank, world_size, is_distributed = setup_distributed()
    set_seed(rank=rank)
    is_main = (rank == 0)

    device = torch.device(f"cuda:{local_rank}")

    # ============================================================
    # 1. 加载模型 (4bit 量化)
    # ============================================================
    if is_main:
        print("=" * 50)
        print(f"[1/5] Loading model with 4-bit quantization ... (world_size={world_size})")
        print("=" * 50)

    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["region_fea_adapter"],
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False)
    tokenizer.pad_token = tokenizer.unk_token

    # 添加特殊 token
    for token_name in ADD_OTHERS_TOKENS:
        tokenizer.add_tokens(token_name, special_tokens=True)
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model_args = vars(args)
    model = MedPLIBForCausalLM.from_pretrained(
        args.version,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        device_map={"": device},
        **model_args,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 初始化 vision modules 和 SAM
    model.get_model().initialize_vision_modules(model.get_model().config)
    model.get_model().initialize_bird_modules(model.get_model().config)

    # 注意：checkpoint 已经是完整的 MoE 模型（config.json 含 moe 配置，
    # 权重已是 deepspeed_moe 格式），from_pretrained 已正确加载。
    # 不需要再调用 initialize_moe_modules（那是从非 MoE 模型创建 MoE 用的）。
    # 但需要手动设置 router_aux_loss_coef（原本在 initialize_moe_modules 里设置）
    model.router_aux_loss_coef = args.router_aux_loss_coef

    # 4-bit 量化后 lm_head 是 uint8，resize_token_embeddings 会报错。
    # checkpoint 已包含所有特殊 token，只在大小不匹配时才 resize。
    current_embed_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != current_embed_size:
        if is_main:
            print(f"  Warning: tokenizer size {len(tokenizer)} != embed size {current_embed_size}, skipping resize for 4-bit model")
    else:
        if is_main:
            print(f"  Embedding size matches tokenizer: {current_embed_size}")

    conversation_lib.default_conversation = conversation_lib.conv_templates["llava_v1"]

    # ============================================================
    # 2. 准备 QLoRA
    # ============================================================
    if is_main:
        print("\n[2/5] Preparing QLoRA ...")
        # 验证 modules_to_not_convert 生效：这两个模块必须是 nn.Linear 而非 bnb.nn.Linear4bit
        rfa = model.get_model().region_fea_adapter
        thf = model.get_model().text_hidden_fcs
        print(f"  region_fea_adapter type: {type(rfa).__name__}, weight dtype: {rfa.weight.dtype}, shape: {rfa.weight.shape}")
        for i, seq in enumerate(thf):
            for j, layer in enumerate(seq):
                if hasattr(layer, 'weight'):
                    print(f"  text_hidden_fcs[{i}][{j}] type: {type(layer).__name__}, weight dtype: {layer.weight.dtype}, shape: {layer.weight.shape}")
        import bitsandbytes as bnb
        assert not isinstance(rfa, bnb.nn.Linear4bit), "FATAL: region_fea_adapter was quantized to 4-bit!"

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 冻结所有参数
    for p in model.parameters():
        p.requires_grad = False

    # LoRA on LLM
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and any(t in name for t in lora_target_modules):
            if not any(x in name for x in ["visual_model", "vision_tower", "mm_projector", "sgcafe"]):
                lora_module_names.append(name)

    if is_main:
        print(f"  LoRA target modules: {len(lora_module_names)} layers")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=lora_module_names,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # SGCAFE 全量微调（只对浮点参数开启梯度）
    if not args.bypass_sgcafe:
        # 重新初始化 SGCAFE：checkpoint 中没有 SGCAFE 权重，4-bit 加载 + prepare_model_for_kbit_training
        # 会把随机初始化的参数搞乱（weight 被 flatten、值变 NaN）。用全新实例替换。
        from model.medplib.model.sgcafe import SGCAFEModule
        fresh_sgcafe = SGCAFEModule(
            clip_dim=1024, inner_dim=256, num_heads=4,
            mask_bias_alpha=5.0, support_dropout=0.1,
        )
        model.base_model.model.get_model().sgcafe = fresh_sgcafe
        if is_main:
            print(f"  Re-initialized SGCAFE module (fresh weights)")

        for p in model.base_model.model.get_model().sgcafe.parameters():
            if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
                p.requires_grad = True
    else:
        if is_main:
            print(f"  SGCAFE bypassed — skipping init/unfreeze")

    # 解冻 region_fea_adapter：让 support 区域特征的投影能适配 ICL 场景
    for p in model.base_model.model.get_model().region_fea_adapter.parameters():
        if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
            p.requires_grad = True

    # 解冻 text_hidden_fcs：让分割头能利用 region 增强后的 [SEG] 表示
    for p in model.base_model.model.get_model().text_hidden_fcs.parameters():
        if p.dtype in (torch.float32, torch.float16, torch.bfloat16):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # 传递损失权重
    model.base_model.model.align_loss_weight = args.align_loss_weight
    model.base_model.model.seg_weak_loss_weight = args.seg_weak_loss_weight
    model.base_model.model.ce_loss_weight = args.ce_loss_weight
    model.base_model.model.dice_loss_weight = args.dice_loss_weight
    model.base_model.model.bce_loss_weight = args.bce_loss_weight
    model.base_model.model.iou_loss_weight = 0.0
    model.base_model.model.focal_loss_weight = 0.0

    # ============================================================
    # 3. Dataset
    # ============================================================
    if is_main:
        print("\n[3/5] Loading dataset ...")

    vision_tower = model.base_model.model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16, device=device)

    # SAM 也需要转 bf16（冻结状态，只做前向）
    visual_model = model.base_model.model.get_model().visual_model
    visual_model.to(dtype=torch.bfloat16, device=device)

    # SGCAFE 保持 float32（内部自行处理 float32 计算，避免 bf16 下 NaN）
    if not args.bypass_sgcafe:
        sgcafe_module = model.base_model.model.get_model().sgcafe
        sgcafe_module.to(device=device)
        from accelerate.hooks import remove_hook_from_module
        for submodule in sgcafe_module.modules():
            remove_hook_from_module(submodule)

    # 确保其他子模块也在 GPU 上，region_fea_adapter 需要 bfloat16 匹配 CLIP 输出
    inner_model = model.base_model.model.get_model()
    for name in ['text_hidden_fcs', 'mm_projector']:
        mod = getattr(inner_model, name, None)
        if mod is not None:
            mod.to(device=device)
    inner_model.region_fea_adapter.to(device=device, dtype=torch.bfloat16)

    data_args = {
        "image_folder": args.image_folder,
        "image_aspect_ratio": "pad",
        "is_multimodal": True,
        "mm_use_im_start_end": True,
        "image_processor": vision_tower.image_processor,
    }
    data_args = types.SimpleNamespace(**data_args)

    train_dataset = LazySupervisedDataset(
        args.data_path, tokenizer, data_args, args.sam_img_size,
        use_visual_icl=True,
        support_pool_path=args.support_pool_path,
        use_region_icl=True,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers, pin_memory=True, drop_last=True,
        collate_fn=partial(DataCollatorForSupervisedDataset),
    )
    if is_main:
        print(f"  Dataset: {len(train_dataset)} samples, {len(train_loader)} batches/epoch/gpu")

    # ============================================================
    # 4. Optimizer
    # ============================================================
    if is_main:
        print("\n[4/5] Setting up optimizer ...")

    # 分离 optimizer：LoRA 用低学习率，SGCAFE 用高学习率，adapter 用中等学习率
    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not any(x in n for x in ['sgcafe', 'region_fea_adapter', 'text_hidden_fcs'])]
    sgcafe_params = [p for n, p in model.named_parameters()
                     if p.requires_grad and 'sgcafe' in n]
    adapter_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and any(x in n for x in ['region_fea_adapter', 'text_hidden_fcs'])]
    if is_main:
        print(f"  LoRA params: {sum(p.numel() for p in lora_params):,} (lr={args.lr})")
        print(f"  SGCAFE params: {sum(p.numel() for p in sgcafe_params):,} (lr={args.sgcafe_lr})")
        print(f"  Adapter params (region_fea_adapter + text_hidden_fcs): {sum(p.numel() for p in adapter_params):,} (lr={args.lr})")
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'lr': args.lr},
        {'params': sgcafe_params, 'lr': args.sgcafe_lr},
        {'params': adapter_params, 'lr': args.lr},
    ], weight_decay=0.01)
    total_steps = args.epochs * len(train_loader) // args.grad_accumulation_steps
    warmup_steps = min(100, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        return max(0.05, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ============================================================
    # 5. Training loop
    # ============================================================
    if is_main:
        print(f"\n{'='*50}")
        print(f"[5/5] Start training: {args.epochs} epochs, {world_size} GPUs")
        print(f"  Effective batch size: {args.batch_size * args.grad_accumulation_steps * world_size}")
        print(f"  Total steps: {total_steps}")
        print(f"{'='*50}\n")

    save_dir = os.path.join("runs", args.run_name)
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        loss_meter = AverageMeter("Loss", ":.4f")
        ce_meter = AverageMeter("CE", ":.4f")
        mask_meter = AverageMeter("Mask", ":.4f")
        align_meter = AverageMeter("Align", ":.4f")
        oom_count = 0

        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        for batch_idx, input_dict in enumerate(pbar):
            skip_step = False
            try:
                input_dict = dict_to_cuda(input_dict, device=device)

                # 精度转换
                input_dict["images"] = input_dict["images"].bfloat16()
                if isinstance(input_dict["images_clip"], list):
                    input_dict["images_clip"] = [x.bfloat16() for x in input_dict["images_clip"]]
                else:
                    input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                if "support_clip" in input_dict:
                    input_dict["support_clip"] = input_dict["support_clip"].bfloat16()
                if "icl_region_clip" in input_dict:
                    input_dict["icl_region_clip"] = input_dict["icl_region_clip"].bfloat16()

                # Forward
                output_dict = model.base_model.model.model_forward(
                    images=input_dict["images"],
                    images_clip=input_dict["images_clip"],
                    input_ids=input_dict["input_ids"],
                    labels=input_dict["labels"],
                    attention_mask=input_dict["attention_mask"],
                    offset=input_dict["offset"],
                    masks_list=input_dict["masks_list"],
                    label_list=input_dict["label_list"],
                    resize_list=input_dict["resize_list"],
                    inference=False,
                    seg_flag=input_dict.get("seg_flag", True),
                    valid_mask_bool=input_dict.get("valid_mask_bool", []),
                    rp_flag=input_dict.get("rp_flag", False),
                    region_masks=input_dict.get("region_masks", []),
                    valid_region_masks_bool=input_dict.get("valid_region_masks_bool", []),
                    support_clip=None if args.bypass_sgcafe else input_dict.get("support_clip", None),
                    support_mask_weights=None if args.bypass_sgcafe else input_dict.get("support_mask_weights", None),
                    icl_region_clip=input_dict.get("icl_region_clip", None),
                )

                loss = output_dict["loss"] / args.grad_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    if is_main:
                        print(f"\n  NaN/Inf loss at batch {batch_idx}, skipping ...")
                    skip_step = True
                else:
                    loss.backward()

                    # Update meters
                    loss_meter.update(output_dict["loss"].item())
                    ce_meter.update(output_dict["ce_loss"].item())
                    mask_meter.update(output_dict.get("mask_loss", torch.tensor(0)).item())
                    align_meter.update(output_dict.get("loss_align", torch.tensor(0)).item())

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                alloc = torch.cuda.memory_allocated(device) / 1e9
                print(f"\n  [rank {dist.get_rank() if is_distributed else 0}] OOM at batch {batch_idx}, "
                      f"allocated={alloc:.2f}GB, skipping ...")
                skip_step = True

            if (batch_idx + 1) % args.grad_accumulation_steps == 0:
                # Propagate skip decision: if ANY rank wants to skip, ALL skip
                if is_distributed:
                    skip_tensor = torch.tensor([int(skip_step)], device=device, dtype=torch.int32)
                    dist.all_reduce(skip_tensor, op=dist.ReduceOp.MAX, group=_GLOO_GROUP)
                    skip_step = bool(skip_tensor.item())

                if skip_step:
                    optimizer.zero_grad()
                    oom_count += 1
                else:
                    # 多卡：手动 all-reduce 可训练参数梯度
                    if is_distributed:
                        sync_trainable_grads(model, world_size)

                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if is_main and global_step % args.print_freq == 0:
                        lr_now = scheduler.get_last_lr()[0]
                        postfix = dict(loss=f"{loss_meter.avg:.3f}",
                                       ce=f"{ce_meter.avg:.3f}",
                                       mask=f"{mask_meter.avg:.3f}",
                                       align=f"{align_meter.avg:.3f}",
                                       lr=f"{lr_now:.6f}",
                                       oom=oom_count)
                        if not args.bypass_sgcafe:
                            with torch.no_grad():
                                gate_bias = model.base_model.model.get_model().sgcafe.gate_proj.bias.data
                                gate_mean = torch.sigmoid(gate_bias).mean().item()
                            postfix["gate"] = f"{gate_mean:.4f}"
                        pbar.set_postfix(**postfix)

                    if is_main and global_step % args.save_steps == 0:
                        ckpt_path = os.path.join(save_dir, f"step_{global_step}")
                        os.makedirs(ckpt_path, exist_ok=True)
                        # 保存 LoRA
                        model.save_pretrained(ckpt_path)
                        if not args.bypass_sgcafe:
                            torch.save(model.base_model.model.get_model().sgcafe.state_dict(),
                                       os.path.join(ckpt_path, "sgcafe.pt"))
                        # 保存 region_fea_adapter + text_hidden_fcs
                        rfa_sd = model.base_model.model.get_model().region_fea_adapter.state_dict()
                        thf_sd = model.base_model.model.get_model().text_hidden_fcs.state_dict()
                        assert rfa_sd['weight'].shape == (4096, 1024), f"FATAL: region_fea_adapter weight shape {rfa_sd['weight'].shape} != (4096, 1024), still quantized?"
                        torch.save({
                            'region_fea_adapter': rfa_sd,
                            'text_hidden_fcs': thf_sd,
                        }, os.path.join(ckpt_path, "adapters.pt"))
                        print(f"\n  Saved checkpoint: {ckpt_path}")

            elif skip_step:
                optimizer.zero_grad()

        # Epoch 结束
        if is_main:
            print(f"\n--- Epoch {epoch+1} done: loss={loss_meter.avg:.4f} ce={ce_meter.avg:.4f} "
                  f"mask={mask_meter.avg:.4f} align={align_meter.avg:.4f} ---")

            ckpt_path = os.path.join(save_dir, f"epoch_{epoch+1}")
            os.makedirs(ckpt_path, exist_ok=True)
            model.save_pretrained(ckpt_path)
            if not args.bypass_sgcafe:
                torch.save(model.base_model.model.get_model().sgcafe.state_dict(),
                           os.path.join(ckpt_path, "sgcafe.pt"))
            torch.save({
                'region_fea_adapter': model.base_model.model.get_model().region_fea_adapter.state_dict(),
                'text_hidden_fcs': model.base_model.model.get_model().text_hidden_fcs.state_dict(),
            }, os.path.join(ckpt_path, "adapters.pt"))
            print(f"  Saved: {ckpt_path}\n")

        # 等所有 rank 完成再进入下一个 epoch
        if is_distributed:
            dist.barrier()

    if is_main:
        print("Training complete!")
    cleanup_distributed()


if __name__ == "__main__":
    args = parse_args()
    train(args)
