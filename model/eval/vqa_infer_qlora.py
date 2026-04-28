"""
QLoRA Region ICL 推理脚本
- 加载 base model (bf16) + merge LoRA adapter + 加载 SGCAFE 权重
- 不修改原始 vqa_infer.py，独立运行
"""
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import time
from functools import partial
import types
import math
import random
import json

import deepspeed
from accelerate.hooks import remove_hook_from_submodules
import numpy as np
import torch
import tqdm
import transformers
from peft import PeftModel
from torch.utils.data import Dataset, Subset

from model.MedPLIB import MedPLIBForCausalLM
from model.medplib import conversation as conversation_lib
from datasets import LazySupervisedDataset, DataCollatorForSupervisedDataset
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU, ADD_OTHERS_TOKENS)


def parse_args(args):
    parser = argparse.ArgumentParser(description="QLoRA Region ICL Inference")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--version", required=True, type=str, help="Base model path")
    parser.add_argument("--lora_path", required=True, type=str, help="LoRA adapter checkpoint dir (contains adapter_config.json + adapter_model.safetensors + sgcafe.pt)")
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--sam_img_size", default=256, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--vision_tower", required=True, type=str)
    parser.add_argument("--vision_pretrained", required=True, type=str)

    # Data
    parser.add_argument('--image_folder', type=str, required=True)
    parser.add_argument('--image_aspect_ratio', type=str, default='pad')
    parser.add_argument('--is_multimodal', action='store_true', default=True)
    parser.add_argument('--val_data_path', type=str, required=True)
    parser.add_argument('--answer_type', type=str, default='closed')

    # ICL
    parser.add_argument('--use_visual_icl', action='store_true', default=False)
    parser.add_argument('--support_pool_path', type=str, default=None)
    parser.add_argument('--use_token_icl_concat', action='store_true', default=False)
    parser.add_argument('--use_token_icl_multi', action='store_true', default=False)
    parser.add_argument('--use_region_icl', action='store_true', default=False)
    parser.add_argument('--region_icl_self', action='store_true', default=False)

    # Eval
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument('--eval_seg', action='store_true', default=False)
    parser.add_argument('--eval_vqa', action='store_true', default=False)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--vis_mask', action='store_true', default=False)

    # Loss (needed by model init)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--iou_loss_weight", default=2.0, type=float)
    parser.add_argument("--focal_loss_weight", default=2.0, type=float)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)

    # Region
    parser.add_argument("--region_fea_adapter", action="store_true", default=False)
    parser.add_argument("--region_geo_sampler", action="store_true", default=False)
    parser.add_argument("--max_sample_point", default=512, type=int)
    parser.add_argument("--sampler_pooler_mode", default='max', type=str)

    # MoE
    parser.add_argument('--moe_enable', action='store_true', default=False)
    parser.add_argument('--moe_mode', type=str, default='second_half')
    parser.add_argument('--num_experts', type=int, default=3)
    parser.add_argument('--top_k_experts', type=int, default=2)
    parser.add_argument('--capacity_factor', type=float, default=1)
    parser.add_argument('--use_residual', action='store_true', default=False)
    parser.add_argument('--router_aux_loss_coef', type=float, default=0.01)
    parser.add_argument('--eval_capacity_factor', type=float, default=2)
    parser.add_argument('--moe_layers_idx', type=str, default=None)
    parser.add_argument('--min_capacity', type=int, default=0)
    parser.add_argument('--ep_size', type=int, default=1)
    parser.add_argument('--expert_pretrained_path', type=str, default=None)
    parser.add_argument('--return_gating_logit', action='store_true', default=False)

    # SGCAFE bypass
    parser.add_argument('--bypass_sgcafe', action='store_true', default=False,
                        help="Bypass SGCAFE module (set gate=0), only use Region ICL")

    return parser.parse_args(args)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_iou(prediction_mask, ground_truth_mask):
    prediction_mask = prediction_mask.bool()
    ground_truth_mask = ground_truth_mask.bool()
    intersection = torch.logical_and(prediction_mask, ground_truth_mask)
    union = torch.logical_or(prediction_mask, ground_truth_mask)
    intersection_pixels = torch.sum(intersection)
    union_pixels = torch.sum(union)
    if union_pixels == 0:
        return 0.0
    return (intersection_pixels.float() / union_pixels.float()).item()


import cv2
def vis_overlay_masks(original_image_path, prediction_mask, ground_truth_mask, save_path):
    original_image = cv2.imread(original_image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    prediction_mask = prediction_mask.squeeze(0).cpu().numpy() * 255
    ground_truth_mask = ground_truth_mask.squeeze(0).cpu().numpy() * 255
    overlay_color = np.array([118, 158, 224], dtype=np.uint8)
    prediction_overlay = np.zeros_like(original_image)
    ground_truth_overlay = np.zeros_like(original_image)
    prediction_overlay[prediction_mask > 0] = overlay_color
    ground_truth_overlay[ground_truth_mask > 0] = overlay_color
    prediction_overlay_image = cv2.addWeighted(original_image, 0.5, prediction_overlay, 0.9, 0)
    ground_truth_overlay_image = cv2.addWeighted(original_image, 0.5, ground_truth_overlay, 0.9, 0)
    combined_image = np.concatenate([original_image, ground_truth_overlay_image, prediction_overlay_image], axis=1)
    cv2.imwrite(save_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))


def main(args):
    args = parse_args(args)
    set_seed(42)

    # 初始化分布式后端（单进程，DeepSpeed MoE 需要）
    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        import socket as _socket
        with _socket.socket() as _s:
            _s.bind(('', 0))
            _free_port = str(_s.getsockname()[1])
        os.environ.setdefault("MASTER_PORT", _free_port)
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
        deepspeed.init_distributed(dist_backend="nccl")
        print("[INFO] Initialized distributed backend for DeepSpeed MoE")

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    # ============================================================
    # 1. Tokenizer
    # ============================================================
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length,
        padding_side="right", use_fast=False, legacy=True)
    args.seg_token_idx = tokenizer("<SEG>", add_special_tokens=False).input_ids[0]
    conversation_lib.default_conversation = conversation_lib.conv_templates['v1']

    # ============================================================
    # 2. 加载 base model (bf16, 不量化)
    # ============================================================
    print("[1/4] Loading base model (bf16) ...")
    model_args = vars(args)
    model_args['test_only'] = True
    model = MedPLIBForCausalLM.from_pretrained(
        args.version,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True,
        device_map="auto",
        **model_args,
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    # ============================================================
    # 3. 加载 LoRA adapter 并 merge
    # ============================================================
    print(f"[2/4] Loading LoRA adapter from {args.lora_path} ...")
    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()
    print("  LoRA merged into base model.")

    # Bypass SGCAFE if requested
    if args.bypass_sgcafe:
        model.config.bypass_sgcafe = True
        print("  [INFO] SGCAFE bypassed — only Region ICL active.")

    # ============================================================
    # 4. 加载 SGCAFE 权重
    # ============================================================
    sgcafe_path = os.path.join(args.lora_path, "sgcafe.pt")
    if not args.bypass_sgcafe and os.path.exists(sgcafe_path):
        print(f"[3/4] Loading SGCAFE weights from {sgcafe_path} ...")
        sgcafe_state = torch.load(sgcafe_path, map_location="cpu")
        model.get_model().sgcafe.load_state_dict(sgcafe_state)
        model.get_model().sgcafe._trained = True
        print("  SGCAFE weights loaded.")
    else:
        print("[3/4] SGCAFE skipped (bypassed or no sgcafe.pt found).")

    # 加载 region_fea_adapter + text_hidden_fcs（V3 训练产生）
    adapters_path = os.path.join(args.lora_path, "adapters.pt")
    if os.path.exists(adapters_path):
        print(f"  Loading adapter weights from {adapters_path} ...")
        adapters_state = torch.load(adapters_path, map_location="cpu")
        try:
            model.get_model().region_fea_adapter.load_state_dict(adapters_state['region_fea_adapter'])
            model.get_model().text_hidden_fcs.load_state_dict(adapters_state['text_hidden_fcs'])
            print("  region_fea_adapter + text_hidden_fcs loaded.")
        except RuntimeError as e:
            print(f"  [WARN] Adapter shape mismatch (4-bit quantized checkpoint), skipping: {e}")
            print("  Using pretrained weights for region_fea_adapter + text_hidden_fcs.")

    # ============================================================
    # 5. 移除 accelerate hooks，统一放到 cuda:0
    # ============================================================
    print("[4/4] Setting up modules on cuda:0 ...")
    model_engine = model
    input_device = torch.device("cuda:0")

    vision_tower = model.get_model().get_vision_tower()
    remove_hook_from_submodules(vision_tower)
    vision_tower.to(dtype=torch_dtype, device="cuda:0")

    visual_model = model.get_model().visual_model
    remove_hook_from_submodules(visual_model)
    visual_model.to(dtype=torch_dtype, device="cuda:0")

    text_hidden_fcs = model.get_model().text_hidden_fcs
    remove_hook_from_submodules(text_hidden_fcs)
    text_hidden_fcs.to(dtype=torch_dtype, device="cuda:0")

    region_fea_adapter = model.get_model().region_fea_adapter
    remove_hook_from_submodules(region_fea_adapter)
    region_fea_adapter.to(dtype=torch_dtype, device="cuda:0")

    if not args.bypass_sgcafe:
        sgcafe = model.get_model().sgcafe
        remove_hook_from_submodules(sgcafe)
        sgcafe.to(dtype=torch_dtype, device="cuda:0")
        with torch.no_grad():
            test_input = torch.zeros(1, 576, sgcafe.clip_dim, dtype=torch_dtype, device="cuda:0")
            gate_val = torch.sigmoid(sgcafe.gate_proj(test_input))
            print(f"  SGCAFE gate (trained): mean={gate_val.mean().item():.8f}, max={gate_val.max().item():.8f}")

    print(f"  region_fea_adapter dtype={region_fea_adapter.weight.dtype}, shape={tuple(region_fea_adapter.weight.shape)}")
    for i, seq in enumerate(text_hidden_fcs):
        for j, layer in enumerate(seq):
            if hasattr(layer, "weight"):
                print(f"  text_hidden_fcs[{i}][{j}] dtype={layer.weight.dtype}, shape={tuple(layer.weight.shape)}")

    for name, param in model.named_parameters():
        param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable_params} trainable / {total_params} total")

    # ============================================================
    # 6. Dataset & DataLoader
    # ============================================================
    data_args = types.SimpleNamespace(
        image_folder=args.image_folder,
        image_aspect_ratio=args.image_aspect_ratio,
        is_multimodal=args.is_multimodal,
        mm_use_im_start_end=args.use_mm_start_end,
        image_processor=vision_tower.image_processor,
    )
    val_dataset = LazySupervisedDataset(
        args.val_data_path, tokenizer, data_args, args.sam_img_size,
        use_visual_icl=args.use_visual_icl,
        support_pool_path=args.support_pool_path,
        use_token_icl_concat=args.use_token_icl_concat,
        use_token_icl_multi=args.use_token_icl_multi,
        use_region_icl=args.use_region_icl,
        region_icl_self=args.region_icl_self,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False, drop_last=False,
        collate_fn=partial(DataCollatorForSupervisedDataset, inference=True),
    )
    print(f"  Validation dataset size: {len(val_dataset)}")

    # ============================================================
    # 7. Run evaluation
    # ============================================================
    model_engine.eval()
    if args.eval_seg:
        validate_seg(val_loader, model_engine, 0, args, tokenizer, input_device=input_device)
    elif args.eval_vqa:
        validate_vqa(val_loader, model_engine, 0, args, tokenizer, input_device=input_device)
    else:
        print("Please specify --eval_seg or --eval_vqa")

def validate_seg(val_loader, model_engine, epoch, args, tokenizer, input_device="cuda"):
    iou_meter = AverageMeter("IoU", ":6.3f", Summary.SUM)
    dice_meter = AverageMeter("Dice", ":6.3f", Summary.SUM)
    modality_metric = {}

    pbar = tqdm.tqdm(val_loader)
    idx = 0
    for input_dict in pbar:
        modality = input_dict["image_paths"][0].split('/')[-1].split('_')[0]
        if modality not in modality_metric:
            modality_metric[modality] = {'iou': [], 'dice': []}

        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict, device=input_device)

        # 精度转换
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            if isinstance(input_dict["images_clip"], list):
                input_dict["images_clip"] = [x.half() for x in input_dict["images_clip"]]
            else:
                input_dict["images_clip"] = input_dict["images_clip"].half()
            if "support_clip" in input_dict:
                input_dict["support_clip"] = input_dict["support_clip"].half()
            if "icl_region_clip" in input_dict:
                input_dict["icl_region_clip"] = input_dict["icl_region_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            if isinstance(input_dict["images_clip"], list):
                input_dict["images_clip"] = [x.bfloat16() for x in input_dict["images_clip"]]
            else:
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            if "support_clip" in input_dict:
                input_dict["support_clip"] = input_dict["support_clip"].bfloat16()
            if "icl_region_clip" in input_dict:
                input_dict["icl_region_clip"] = input_dict["icl_region_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            if isinstance(input_dict["images_clip"], list):
                input_dict["images_clip"] = [x.float() for x in input_dict["images_clip"]]
            else:
                input_dict["images_clip"] = input_dict["images_clip"].float()
            if "support_clip" in input_dict:
                input_dict["support_clip"] = input_dict["support_clip"].float()
            if "icl_region_clip" in input_dict:
                input_dict["icl_region_clip"] = input_dict["icl_region_clip"].float()

        indices = (input_dict['input_ids'] == 29901).nonzero(as_tuple=True)
        input_ids = input_dict['input_ids'][:, :indices[1][-1]+1]
        attention_mask = input_dict['attention_mask'][:, :indices[1][-1]+1]

        is_token_icl = isinstance(input_dict["images_clip"], torch.Tensor) and input_dict["images_clip"].ndim == 5
        with torch.no_grad():
            output_ids, pred_masks = model_engine.evaluate(
                input_dict["images_clip"],
                input_dict["images"],
                input_ids,
                input_dict['resize_list'],
                input_dict['label_list'],
                region_masks=input_dict.get('region_masks', []),
                valid_region_masks_bool=input_dict.get('valid_region_masks_bool', []),
                max_new_tokens=1024,
                tokenizer=tokenizer,
                attention_mask=attention_mask,
                support_images=None if (is_token_icl or args.bypass_sgcafe) else input_dict.get('support_clip', None),
                support_mask_weights=None if (is_token_icl or args.bypass_sgcafe) else input_dict.get('support_mask_weights', None),
                icl_region_clip=input_dict.get('icl_region_clip', None),
            )

        masks_list = input_dict["masks_list"][0].int().unsqueeze(0)
        if len(pred_masks) == 0:
            iou = 0.0
        else:
            output_list = (torch.sigmoid(pred_masks[0]) > 0.1).int()
            for mask_i, output_i in zip(masks_list, output_list):
                mask_i = mask_i.unsqueeze(0)
                output_i = output_i.unsqueeze(0)
                iou = calculate_iou(output_i, mask_i)

        iou_meter.update(iou)
        dice = 2 * iou / (1 + iou)
        dice_meter.update(dice)
        modality_metric[modality]['iou'].append(iou)
        modality_metric[modality]['dice'].append(dice)
        pbar.set_description(f'iou={iou:.4f}')
        idx += 1

        if args.vis_mask:
            lora_name = os.path.basename(args.lora_path)
            test_file_name = args.val_data_path.split('/')[-2]
            img_name = os.path.basename(input_dict['image_paths'][0])[:-4]
            save_path = os.path.join('./runs/qlora_infer', lora_name, test_file_name,
                                     img_name + '_iou' + str(round(iou, 4)) + '.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis_overlay_masks(input_dict['image_paths'][0], output_i, mask_i, save_path)

    miou = iou_meter.avg
    mDice = dice_meter.avg
    print(f"\nmiou: {miou:.6f}, mDice: {mDice:.6f}")

    modality_metric_res = {}
    for mod in modality_metric:
        modality_metric_res[mod] = {
            'iou': round(sum(modality_metric[mod]['iou']) / len(modality_metric[mod]['iou']), 6),
            'dice': round(sum(modality_metric[mod]['dice']) / len(modality_metric[mod]['dice']), 6),
        }
    print(modality_metric_res)
    return miou, mDice

def validate_vqa(val_loader, model_engine, epoch, args, tokenizer, input_device="cuda"):
    import shortuuid
    lora_name = os.path.basename(args.lora_path)
    test_file_name = os.path.splitext(os.path.basename(args.val_data_path))[0]
    answers_dir = os.path.join('./runs/qlora_infer', lora_name, 'vqa_res')
    os.makedirs(answers_dir, exist_ok=True)
    answers_file = os.path.join(answers_dir, test_file_name + '.jsonl')
    ans_file = open(answers_file, "a")

    pbar = tqdm.tqdm(val_loader)
    idx = 0
    for input_dict in pbar:
        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict, device=input_device)

        if args.precision == "fp16":
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images_clip"] = input_dict["images_clip"].float()

        indices = (input_dict['input_ids'] == 29901).nonzero(as_tuple=True)
        input_ids = input_dict['input_ids'][:, :indices[1][-1]+1]
        attention_mask = input_dict['attention_mask'][:, :indices[1][-1]+1]

        with torch.no_grad():
            output_ids = model_engine.generate(
                input_ids,
                images=input_dict['images_clip'],
                attention_mask=attention_mask,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True)

        input_token_len = input_ids.shape[1]
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()

        ans_file.write(json.dumps({
            "question_id": idx,
            "image_path": input_dict['image_paths'][0],
            "prompt": input_dict['questions_list'][0][0],
            "gt": input_dict['gts_list'][0][0],
            "text": outputs,
            "answer_id": shortuuid.uuid(),
            "answer_type": args.answer_type,
            "metadata": {}
        }) + "\n")
        ans_file.flush()
        idx += 1
    ans_file.close()
    print(f"VQA results saved to: {answers_file}")


if __name__ == "__main__":
    main(sys.argv[1:])
