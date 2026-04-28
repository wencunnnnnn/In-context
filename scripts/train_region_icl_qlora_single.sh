#!/bin/bash
# 单卡训练版本（仅 GPU 0），其余逻辑与 train_region_icl_qlora.sh 完全一致

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TRANSFORMERS_OFFLINE=2

export CUDA_VISIBLE_DEVICES=2

mkdir -p runs/region_icl_qlora_v3_single1

torchrun --nproc_per_node=1 --master_port=29501 \
  train_region_icl_qlora.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --data_path="/media/userdisk2/zhli/data/MeCoVQA/train/MeCoVQA-Grounding.json" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl" \
  --expert_pretrained_path="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --run_name="region_icl_qlora_v3_single1" \
  --epochs=3 \
  --batch_size=1 \
  --grad_accumulation_steps=4 \
  --lr=5e-5 \
  --sgcafe_lr=1e-3 \
  --lora_r=8 \
  --lora_alpha=16 \
  --lora_target_modules="q_proj,v_proj" \
  --model_max_length=512 \
  --sam_img_size=256 \
  --ce_loss_weight=1.0 \
  --dice_loss_weight=0.5 \
  --bce_loss_weight=2.0 \
  --align_loss_weight=3.0 \
  --seg_weak_loss_weight=0.5 \
  --print_freq=20 \
  --save_steps=500 \
  --bypass_sgcafe \
  2>&1 | tee runs/region_icl_qlora_v3_single1/train.log
