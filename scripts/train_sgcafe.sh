#!/bin/bash
# SGCAFE 训练：只训练 SGCAFE 模块（~2M 参数），冻结所有其他参数
# 8 卡 DeepSpeed ZeRO-3 + CPU offload

time=$(date +%Y-%m-%d-%H-%M-%S)
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64

exp_name="sgcafe-train"
exp_dir="runs/$exp_name"
mkdir -p "$exp_dir"

TRANSFORMERS_OFFLINE=1 deepspeed \
  --include=localhost:0,1,2,3,4,5,6,7 \
  --master_port=24999 \
  train_ds_medplib.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --data_path="/media/userdisk2/zhli/data/MeCoVQA/train/MeCoVQA-Grounding.json" \
  --val_data_path="/media/userdisk2/zhli/data/MeCoVQA/test/MeCoVQA_Grounding_test.json" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --exp_name=$exp_name \
  --epochs=5 \
  --batch_size=1 \
  --workers=1 \
  --image_aspect_ratio='pad' \
  --is_multimodal=True \
  --model_max_length 256 \
  --grad_accumulation_steps 16 \
  --out_dim 256 \
  --ce_loss_weight 1.0 \
  --dice_loss_weight 0.5 \
  --bce_loss_weight 2.0 \
  --focal_loss_weight 0 \
  --iou_loss_weight 0 \
  --lora_r 0 \
  --sft_modules "sgcafe" \
  --lr 0.0003 \
  --save_steps 500 \
  --sam_img_size 256 \
  --precision "bf16" \
  --moe_enable true \
  --moe_mode dense \
  --num_experts 2 \
  --capacity_factor 1.5 \
  --region_fea_adapter \
  --top_k_experts 1 \
  --router_aux_loss_coef 0.0 \
  --expert_pretrained_path "/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --training_stage 1 \
  --use_visual_icl \
  --support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl" \
  --align_loss_weight 1.0 \
  --seg_weak_loss_weight 0.1 \
  --no_eval \
  --print_freq 10 \
  2>&1|tee -a runs/$exp_name/$time.log
