#!/bin/bash
# QLoRA 训练 v3：Region ICL，解冻 region_fea_adapter + text_hidden_fcs
# 相比 v2：去掉 SGCAFE，解冻 adapter 让分割头适配 region 增强后的 [SEG] 表示
# 多卡训练：torchrun 启动 4 GPU，手动 all-reduce 梯度

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export TRANSFORMERS_OFFLINE=0
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# 4x RTX 4070 Ti Super (16GB, compute 8.9, 支持 bf16)
export CUDA_VISIBLE_DEVICES=0,1,2,3

mkdir -p runs/region_icl_qlora_v3

torchrun --nproc_per_node=4 --master_port=29500 \
  train_region_icl_qlora.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --data_path="/media/userdisk2/zhli/data/MeCoVQA/train/MeCoVQA-Grounding.json" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl" \
  --expert_pretrained_path="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --run_name="region_icl_qlora_v3" \
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
  2>&1 | tee runs/region_icl_qlora_v3/train.log
