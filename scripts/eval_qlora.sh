#!/bin/bash
# QLoRA Region ICL 推理脚本
# 加载 base model + merge LoRA + 加载 SGCAFE，然后评估分割性能
# 不影响原始 vqa_infer.py

export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_OFFLINE=1

BASE_MODEL="/media/userdisk2/zhli/MedPLIB/checkpoints"
LORA_PATH="/media/userdisk2/zhli/MedPLIB/runs/region_icl_qlora/epoch_3"
VISION_TOWER="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14"
SAM_CKPT="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth"
IMAGE_FOLDER="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1"
SUPPORT_POOL="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"
ZEROSHOT_DIR="/media/userdisk2/zhli/data/MeCoVQA/zeroshot4grounding"

# 评估所有 zero-shot 模态
for modality in fundus endoscopy ultrasound xray mr pet; do
    echo "=========================================="
    echo "Evaluating: ${modality}"
    echo "=========================================="
    python model/eval/vqa_infer_qlora.py \
        --version="${BASE_MODEL}" \
        --lora_path="${LORA_PATH}" \
        --vision_tower="${VISION_TOWER}" \
        --vision_pretrained="${SAM_CKPT}" \
        --image_folder="${IMAGE_FOLDER}" \
        --val_data_path="${ZEROSHOT_DIR}/${modality}.json" \
        --support_pool_path="${SUPPORT_POOL}" \
        --answer_type="open" \
        --eval_seg \
        --moe_enable \
        --region_fea_adapter \
        --use_region_icl \
        --sam_img_size=256 \
        --model_max_length=512 \
        2>&1 | tee "runs/qlora_infer/eval_${modality}.log"
    echo ""
done

echo "All evaluations complete!"
