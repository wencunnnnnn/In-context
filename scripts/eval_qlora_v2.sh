#!/bin/bash
# V2 QLoRA Region ICL 推理对比实验
# 实验1: 带 SGCAFE（baseline）
# 实验2: bypass SGCAFE，只保留 Region ICL

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_OFFLINE=1

BASE_MODEL="/media/userdisk2/zhli/MedPLIB/checkpoints"
LORA_PATH="/media/userdisk2/zhli/MedPLIB/runs/region_icl_qlora_v2/step_6000"
VISION_TOWER="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14"
SAM_CKPT="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth"
IMAGE_FOLDER="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1"
SUPPORT_POOL="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"
VAL_DATA="/media/userdisk2/zhli/data/MeCoVQA/test/MeCoVQA_Grounding_test.json"

OUTPUT_DIR="runs/qlora_infer_v2"
mkdir -p ${OUTPUT_DIR}

# ============================================================
# 实验1: 带 SGCAFE
# ============================================================
echo "####################################################"
echo "# Experiment 1: V2 with SGCAFE"
echo "####################################################"
python model/eval/vqa_infer_qlora.py \
    --version="${BASE_MODEL}" \
    --lora_path="${LORA_PATH}" \
    --vision_tower="${VISION_TOWER}" \
    --vision_pretrained="${SAM_CKPT}" \
    --image_folder="${IMAGE_FOLDER}" \
    --val_data_path="${VAL_DATA}" \
    --support_pool_path="${SUPPORT_POOL}" \
    --answer_type="open" \
    --eval_seg \
    --moe_enable \
    --region_fea_adapter \
    --use_visual_icl \
    --use_region_icl \
    --sam_img_size=256 \
    --model_max_length=512 \
    2>&1 | tee "${OUTPUT_DIR}/eval_with_sgcafe.log"

# ============================================================
# 实验2: Bypass SGCAFE，只保留 Region ICL
# ============================================================
echo "####################################################"
echo "# Experiment 2: V2 without SGCAFE (bypass)"
echo "####################################################"
python model/eval/vqa_infer_qlora.py \
    --version="${BASE_MODEL}" \
    --lora_path="${LORA_PATH}" \
    --vision_tower="${VISION_TOWER}" \
    --vision_pretrained="${SAM_CKPT}" \
    --image_folder="${IMAGE_FOLDER}" \
    --val_data_path="${VAL_DATA}" \
    --support_pool_path="${SUPPORT_POOL}" \
    --answer_type="open" \
    --eval_seg \
    --moe_enable \
    --region_fea_adapter \
    --use_visual_icl \
    --use_region_icl \
    --bypass_sgcafe \
    --sam_img_size=256 \
    --model_max_length=512 \
    2>&1 | tee "${OUTPUT_DIR}/eval_no_sgcafe.log"

echo "All evaluations complete!"
echo "Compare results in: ${OUTPUT_DIR}/"
