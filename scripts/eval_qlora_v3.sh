#!/bin/bash
# V3 QLoRA Region ICL 推理（bypass SGCAFE，解冻 region_fea_adapter + text_hidden_fcs）

export CUDA_VISIBLE_DEVICES=4,5,6,7
export TRANSFORMERS_OFFLINE=1

BASE_MODEL="/media/userdisk2/zhli/MedPLIB/checkpoints"
RUN_DIR="/media/userdisk2/zhli/MedPLIB/runs/region_icl_qlora_v3"
if [ -z "${LORA_PATH:-}" ]; then
    LORA_PATH=$(find "${RUN_DIR}" -maxdepth 1 -type d -name 'step_*' | sort -V | tail -n 1)
fi
VISION_TOWER="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14"
SAM_CKPT="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth"
IMAGE_FOLDER="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1"
SUPPORT_POOL="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"
VAL_DATA="/media/userdisk2/zhli/data/MeCoVQA/test/MeCoVQA_Grounding_test.json"

OUTPUT_DIR="runs/qlora_infer_v3"
mkdir -p ${OUTPUT_DIR}

if [ -z "${LORA_PATH}" ]; then
    echo "No V3 checkpoint found under ${RUN_DIR}"
    exit 1
fi

STEP_NAME=$(basename "${LORA_PATH}")
LOG_PATH="${OUTPUT_DIR}/eval_v3_${STEP_NAME}.log"

echo "Using checkpoint: ${LORA_PATH}"

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
    2>&1 | tee "${LOG_PATH}"

echo "Done. Results in: ${LOG_PATH}"
