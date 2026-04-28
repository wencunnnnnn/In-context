#!/bin/bash
# 零样本推理脚本（带 ICL）：从同一测试集中选 support 示例

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TRANSFORMERS_OFFLINE=1

COMMON_ARGS="
  --version=/media/userdisk2/zhli/MedPLIB/checkpoints
  --vision_tower=/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14
  --answer_type=open
  --image_folder=/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1
  --vision_pretrained=/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth
  --eval_seg
  --moe_enable
  --region_fea_adapter
  --use_token_icl_multi
"

ZEROSHOT_DIR="/media/userdisk2/zhli/data/MeCoVQA/zeroshot4grounding"
LOG_DIR="/media/userdisk2/zhli/MedPLIB/zeroshot_results_icl_concat"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "MedPLIB 零样本推理 (Zero-shot ICL方案B)"
echo "=========================================="

for modality in endoscopy fundus mr pet ultrasound xray; do
    echo ""
    echo ">>> 推理模态: ${modality}"
    echo "---"
    python model/eval/vqa_infer.py \
        $COMMON_ARGS \
        --val_data_path="${ZEROSHOT_DIR}/${modality}.json" 2>&1 | tee "${LOG_DIR}/${modality}.log"
    echo "<<< ${modality} 完成"
    echo ""
done

echo ""
echo "=========================================="
echo "        零样本 ICL 推理结果汇总"
echo "=========================================="
printf "%-15s | %-10s | %-10s\n" "Modality" "mIoU" "mDice"
echo "-------------------------------------------"

for modality in endoscopy fundus mr pet ultrasound xray; do
    log="${LOG_DIR}/${modality}.log"
    if [ -f "$log" ]; then
        miou=$(grep -oP 'miou: \K[0-9.]+' "$log" | tail -1)
        mdice=$(grep -oP 'mDice: \K[0-9.]+' "$log" | tail -1)
        printf "%-15s | %-10s | %-10s\n" "$modality" "${miou:-N/A}" "${mdice:-N/A}"
    else
        printf "%-15s | %-10s | %-10s\n" "$modality" "N/A" "N/A"
    fi
done

echo "=========================================="
