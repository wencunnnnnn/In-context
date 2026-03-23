环境安装（Installation）
git clone https://github.com/ShawnHuang497/MedPLIB.git
cd MedPLIB


创建虚拟环境并安装依赖：

conda create -n medplib python=3.10 -y
conda activate medplib
pip install --upgrade pip 
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118


用于训练的额外包：

pip install ninja==1.11.1.1
pip install flash-attn==2.5.2 --no-build-isolation

🗃️ 数据集（Dataset）

从 Google Drive
 下载 MeCoVQA 数据集。

从 Hugging Face
 下载 SA-Med2D-20M 图像数据集。

📀 模型训练（Train）
阶段 I

进行预训练阶段 I 以获得投影器（projector）权重。
请参考 LLaVA-Med
 和 LLaVA-v1.5
。

阶段 II
sh scripts/train_stage2.sh

阶段 III
sh scripts/train_stage3.sh

阶段 IV
sh scripts/train_stage4.sh

🥭 模型库（Model Zoo）

从 Hugging Face
 下载预训练权重。

🧪 模型测试（Test）
1️⃣ 像素级问答（Pixel Grounding）
TRANSFORMERS_OFFLINE=1 deepedia/userdisk8/zhli/MedPLIB/checkpoints" \
  --vision_tower="speed --include=localhost:1 --master_port=64995 model/eval/vqa_infer.py \
    --version="/path/to/the/medplib_checkpoints" \
    --vision_tower='/path/to/the/clip-vit-large-patch14-336' \
    --answer_type='open' \
    --val_data_path='/path/to/the/pixel_grounding_json_file' \
    --image_folder='/path/to/the/SAMed2D_v1' \
    --vision_pretrained="/path/to/the/sam-med2d_b.pth" \
    --eval_seg \
    --moe_enable \
    --region_fea_adapter
    # --vis_mask

2️⃣ 区域问答与普通 VQA

生成预测结果：

sh model/eval/infer_parallel_medplib.sh


计算评价指标：

python model/eval/cal_metric.py \
    --pred="/path/to/the/jsonl_file"


export CUDA_VISIBLE_DEVICES=0,1,3,4,5 && \
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && \
TRANSFORMERS_OFFLINE=1 \
deepspeed --include=localhost:0,1,2,3,4 --master_port=64995 \
model/eval/vqa_infer.py \
--version="/media/userdisk8/zhli/MedPLIB/checkpoints" \
--vision_tower="/media/userdisk8/zhli/MedPLIB/clip-vit-large-patch14" \
--answer_type="open" \
--val_data_path="/media/userdisk8/zhli/MedPLIB/data/MeCoVQA/test/MeCoVQA_Grounding_test.json" \
--image_folder="/media/userdisk8/zhli/MedPLIB/data/SA-Med2D-20M/SA-Med2D-16M/SAMed2Dv1" \
--vision_pretrained="/media/userdisk8/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
--eval_seg \
--moe_enable \
--region_fea_adapter

TRANSFORMERS_OFFLINE=5 deepspeed --include=localhost:0,1 --master_port=64995 model/eval/vqa_infer.py \
--version="/media/userdisk8/zhli/MedPLIB/checkpoints" \
--vision_tower="/media/userdisk8/zhli/MedPLIB/clip-vit-large-patch14" \
--answer_type="open" \
--val_data_path="/media/userdisk8/zhli/MedPLIB/data/MeCoVQA/test/MeCoVQA_Grounding_test.json" \
--image_folder="/media/userdisk8/zhli/MedPLIB/data/SA-Med2D-20M/SA-Med2D-16M/SAMed2Dv1" \
--vision_pretrained="/media/userdisk8/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
--eval_seg \
--moe_enable \
--region_fea_adapter



TRANSFORMERS_OFFLINE=1  deepspeed --include=localhost:0,1,2 --master_port=64995 model/eval/vqa_infer.py \
--version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
--vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
--answer_type="open" \
--val_data_path="/media/userdisk2/zhli/data/MeCoVQA/test/MeCoVQA_Grounding_test.json" \
--image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
--vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
--eval_seg \
--moe_enable \
--region_fea_adapter




export CUDA_VISIBLE_DEVICES=0,2,5 && \
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && \
TRANSFORMERS_OFFLINE=1 \
python model/eval/vqa_infer.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --answer_type="open" \
  --val_data_path="/media/userdisk2/zhli/data/MeCoVQA/test/MeCoVQA_Grounding_test.json" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --eval_seg \
  --moe_enable \
  --region_fea_adapter




export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

export CUDA_VISIBLE_DEVICES=0,1,2,3,4 && \
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 && \
TRANSFORMERS_OFFLINE=1 \
python model/eval/vqa_infer.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --answer_type="open" \
  --val_data_path="/media/userdisk2/zhli/data/MeCoVQA/test/MeCoVQA_Grounding_test.json" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --eval_seg \
  --moe_enable \
  --region_fea_adapter \
  --use_visual_icl \
  --support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"


