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



模态	原文 Dice	复现 Dice	差距
Dermoscopy	79.90	79.87	-0.03
PET	        64.59	64.52	-0.07
CT	        59.83	59.76	-0.07
Overall   	68.11	68.05	-0.06



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

仅仅加入ICL 就是CLIP 将两个图编码成token  query和suoport（原图和mask拼接之后）统一输入到LLM中
并且将文本改成了上下文的模式
The first image shows an example segmentation highlighted in green. 
Please segment the same target in the second image.


模态	Baseline Dice	第一轮 ICL（bug版）	第二轮 ICL（修复后）	差异（修复后 vs Baseline）
Dermoscopy	79.87	80.58	80.56	+0.69
PET	        64.52 64.49	64.53	+0.01
CT	        59.76 59.33	59.32	-0.44
Overall	    68.05 68.13	68.14	+0.09




TRANSFORMERS_OFFLINE=1 python model/eval/vqa_infer.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --val_data_path="/media/userdisk2/zhli/data/MeCoVQA/zeroshot4grounding/endoscopy.json" \   
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --eval_seg \
  --moe_enable \
  --region_fea_adapter
  --use_visual_icl   #加入ICL策略

零样本推理
 cd /media/userdisk2/zhli/MedPLIB && bash run_zeroshot.sh

 
Zero-shot

      X-Ray  End  MR    US   FP          
原文  28.25 44.19 27.52 35.64 25.76 
复现  8.47 44.85  27.41  34.75 5.44          
ICL  6.96  41.84  27.18  34.75  13.15       
ICLA  7.89  35.07  27.12  0.66  11.93  
ICLB  7.19  43.13  27.86  23.03  0.092

方案A — 横向拼接（2图输入LLM）：


--use_token_icl_concat \
--support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"


<image> <image> The first image is a reference (left: scan, right: mask). For the second image, please generate a mask that segments the hepatic tumor in this image by referring to the example.

{'fundus': {'iou': 0.084957, 'dice': 0.120292}}
{'endoscopy': {'iou': 0.260238, 'dice': 0.349663}}
{'ultrasound': {'iou': 0.003881, 'dice': 0.006782}}


==========================================
        零样本 ICL 推理结果汇总
==========================================
Modality        | mIoU       | mDice     
-------------------------------------------
endoscopy       | 0.262252   | 0.350701  
fundus          | 0.084309   | 0.119344  
mr              | 0.182348   | 0.271288  
pet             | 0.503065   | 0.631061  
ultrasound      | 0.003787   | 0.006622  
xray            | 0.050591   | 0.078821  
==========================================

方案B — 三图注入（3图输入LLM）：
--use_token_icl_multi \
--support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"

<image> <image> <image> Image 1 is a reference scan and Image 2 is its target mask. Based on this context, for Image 3, please generate a mask that segments the hepatic tumor in this image.

iou=0.5499768853187561.: 100%|█████████████████████████████████| 2344/2344 [1:37:37<00:00,  2.50s/it]
miou: 0.544151, mDice: 0.660323
{'pet': {'iou': 0.512706, 'dice': 0.642954}, 'ct': {'iou': 0.485471, 'dice': 0.593416}, 'dermoscopy': {'iou': 0.692637, 'dice': 0.803829}}

{'fundus': {'iou': 0.000358, 'dice': 0.000695}}
{'endoscopy': {'iou': 0.327643, 'dice': 0.43494}}
{'ultrasound': {'iou': 0.149854, 'dice': 0.228144}}

==========================================
        零样本 ICL 推理结果汇总
==========================================
Modality        | mIoU       | mDice     
-------------------------------------------
endoscopy       | 0.323611   | 0.431335  
fundus          | 0.000480   | 0.000923  
mr              | 0.186870   | 0.278617  
pet             | 0.503338   | 0.631361  
ultrasound      | 0.151174   | 0.230262  
xray            | 0.046812   | 0.071899  
==========================================

cd /media/userdisk2/zhli/MedPLIB
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TRANSFORMERS_OFFLINE=1 python model/eval/vqa_infer.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --answer_type="open" \
  --val_data_path="/media/userdisk2/zhli/data/MeCoVQA/zeroshot4grounding/ultrasound.json" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --eval_seg \
  --moe_enable \
  --region_fea_adapter \
  --use_token_icl_concat \
  --support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"

cd /media/userdisk2/zhli/MedPLIB
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
TRANSFORMERS_OFFLINE=1 python model/eval/vqa_infer.py \
  --version="/media/userdisk2/zhli/MedPLIB/checkpoints" \
  --vision_tower="/media/userdisk2/zhli/MedPLIB/clip-vit-large-patch14" \
  --answer_type="open" \
  --val_data_path="/media/userdisk2/zhli/data/MeCoVQA/zeroshot4grounding/xray.json" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --vision_pretrained="/media/userdisk2/zhli/MedPLIB/SAM/sam-med2d_b.pth" \
  --eval_seg \
  --moe_enable \
  --region_fea_adapter \
  --use_region_icl  \
  --support_pool_path="/media/userdisk2/zhli/MedPLIB/support_pool.pkl"



iou=0.5591924786567688.: 100%|██████████████████████████████| 2344/2344 [44:08<00:00,  1.13s/it]
miou: 0.543676, mDice: 0.660174
{'pet': {'iou': 0.515161, 'dice': 0.645455}, 'ct': {'iou': 0.486306, 'dice': 0.594091}, 'dermoscopy': {'iou': 0.685739, 'dice': 0.798515}}


  
           X-Ray  End  MR    US   FP          
原文       28.25 44.19 27.52 35.64 25.76 
复现       8.47 44.85  27.41  34.75 5.44 
regionicl  9.82 42.49  24.30  37.23 4.46

==========================================
        零样本 ICL 推理结果汇总
==========================================
Modality        | mIoU       | mDice     
-------------------------------------------
endoscopy       | 0.316958   | 0.424916  
fundus          | 0.032065   | 0.044612  
mr              | 0.161331   | 0.243047  
pet             | 0.514041   | 0.644287  
ultrasound      | 0.264542   | 0.372381  
xray            | 0.063978   | 0.098271  
==========================================





# 创建一个命名会话
screen -S train

# 在里面运行训练
cd /media/userdisk2/zhli/MedPLIB
bash scripts/train_region_icl_qlora.sh

# 断开（不会停止训练）：按 Ctrl+A 然后按 D

# 重新连接
screen -r train



cd /media/userdisk2/zhli/MedPLIB

# 方式1：运行全部模态
bash scripts/eval_qlora.sh

# 方式2：单独运行测试集
python model/eval/vqa_infer_qlora.py \
  --version="checkpoints" \
  --lora_path="runs/region_icl_qlora/epoch_3" \
  --vision_tower="clip-vit-large-patch14" \
  --vision_pretrained="SAM/sam-med2d_b.pth" \
  --image_folder="/media/userdisk2/zhli/data/SAM-MED-20M/SA-Med2D-16M/SAMed2Dv1" \
  --val_data_path="/media/userdisk2/zhli/data/MeCoVQA/zeroshot4grounding/fundus.json" \
  --support_pool_path="support_pool.pkl" \
  --eval_seg --moe_enable --region_fea_adapter --use_region_icl


iou=0.5824: 100%|█████████████████████████████████████| 2344/2344 [1:07:49<00:00,  1.74s/it]

miou: 0.494035, mDice: 0.602264
{'pet': {'iou': 0.509274, 'dice': 0.639663}, 'ct': {'iou': 0.370357, 'dice': 0.461148}, 'dermoscopy': {'iou': 0.693225, 'dice': 0.801608}}



我一开始的想法是SFCAFE是促进query和support空间信息的交互，但是虽然在这里进行了空间信息的交互，但是经过了巨大的LLM层，这里的空间信息被压缩的几乎没有了？
所以现在实验结果不好的原因可能是SGCAFE随机更改掉了query的特征，但是按照理论来说，Region ICL应该很有作用？  ICL在这里就是另一种的文本去双重加强，但是原来模型没有用过这种prompt，我们用的prompt是：这个区域是个实例，请分割XXX，我lora微调的想法是让模型学习到同时使用两个专家的能力
我觉得只用region ICL这样效果还不好，就需要训练一个adapter，进入LLM之前将ICL投影成可学习的信息？  目前这个Qlora微调的意义是什么


miou: 0.543471, mDice: 0.659974
{'pet': {'iou': 0.514241, 'dice': 0.644548}, 'ct': {'iou': 0.485965, 'dice': 0.593755}, 'dermoscopy': {'iou': 0.686772, 'dice': 0.79954}}
All evaluations complete!
Compare results in: runs/qlora_infer_v2/