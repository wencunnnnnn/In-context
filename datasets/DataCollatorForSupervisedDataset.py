
import transformers
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

from utils.utils import IGNORE_INDEX



def DataCollatorForSupervisedDataset(list_data_dict: Sequence[Dict], inference: bool = False) -> Dict[str, torch.Tensor]:
    tokenizer = list_data_dict[0]['tokenizer']
    input_ids, labels = tuple([instance[key] for instance in list_data_dict]
                                for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                batch_first=True,
                                                padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :tokenizer.model_max_length]
    labels = labels[:, :tokenizer.model_max_length]
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    seg_flag = False
    max_mask_nums = max([len(instance['masks']) for instance in list_data_dict])
    #### fullfill mask instances according to the max_mask_nums in this batch
    masks = []
    label_list = []
    valid_mask_bool = []
    resize_list = []
    if max_mask_nums > 0:
        seg_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['masks'])>0:
                masks.extend(instance['masks'])
                label_list.extend(instance['label'])
                resize_list.extend(instance['resize'])
                valid_mask_bool.append([True]* len(instance['masks']))
            else:
                valid_mask_bool.append([])

    batch['masks'] = masks
    batch['valid_mask_bool'] = valid_mask_bool
    batch['label_list'] = label_list
    batch['resize_list'] = resize_list


    region_masks = []
    valid_region_masks_bool = []
    max_region_masks_nums = max([len(instance['region_masks']) for instance in list_data_dict])

    rp_flag = False
    if max_region_masks_nums > 0:
        rp_flag = True
        for idx, instance in enumerate(list_data_dict):
            if len(instance['region_masks'])>0:
                region_masks.extend(instance['region_masks'])
                valid_region_masks_bool.append([torch.ones(1).bool()]* len(instance['region_masks']))
            else:
                valid_region_masks_bool.append([torch.zeros(1).bool()])

    image_path_list = []
    images_list = []
    images_clip_list = []
    support_clip_list = []
    support_mask_weights_list = []
    icl_region_clip_list = []
    conversation_list = []
    questions_list = []
    gts_list = []
    sampled_classes_list = []
    offset_list = [0]
    answer_type_list = []
    cnt = 0
    for data_dict in list_data_dict:
        image_path_list.append(data_dict.get('image_path', None))
        images_list.append(data_dict.get('image_sam', None))
        images_clip_list.append(data_dict.get('image_clip', None))
        # SGCAFE: support 数据作为独立字段
        if 'support_clip' in data_dict:
            support_clip_list.append(data_dict['support_clip'])
            support_mask_weights_list.append(data_dict['support_mask_weights'])
        if 'icl_region_clip' in data_dict:
            icl_region_clip_list.append(data_dict['icl_region_clip'])
        conversation_list.extend(data_dict.get('conversations', None))
        questions_list.append(data_dict.get('question', None))
        gts_list.append(data_dict.get('gt', None))
        sampled_classes_list.append(data_dict.get('sampled_classes', None))
        cnt += len(data_dict.get('conversations', None))
        offset_list.append(cnt)
        answer_type_list.append(data_dict.get('answer_type', None))

    # images_clip: 支持单图 tensor 和多图 list 两种格式
    # Token ICL 模式下 image_clip 是 list of tensors，普通模式下是单个 tensor
    has_token_icl = any(isinstance(d.get('image_clip', None), list) for d in list_data_dict)
    if has_token_icl:
        # Token ICL: 每个样本的 image_clip 是 [support_clip..., query_clip] 的 list
        # 将每个样本的多图 stack 成 (num_images, C, H, W)，再 batch stack 成 (B, num_images, C, H, W)
        # medplib_arch.py 中 images.ndim == 5 分支会 reshape 为 (B*num_images, C, H, W) 统一处理
        num_images_per_sample = max(
            len(d['image_clip']) if isinstance(d['image_clip'], list) else 1
            for d in list_data_dict
        )
        stacked_samples = []
        for d in list_data_dict:
            clip_data = d.get('image_clip', None)
            if isinstance(clip_data, list):
                # 如果图片数少于 max（不应该发生，但防御性处理），用零填充
                while len(clip_data) < num_images_per_sample:
                    clip_data.append(torch.zeros_like(clip_data[0]))
                stacked_samples.append(torch.stack(clip_data, dim=0))  # (num_images, C, H, W)
            else:
                # 非 ICL 样本：query 图放在最前面，其余用零填充
                # 因为它的 input_ids 里只有 1 个 <image> token，
                # prepare_inputs_labels_for_multimodal 中 cur_image_idx 只取第一个
                padding = [clip_data] + [torch.zeros_like(clip_data)] * (num_images_per_sample - 1)
                stacked_samples.append(torch.stack(padding, dim=0))  # (num_images, C, H, W)
        images_clip_final = torch.stack(stacked_samples, dim=0)  # (B, num_images, C, H, W)
    else:
        images_clip_final = torch.stack(images_clip_list, dim=0)

    final_batch = {
            "image_paths": image_path_list,
            "images": torch.stack(images_list, dim=0),
            "images_clip": images_clip_final,
            "input_ids": batch['input_ids'],
            "labels": batch['labels'],
            "attention_mask": batch['attention_mask'],
            "masks_list": batch['masks'],
            "label_list": batch['label_list'],
            "resize_list": resize_list,
            "offset": torch.LongTensor(offset_list),
            "questions_list": questions_list,
            "gts_list": gts_list,
            "sampled_classes_list": sampled_classes_list,
            "conversation_list": conversation_list,
            "seg_flag": seg_flag,
            "valid_mask_bool": batch.get('valid_mask_bool', []),
            "inference": inference,
            "answer_type_list": answer_type_list,
            "rp_flag": rp_flag,
            "region_masks": region_masks,
            "valid_region_masks_bool": valid_region_masks_bool,
        }

    # SGCAFE: 只有当 batch 中所有样本都有 support 时才传递
    if len(support_clip_list) == len(list_data_dict) and len(support_clip_list) > 0:
        final_batch["support_clip"] = torch.stack(support_clip_list, dim=0)
        final_batch["support_mask_weights"] = torch.stack(support_mask_weights_list, dim=0)

    # Region ICL: 传递 support 图的 CLIP tensor
    if len(icl_region_clip_list) == len(list_data_dict) and len(icl_region_clip_list) > 0:
        final_batch["icl_region_clip"] = torch.stack(icl_region_clip_list, dim=0)

    return final_batch
