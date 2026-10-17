# feedback_meta_architecture.py
# Copyright (c) 2026. All rights reserved.
# Written for DVIS_Plus Online Feedback Autoregressive Training & Inference.

from typing import Tuple, List, Dict, Any, Optional
import einops
import pycocotools.mask as mask_util
import torch
from torch import nn
from torch.nn import functional as F
import gc
import os
import psutil
import numpy as np
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
from mask2former_video.utils.memory import retry_if_cuda_oom
from scipy.optimize import linear_sum_assignment

from dvis_Plus.tracker import ReferringTracker_noiser
from dvis_Plus.refiner import TemporalRefiner
from dvis_Plus.utils import loss_reid, Outputs_Memory_PerClasses
from dvis_Plus.meta_architecture import DVIS_Plus_online, MinVIS

class PromptGenerator(nn.Module):
    """可训练的提示生成器，将轨迹特征映射为分割查询提示。"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),   # 输入 → 输出
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, q_track):
        # 残差连接：先将输入投影到输出维度，再加 MLP 输出
        projected = self.mlp[0](q_track)
        return self.norm(projected + self.mlp(q_track))


@META_ARCH_REGISTRY.register()
class DVIS_Plus_online_Feedback(DVIS_Plus_online):
    """
    DVIS-Plus 在线反馈版本（训练/推理均使用逐帧 Prompt 注入）。
    """

    @configurable
    def __init__(
        self,
        *,
        prompt_dim: int,          # PromptGenerator 的维度
        hidden_dim: int,
        prompt_num: int,
        **kwargs,                 # 其余参数传递给父类
    ):
        super().__init__(**kwargs)
    
        # 可学习初始 prompt（用于第一帧）
        self.initial_prompt = nn.Parameter(torch.zeros(1, prompt_dim))
        self.prompt_num = min(prompt_num, self.num_queries)  # 确保不超过总 query 数

        # 可训练提示生成器
        self.prompt_generator = PromptGenerator(in_dim=prompt_dim, out_dim=hidden_dim)
        self.prompt_generator.to(self.device)

    @classmethod
    def from_config(cls, cfg):
        base = super().from_config(cfg)
        if cfg.MODEL.MASK_FORMER.REID_BRANCH:
            prompt_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM * 2
        else:
            prompt_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        base["prompt_dim"] = prompt_dim
        base["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        base["prompt_num"] = cfg.MODEL.MASK_FORMER.TEST.MAX_NUM
        return base

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.prompt_generator.train(mode)
        if hasattr(self, 'tracker'):
            self.tracker.train(mode)

    # ----------------------------------------------------------------- #
    # 主前向入口
    # ----------------------------------------------------------------- #
    def forward(self, batched_inputs):
        if self.training:
            return self.forward_train_feedback(batched_inputs)
        else:
            return self.forward_inference_feedback(batched_inputs)

    # ----------------------------------------------------------------- #
    # 训练：逐帧自回归反馈
    # ----------------------------------------------------------------- #
    def forward_train_feedback(self, batched_inputs):
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        # 预处理所有帧图像
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        num_total_frames = len(images.tensor)

        # 设置模块状态
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.prompt_generator.train()
        self.tracker.train()

        outputs_all_frames = []
        prev_tracker_outputs = None
        prompt_list = []          # 存储每帧生成的 prompt，用于辅助损失

        for t_idx in range(num_total_frames):
            single_frame_img = images.tensor[t_idx : t_idx + 1]

            # 冻结 backbone 提取特征
            with torch.no_grad():
                features = self.backbone(single_frame_img)

            # 3. 构造当前帧的 prompt 输入（q_track_detached）
            if t_idx == 0:
                # 第一帧：使用可学习的初始 prompt，形状 [prompt_num, 1, prompt_dim]
                q_track_detached = self.initial_prompt.unsqueeze(0).repeat(self.prompt_num, 1, 1)
            else:
                # 后续帧：从上一帧 tracker 输出中提取轨迹特征
                q_track_raw = prev_tracker_outputs['pred_references'].squeeze(2)  # (b, c, q)
                q_track_raw = q_track_raw.permute(2, 0, 1)                        # (q, b, c)
                q_track_raw = q_track_raw[:self.prompt_num]                       # 截取前 prompt_num 个
                q_track_detached = q_track_raw.detach()                           # 阻断梯度流向 tracker

            # 4. 通过可训练的 PromptGenerator 生成最终 prompt
            q_prompt = self.prompt_generator(q_track_detached)   # [K, 1, hidden_dim]
            prompt_list.append(q_prompt)                         # 收集用于辅助损失

            # 5. 分割解码器注入 prompt
            image_outputs = self.sem_seg_head(features, query_prompts=q_prompt)

            # 6. 提取跟踪所需变量（detach 避免梯度流入 backbone/segmenter）
            object_labels = self._get_instance_labels(image_outputs['pred_logits'])
            frame_embds = image_outputs['pred_embds'].clone().detach()
            frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone().detach()
            mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)

            # 7. 跟踪器更新轨迹
            tracker_outputs, indices = self.tracker(
                frame_embds, mask_features, return_indices=True,
                resume=(t_idx > 0 or self.keep),
                frame_classes=object_labels,
                frame_embeds_no_norm=frame_embds_no_norm
            )

            # 8. 对齐分割输出顺序
            image_outputs = self.reset_image_output_order(image_outputs, indices)
            outputs_all_frames.append((tracker_outputs, image_outputs))

            # 9. 保存当前帧 tracker 输出，供下一帧使用
            prev_tracker_outputs = tracker_outputs

        # 合并所有帧输出
        merged_outputs = self.merge_autoregressive_outputs(outputs_all_frames)

        # 准备 targets
        targets = self.prepare_targets(batched_inputs, images)

        # 重塑并计算损失
        image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
            merged_outputs['tracker_out'], targets,
            image_outputs=merged_outputs['segmenter_out']
        )

        if self.iter < self.max_iter_num // 2:
            losses, reference_match_result = self.criterion(
                outputs, targets, matcher_outputs=image_outputs, ret_match_result=True
            )
        else:
            losses, reference_match_result = self.criterion(
                outputs, targets, matcher_outputs=None, ret_match_result=True
            )

        if self.use_cl:
            losses_cl = self.get_cl_loss_ref(outputs, reference_match_result)
            losses.update(losses_cl)

        # ========================== 辅助损失 ==========================
        # 强制 prompt_generator 和 initial_prompt 在所有 rank 上都获得梯度，
        # 避免 DDP 的 unused parameter 错误。
        if prompt_list:
            aux_loss = sum([p.norm() for p in prompt_list]) * 1e-6
            losses['loss_prompt_aux'] = aux_loss

        self.iter += 1

        for k in list(losses.keys()):
            if k == 'loss_prompt_aux':
                continue  # 不乘权重，直接保留
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return losses

# ----------------------------------------------------------------- #
    # 推理：逐帧自回归反馈（无梯度）
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def forward_inference_feedback(self, batched_inputs):
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        # 1. 预处理当前视频的所有帧图像
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        num_total_frames = len(images.tensor)

        # 2. 所有相关子模块设为评估模式
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.prompt_generator.eval()
        self.tracker.eval()

        # 🌟 初始化：在此处被无损定义
        outputs_all_frames = []
        q_prompt = None

        # 3. 逐帧自回归推理
        for t_idx in range(num_total_frames):
            single_frame_img = images.tensor[t_idx : t_idx + 1]

            # 骨干网络提取图像特征
            features = self.backbone(single_frame_img)
            # 注入 Prompt 引导分割
            image_outputs = self.sem_seg_head(features, query_prompts=q_prompt)

            object_labels = self._get_instance_labels(image_outputs['pred_logits'])
            frame_embds = image_outputs['pred_embds'].clone()
            frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone()
            mask_features = image_outputs['mask_features'].clone().unsqueeze(0)

            # 跟踪器匹配与时序更新
            tracker_outputs, indices = self.tracker(
                frame_embds, mask_features, return_indices=True,
                resume=(t_idx > 0 or self.keep),
                frame_classes=object_labels,
                frame_embeds_no_norm=frame_embds_no_norm
            )

            # 对齐分割结果并存入列表
            image_outputs = self.reset_image_output_order(image_outputs, indices)
            outputs_all_frames.append((tracker_outputs, image_outputs))

            # 🌟 修复处 1：将原先错误的 'pred_reid_embed' 修正为 'pred_references'，
            # 从而与训练过程保持 100% 对齐，并打通多卡推理通道。
            if 'pred_references' in tracker_outputs:
                q_track_raw = tracker_outputs['pred_references'].squeeze(2)  # (b, c, q)
            else:
                q_track_raw = tracker_outputs['pred_embds'].squeeze(2)      # 兜底回退
                
            q_track_raw = q_track_raw.permute(2, 0, 1)                      # (q, b, c)
            q_prompt = self.prompt_generator(q_track_raw)                   # 推理时无需 detach

        # 4. 合并输出
        merged_outputs = self.merge_autoregressive_outputs(outputs_all_frames)
        outputs = merged_outputs['tracker_out']  # 只使用 tracker 输出用于最终结果

        # 5. 后处理（平均 logits，添加 ids）
        outputs = self.post_processing(outputs)

        mask_cls_result = outputs["pred_logits"][0]
        mask_pred_result = outputs["pred_masks"][0]
        
        # 🌟 修复处 2：将默认建在 CPU 上的 pred_id 动态转移到与 mask_cls_result 相同的 GPU 上！
        # 这能彻底消除 `indices should be on the same device` 跨进程跨设备索引异常。
        pred_id = outputs["ids"][0].to(mask_cls_result.device)

        first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])
        input_per_image = batched_inputs[0]
        image_size = images.image_sizes[0]
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])

        # 6. 释放显存与内存
        del images, input_per_image
        gc.collect()
        torch.cuda.empty_cache()

        # 7. 调用对应任务的最终输出函数进行评估
        return self.inference_video_task(
            mask_cls_result,
            mask_pred_result,
            image_size,
            height,
            width,
            first_resize_size,
            pred_id
        )
    # ----------------------------------------------------------------- #
    # 合并逐帧输出为时序张量
    # ----------------------------------------------------------------- #
    def merge_autoregressive_outputs(self, outputs_all_frames):
        tracker_outs = [x[0] for x in outputs_all_frames]
        seg_outs = [x[1] for x in outputs_all_frames]

        merged = {
            'tracker_out': {
                'pred_logits': torch.cat([x['pred_logits'] for x in tracker_outs], dim=1),
                'pred_masks': torch.cat([x['pred_masks'] for x in tracker_outs], dim=2),
                'pred_references': torch.cat([x['pred_references'] for x in tracker_outs], dim=2),
                'aux_outputs': [
                    {
                        'pred_logits': torch.cat([x['aux_outputs'][i]['pred_logits'] for x in tracker_outs], dim=1),
                        'pred_masks': torch.cat([x['aux_outputs'][i]['pred_masks'] for x in tracker_outs], dim=2)
                    }
                    for i in range(len(tracker_outs[0]['aux_outputs']))
                ]
            },
            'segmenter_out': {
                'pred_logits': torch.cat([x['pred_logits'] for x in seg_outs], dim=1),
                'pred_masks': torch.cat([x['pred_masks'] for x in seg_outs], dim=2)
            }
        }
        return merged