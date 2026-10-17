# fbfl_meta_architecture.py
# Copyright (c) 2026. All rights reserved.
# Written for DVIS_FBFL Video Object Tracking & Segmentation.

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
from mask2former_video.modeling.matcher import VideoHungarianMatcher_Consistent
from mask2former_video.utils.memory import retry_if_cuda_oom
from scipy.optimize import linear_sum_assignment

# 基础框架在线父类
from dvis_Plus.meta_architecture import DVIS_Plus_online

# 🌟 引入我们高精度重构并保存的各个组件
try:
    from research.modeling.tracker.track_state import TrackStateTensor
except ImportError:
    from .tracker.track_state import TrackStateTensor

try:
    from research.modeling.sam2_utils.fbfl_prompt_encoder import FBFLPromptEncoder
except ImportError:
    try:
        from research.modeling.prompt_encoder import FBFLPromptEncoder
    except ImportError:
        from .sam2_utils.fbfl_prompt_encoder import FBFLPromptEncoder

try:
    from research.modeling.tracker.referring_tracker_fbfl import ReferringTracker_FBFL
except ImportError:
    from .tracker.referring_tracker_fbfl import ReferringTracker_FBFL


@META_ARCH_REGISTRY.register()
class DVIS_FBFL_Online(DVIS_Plus_online):
    """
    两阶段可训练的 Frame-by-Frame Loop 视频实例分割架构。
    
    两阶段设计：
    - Stage 1: 训练 Segmenter 和 PromptEncoder。在训练前向时以一定概率 (gt_prompt_prob) 
               向网络提供加噪的 GT 框作为空间 Prompt，其余时间不提供以训练网络的无状态分割能力。此时 Tracker 参数冻结。
    - Stage 2: 冻结 Segmenter 的所有参数，解冻并只训练 Tracker（和可选的 PromptEncoder）。此时全部使用 Tracker 自反馈生成的 Prompt。
    """
    @configurable
    def __init__(
        self, 
        training_stage: int, 
        gt_prompt_prob: float, 
        prompt_noise_level: float, 
        input_image_size: Tuple[int, int],
        image_embedding_size: Tuple[int, int],
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.training_stage = training_stage         # 1 或 2
        self.gt_prompt_prob = gt_prompt_prob         # Stage 1 训练时，输入 GT 框作为 Prompt 的概率
        self.prompt_noise_level = prompt_noise_level # Jittering 噪声系数 (避免过拟合)
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        
        # 注册用于显式时空几何反馈到语义维度的 Prompt 投影层
        hidden_dim = self.sem_seg_head.predictor.query_embed.embedding_dim if hasattr(
            self.sem_seg_head, "predictor"
        ) else 256
        
        self.prompt_encoder = FBFLPromptEncoder(
            embed_dim=hidden_dim,
            image_embedding_size=self.image_embedding_size, # 动态分配特征图尺寸
            input_image_size=self.input_image_size,         # 动态分配输入图像大小
            mask_in_chans=16,                               # 与 SAM 2 保持一致
        ).to(self.device)
        
        self.motion_weight = 0.4 
        self.next_track_id = 0

        # 根据训练阶段实施参数梯度管理 (两阶段解耦控制)
        self._manage_gradients()

    @classmethod
    def from_config(cls, cfg):
        # 1. 兼容性读取：读取两阶段训练和运动几何的相关配置
        training_stage = cfg.MODEL.FBFL.TRAINING_STAGE if hasattr(cfg.MODEL, "FBFL") and hasattr(cfg.MODEL.FBFL, "TRAINING_STAGE") else 1
        gt_prompt_prob = cfg.MODEL.FBFL.GT_PROMPT_PROB if hasattr(cfg.MODEL, "FBFL") and hasattr(cfg.MODEL.FBFL, "GT_PROMPT_PROB") else 0.5
        prompt_noise_level = cfg.MODEL.FBFL.PROMPT_NOISE_LEVEL if hasattr(cfg.MODEL, "FBFL") and hasattr(cfg.MODEL.FBFL, "PROMPT_NOISE_LEVEL") else 0.05
        
        # 2. 计算图像及下采样特征图的分辨率，用于安全初始化 PromptEncoder
        input_image_size = cfg.INPUT.MIN_SIZE_TRAIN[0] if hasattr(cfg.INPUT, "MIN_SIZE_TRAIN") else 1024
        if isinstance(input_image_size, int):
            input_image_size = (input_image_size, input_image_size)
        # 依照 SAM 2 的 1/16 降采样比率计算 Embedding Size
        image_embedding_size = (input_image_size[0] // 16, input_image_size[1] // 16)

        # 3. 重定向重载 Tracker：
        # 这里使用我们全新重载、支持双流运动几何度量的 ReferringTracker_FBFL
        hidden_channel = cfg.MODEL.MASK_FORMER.HIDDEN_DIM * 2 if cfg.MODEL.MASK_FORMER.REID_BRANCH else cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        motion_weight = cfg.MODEL.TRACKER.MOTION_WEIGHT if hasattr(cfg.MODEL.TRACKER, "MOTION_WEIGHT") else 0.4
        
        tracker = ReferringTracker_FBFL(
            hidden_channel=hidden_channel,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            noise_mode=cfg.MODEL.TRACKER.NOISE_MODE,
            noise_ratio=cfg.MODEL.TRACKER.NOISE_RATIO,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            motion_weight=motion_weight
        )

        base_dict = super().from_config(cfg)
        
        # 注入重构后的控制参数、计算尺寸和新版 Tracker
        base_dict.update({
            "training_stage": training_stage,
            "gt_prompt_prob": gt_prompt_prob,
            "prompt_noise_level": prompt_noise_level,
            "input_image_size": input_image_size,
            "image_embedding_size": image_embedding_size,
            "tracker": tracker # 覆盖父类的 tracker 构造
        })
        return base_dict

    def _manage_gradients(self):
        """
        精细化控制两阶段下的参数冻结与梯度回传。
        """
        if self.training_stage == 1:
            print(">>> [DVIS_FBFL] Stage 1 Training: Optimizing Segmenter & PromptEncoder, Freezing Tracker...")
            # 解冻 Segmenter (Backbone 和 sem_seg_head)
            for p in self.backbone.parameters():
                p.requires_grad_(True)
            for p in self.sem_seg_head.parameters():
                p.requires_grad_(True)
            # 解冻 Prompt 编码器
            for p in self.prompt_encoder.parameters():
                p.requires_grad_(True)
            # 冻结 Tracker
            for p in self.tracker.parameters():
                p.requires_grad_(False)
                
        elif self.training_stage == 2:
            print(">>> [DVIS_FBFL] Stage 2 Training: Freezing Segmenter, Optimizing Tracker & PromptEncoder...")
            # 冻结 Segmenter (Backbone 和 sem_seg_head)
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.sem_seg_head.parameters():
                p.requires_grad_(False)
            # 解冻 Tracker 和 Prompt 编码器
            for p in self.prompt_encoder.parameters():
                p.requires_grad_(True)
            for p in self.tracker.parameters():
                p.requires_grad_(True)

    def forward(self, batched_inputs):
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        pixel_mean = self.pixel_mean
        pixel_std = self.pixel_std
        images = [(x / 255.0 - pixel_mean) / pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        num_frames = len(images.tensor)

        # 准备 Targets 字典，在训练阶段将其直接输入到自回归回路中
        targets_dict = None
        if self.training:
            targets_list = self.prepare_ta