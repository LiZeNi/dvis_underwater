from typing import Tuple
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
from dvis_Plus.ctvis import CTMinVIS

@META_ARCH_REGISTRY.register()
class freeze_CTMinVIS(CTMinVIS):
    """
    CTMinVIS 的子类，在初始化时自动禁用 Backbone 的所有梯度更新，
    并在整个 DVIS 训练期间锁定其为 eval 评估模式。
    """
    @configurable
    def __init__(self, **kwargs):
        # 1. 调用父类 CTMinVIS 的初始化，完整构建 Backbone 和各项组件
        super().__init__(**kwargs)

        # 2. 🟢 严格冻结 Backbone 的所有参数（禁用反向传播的梯度计算）
        if hasattr(self, "backbone") and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
            # 3. 🟢 设置骨干网络为 eval 评估模式
            self.backbone.eval()
            print("🔒 [freeze_CTMinVIS] Backbone parameters and gradients are strictly frozen.")

    # 4. 🟢 重写 train 方法，防止 DVIS 训练时骨干网络意外切换回 train 模式
    # 这可以完美锁定 BatchNorm/SyncBatchNorm 的运行统计，并完全关闭 Dropout 干扰
    def train(self, mode=True):
        # 直接调用 PyTorch nn.Module 基类的 train，安全避开 MRO 继承冲突
        nn.Module.train(self, mode)
        if hasattr(self, "backbone") and self.backbone is not None:
            self.backbone.eval()