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
class CTMinVIS_sam2(CTMinVIS):

    def forward(self, batched_inputs):
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        pixel_mean = self.pixel_mean
        pixel_std = self.pixel_std
        images = [(x / 255.0 - pixel_mean) / pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        num_frames = len(images.tensor)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=3)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            # for cl loss
            det_outputs, gt_instances = self.prepare_for_cl_plugin(outputs, targets)
            losses.update(self.cl_plugin.train_loss(
                det_outputs, gt_instances, self.image_matcher))
            return losses
        else:
            outputs = self.post_processing(outputs)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(
                mask_cls_result,
                mask_pred_result,
                image_size,
                height,
                width,
                first_resize_size)
