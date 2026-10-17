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
from dvis_Plus.meta_architecture import DVIS_Plus_online

from research.modeling.sam2_utils.hieradet import Hiera
from research.modeling.sam2_utils.image_encoder import FpnNeck
from research.modeling.sam2_utils.position_encoding import PositionEmbeddingSine
from research.modeling.backbone.sam2_hiera import SAM2HieraBackbone

@META_ARCH_REGISTRY.register()
class DVIS_Plus_SAM2_online(DVIS_Plus_online):
    """
    SAM 2 的 Hiera Encoder 替换DVIS++ 的ResNet5 的backbone。
    """
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg,backbone.output_shape() )

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = VideoHungarianMatcher_Consistent(
            cost_class=class_weight, cost_mask=mask_weight, cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS, frames=cfg.INPUT.SAMPLING_FRAME_NUM
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if cfg.MODEL.TRACKER.USE_CL:
            weight_dict.update({'loss_reid': 2})

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes, matcher=matcher, weight_dict=weight_dict,
            eos_coef=no_object_weight, losses=["labels", "masks"],
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        hidden_channel = cfg.MODEL.MASK_FORMER.HIDDEN_DIM * 2 if cfg.MODEL.MASK_FORMER.REID_BRANCH else cfg.MODEL.MASK_FORMER.HIDDEN_DIM

        tracker = ReferringTracker_noiser(
            hidden_channel=hidden_channel,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            noise_mode=cfg.MODEL.TRACKER.NOISE_MODE,
            noise_ratio=cfg.MODEL.TRACKER.NOISE_RATIO,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "tracker": tracker,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": cfg.SOLVER.MAX_ITER,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            "use_cl": cfg.MODEL.REFINER.USE_CL,
        }

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

        # ---------------------------------------------------------------------
        # sam2的输出示例{"vision_features": [...], "vision_pos_enc": [...], "backbone_fpn": [...]}
        # 而 Detectron2 的 sem_seg_head 的输入示例：{"res2": tensor, "res3": tensor...}
        # ---------------------------------------------------------------------
        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=self.window_size)
        else:
            # self.backbone.eval()
            # self.sem_seg_head.eval()
            with torch.no_grad():
                features = self.backbone(images.tensor)
                image_outputs = self.sem_seg_head(features)
                object_labels = self._get_instance_labels(image_outputs['pred_logits'])
                frame_embds = image_outputs['pred_embds'].clone().detach()  
                frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone().detach() 
                mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
                del image_outputs['mask_features']
                torch.cuda.empty_cache()
                
            outputs, indices = self.tracker(
                frame_embds, mask_features, return_indices=True,
                resume=self.keep, frame_classes=object_labels,
                frame_embeds_no_norm=frame_embds_no_norm
            )
            image_outputs = self.reset_image_output_order(image_outputs, indices)
        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            # use the segmenter prediction results to guide the matching process during early training phase
            image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                outputs, targets, image_outputs=image_outputs
            )
            if self.iter < self.max_iter_num // 2:
                losses, reference_match_result = self.criterion(outputs, targets,
                                                                matcher_outputs=image_outputs,
                                                                ret_match_result=True)
            else:
                losses, reference_match_result = self.criterion(outputs, targets,
                                                                matcher_outputs=None,
                                                                ret_match_result=True)
            if self.use_cl:
                losses_cl = self.get_cl_loss_ref(outputs, reference_match_result)
                losses.update(losses_cl)

            self.iter += 1

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs = self.post_processing(outputs)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            pred_ids = outputs["ids"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            pred_id = pred_ids[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            del images, input_per_image
            gc.collect()
            torch.cuda.empty_cache()

            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id
            )
    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size
            # segmeter inference
            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            # referring tracker inference
            frame_embds = out['pred_embds']  # (b, c, t, q)
            frame_embds_no_norm = out['pred_embds_without_norm']
            mask_features = out['mask_features'].unsqueeze(0)
            if i != 0 or self.keep:
                track_out = self.tracker(frame_embds, mask_features,
                                         resume=True, frame_embeds_no_norm=frame_embds_no_norm)
            else:
                track_out = self.tracker(frame_embds, mask_features, frame_embeds_no_norm=frame_embds_no_norm)
            # remove unnecessary variables to save GPU memory
            del mask_features
            for j in range(len(track_out['aux_outputs'])):
                del track_out['aux_outputs'][j]['pred_masks'], track_out['aux_outputs'][j]['pred_logits']
            track_out['pred_logits'] = track_out['pred_logits'].to(torch.float32).detach().cpu()
            track_out['pred_masks'] = track_out['pred_masks'].to(torch.float32).detach().cpu()
            track_out['pred_embds'] = track_out['pred_embds'].to(torch.float32).detach().cpu()
            # track_out['pred_logits'] = track_out['pred_logits'].detach()
            # track_out['pred_masks'] = track_out['pred_masks'].detach()
            # track_out['pred_embds'] = track_out['pred_embds'].detach()
            out_list.append(track_out)

            del frame_embds, frame_embds_no_norm, out
            gc.collect()
            torch.cuda.empty_cache()

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1)
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2)
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2)
        

        return outputs