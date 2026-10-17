import logging
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import gc
import numpy as np
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, BitMasks
from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
from mask2former_video.utils.memory import retry_if_cuda_oom
from dvis_Plus.tracker import ReferringTracker_noiser
from dvis_Plus.utils import loss_reid
from dvis_Plus.meta_architecture import DVIS_Plus_online

logger = logging.getLogger(__name__)


# =====================================================================
# 🟢 终极优雅重构版：DVIS_Plus_online_AttnMaskRefine
# 继承自 DVIS_Plus_online，重写 run_window_inference 实现两阶段全流程窗口推理
# =====================================================================
@META_ARCH_REGISTRY.register()
class DVIS_Plus_online_AttnMaskRefine(DVIS_Plus_online):
    """
    两阶段在线细化模型（基于注意力掩码）：
    Phase 1: 原 DVIS-Plus 流程，获得轨迹级 mask 和特征。
    Phase 2: 利用轨迹 mask 构造注意力掩码，再次通过分割器得到 refined 结果。
    """

    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg):
        base = super().from_config(cfg)
        return base

    def forward(self, batched_inputs):
        if self.training:
            return self.forward_train_refine(batched_inputs)
        else:
            return self.forward_inference_refine(batched_inputs)

    # ----------------------------------------------------------------- #
    # 🟢 完美重构版：两阶段时序闭环窗口推理函数
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def run_window_inference(self, images_tensor, window_size=30):
        """
        覆盖父类的单阶段窗口推理。
        在每个分段窗口（以 window_size 为大小）内部，完整执行两阶段时序细化。
        由于每次只处理 window_size 帧（如 20 或 30 帧），
        这能将 Pixel Decoder 和自注意力层的显存峰值降低数倍，彻底杜绝 OOM。
        """
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []

        # 垃圾回收，确保初始显存纯净
        gc.collect()
        torch.cuda.empty_cache()

        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size
            
            # 当前窗口的时序衔接状态：除第一窗口外，后续窗口一律 resume=True
            is_resume = (i != 0 or self.keep)

            # ==========================================
            # 🟢 第一阶段 (Phase 1): 提取初始特征并进行第一次追踪
            # ==========================================
            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            
            # 及时释放主干层特征图，腾出显存空间
            del features['res2'], features['res3'], features['res4'], features['res5']
            
            # 提取变量进行第一次 Tracker 推理
            frame_embds = out['pred_embds']  # (b, c, t, q)
            frame_embds_no_norm = out['pred_embds_without_norm']
            mask_features = out['mask_features'].unsqueeze(0)
            
            track_out1 = self.tracker(
                frame_embds, mask_features, 
                resume=is_resume, 
                frame_embeds_no_norm=frame_embds_no_norm
            )
            
            # ==========================================
            # 🟢 第二阶段 (Phase 2): 基于预测的 mask 构建正向引导注意力偏置并进行二次细化
            # ==========================================
            # 1. 提取当前窗口下的 tracker1 预测的掩码 logits
            track_masks = track_out1['pred_masks'] # (B, Q, T, H, W)
            
            # 2. 构造注意力偏置
            attention_mask = self._prepare_track_masks_for_attention(track_masks)
            attention_mask = attention_mask.float()
            
            # 3. 重新以前端极轻量级的开销，提取当前窗口的骨干特征用于二次细化
            features = self.backbone(images_tensor[start_idx:end_idx])
            
            # 4. 🟢 运行第二次细化分割头（特征经过 AnyUp/LoftUP 提纯与 AttentionMask 引导）
            refined_out1 = self.sem_seg_head(features, track_masks=attention_mask)
            
            # 再次释放主干特征图
            del features['res2'], features['res3'], features['res4'], features['res5']
            
            # 5. 提取变量，送入第二次在线追踪，更新最终轨迹并进行时序去噪
            refined_embds = refined_out1['pred_embds']
            refined_embds_no_norm = refined_out1['pred_embds_without_norm']
            
            track_out2 = self.tracker(
                refined_embds, mask_features, 
                resume=is_resume, 
                frame_embeds_no_norm=refined_embds_no_norm
            )
            
            # ==========================================
            # 🟢 垃圾回收与变量清理，准备拼接
            # ==========================================
            del mask_features
            for j in range(len(track_out2['aux_outputs'])):
                del track_out2['aux_outputs'][j]['pred_masks'], track_out2['aux_outputs'][j]['pred_logits']
            
            # 将输出转换为 float32 并搬运回 CPU，防止长视频导致 VRAM 累计堆叠
            track_out2['pred_logits'] = track_out2['pred_logits'].to(torch.float32).detach().cpu()
            track_out2['pred_masks'] = track_out2['pred_masks'].to(torch.float32).detach().cpu()
            track_out2['pred_embds'] = track_out2['pred_embds'].to(torch.float32).detach().cpu()
            
            out_list.append(track_out2)

            # 释放本窗口所有不再使用的临时大张量
            del frame_embds, frame_embds_no_norm, out, track_out1, refined_out1, refined_embds, refined_embds_no_norm
            gc.collect()
            torch.cuda.empty_cache()

        # 合并所有分段窗口的输出结果，无损拼接回长视频空间
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1)
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2)
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2)
        
        return outputs

    # ----------------------------------------------------------------- #
    # 🟢 极大简化后的推理函数（直接委托给上方的窗口推理）
    # ----------------------------------------------------------------- #
    @torch.no_grad()
    def forward_inference_refine(self, batched_inputs):
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        self.backbone.eval()
        self.sem_seg_head.eval()
        self.tracker.eval()

        # 🟢 直接调用我们重组后的 run_window_inference
        # 它会自动安全地读取配置里的 self.window_size (如 20)，以最优段落跑通长视频
        outputs = self.run_window_inference(images.tensor, window_size=self.window_size)

        # 最终后处理
        refined_outputs = self.post_processing(outputs)

        mask_cls_result = refined_outputs["pred_logits"][0]
        mask_pred_result = refined_outputs["pred_masks"][0]
        pred_id = refined_outputs["ids"][0]

        first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])
        input_per_image = batched_inputs[0]
        image_size = images.image_sizes[0]
        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])

        del images, input_per_image
        gc.collect()
        torch.cuda.empty_cache()

        return self.inference_video_task(
            mask_cls_result, mask_pred_result, image_size, height, width,
            first_resize_size, pred_id
        )

    # ----------------------------------------------------------------- #
    # 训练函数保持不变
    # ----------------------------------------------------------------- #
    def forward_train_refine(self, batched_inputs):
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        # 阶段0：冻结 backbone 和 sem_seg_head，提取特征
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.tracker.train()

        with torch.no_grad():
            features = self.backbone(images.tensor)
            image_outputs = self.sem_seg_head(features)
            object_labels = self._get_instance_labels(image_outputs['pred_logits'])
            frame_embds = image_outputs['pred_embds'].clone().detach()
            frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone().detach()
            mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)

        targets = self.prepare_targets(batched_inputs, images)

        # 阶段1：原版 tracker，计算损失 loss_t1
        tracker_out1, indices1 = self.tracker(
            frame_embds, mask_features, return_indices=True,
            resume=self.keep, frame_classes=object_labels,
            frame_embeds_no_norm=frame_embds_no_norm
        )
        image_outputs = self.reset_image_output_order(image_outputs, indices1)

        # 保存 mask（detach）用于后续 refine
        track_masks = tracker_out1['pred_masks'].detach()

        t1_loss_input = {
            'pred_logits': tracker_out1['pred_logits'],
            'pred_masks': tracker_out1['pred_masks'],
            'pred_references': tracker_out1['pred_references'],
        }
        if 'aux_outputs' in tracker_out1:
            t1_loss_input['aux_outputs'] = tracker_out1['aux_outputs']

        image_outputs, t1_loss_input, t1_targets = self.frame_decoder_loss_reshape(
            t1_loss_input, targets, image_outputs=image_outputs
        )

        if self.iter < self.max_iter_num // 2:
            losses_t1, _ = self.criterion(
                t1_loss_input, t1_targets,
                matcher_outputs=image_outputs,
                ret_match_result=True
            )
        else:
            losses_t1, _ = self.criterion(
                t1_loss_input, t1_targets,
                matcher_outputs=None,
                ret_match_result=True
            )

        # 阶段2：第一次 refine，计算损失 loss_r1
        # self.sem_seg_head.train()
        refined_out1 = self.sem_seg_head(features, track_masks=track_masks)

        refined_embds = refined_out1['pred_embds']
        refined_embds_no_norm = refined_out1['pred_embds_without_norm']
        refined_out1['pred_references'] = tracker_out1['pred_references'].detach()

        r1_loss_input = {
            'pred_logits': refined_out1['pred_logits'],
            'pred_masks': refined_out1['pred_masks'],
            'pred_references': refined_out1['pred_references'],
        }
        if 'aux_outputs' in refined_out1:
            r1_loss_input['aux_outputs'] = refined_out1['aux_outputs']

        _, r1_loss_input, gt_targets_r1 = self.frame_decoder_loss_reshape(
            r1_loss_input, targets, image_outputs=None
        )
        if self.iter < self.max_iter_num // 2:
            loss_r1, ref_match1 = self.criterion(
                r1_loss_input, gt_targets_r1, matcher_outputs=image_outputs, ret_match_result=True
            )
        else:
            loss_r1, ref_match1 = self.criterion(
                r1_loss_input, targets, matcher_outputs=None, ret_match_result=True
            )

        # 阶段3：第二次 tracker，计算损失 loss_t2
        tracker_out2, indices2 = self.tracker(
            refined_embds, mask_features, return_indices=True,
            resume=self.keep, frame_classes=object_labels,
            frame_embeds_no_norm=refined_embds_no_norm
        )

        t2_loss_input = {
            'pred_logits': tracker_out2['pred_logits'],
            'pred_masks': tracker_out2['pred_masks'],
            'pred_references': tracker_out2['pred_references'],
        }
        if 'aux_outputs' in tracker_out2:
            t2_loss_input['aux_outputs'] = tracker_out2['aux_outputs']
        _, t2_loss_input, t2_targets = self.frame_decoder_loss_reshape(
            t2_loss_input, targets, image_outputs=None
        )
        loss_t2, _ = self.criterion(
            t2_loss_input, t2_targets, matcher_outputs=None, ret_match_result=True
        )

        # 融合多阶段损失
        weight_t1 = 0.5
        weight_r1 = 0.5
        weight_t2 = 0.5

        losses = {}
        for k, v in losses_t1.items():
            losses[k] = v * weight_t1
        for k, v in loss_r1.items():
            if k in losses:
                losses[k] += v * weight_r1
            else:
                losses[k] = v * weight_r1
        for k, v in loss_t2.items():
            if k in losses:
                losses[k] += v * weight_t2
            else:
                losses[k] = v * weight_t2

        self.iter += 1
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                losses.pop(k)
        return losses

