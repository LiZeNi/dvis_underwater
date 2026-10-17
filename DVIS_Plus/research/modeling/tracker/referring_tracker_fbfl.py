# referring_tracker_fbfl.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for Motion-Geometric FBFL Tracker.

import random
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

# 导入原生 Tracker 的依赖组件
from dvis_Plus.tracker import ReferringTracker_noiser, ReferringCrossAttentionLayer
from dvis_Plus.noiser import Noiser


class KalmanTrackState:
    """
    可微的时序轨迹运动状态机。
    存储每一个活跃轨线的历史 Bbox 坐标、外观 Query 特征，并预测下一帧的几何区间。
    """
    def __init__(self, track_id: int, bbox: torch.Tensor, query_embd: torch.Tensor):
        self.track_id = track_id
        self.history_bboxes = [bbox]  # 存储最近 N 帧的归一化 Bbox [4] -> (x1, y1, x2, y2)
        self.query_embd = query_embd  # 关联上的最新 Query 语义特征 [C]
        self.age = 1

    def predict_next_bbox(self) -> torch.Tensor:
        """
        利用常速运动公式预测下一帧目标的空间状态。
        由于全部基于 Tensor 运算，因此对历史帧的参数保持完全可导。
        """
        if len(self.history_bboxes) < 2:
            return self.history_bboxes[-1]
        
        # 简单一阶差分速度模型: v = box_t - box_{t-1}
        velocity = self.history_bboxes[-1] - self.history_bboxes[-2]
        predicted_box = self.history_bboxes[-1] + velocity
        return torch.clamp(predicted_box, min=0.0, max=1.0)

    def update(self, bbox: torch.Tensor, query_embd: torch.Tensor):
        self.history_bboxes.append(bbox)
        self.query_embd = query_embd
        if len(self.history_bboxes) > 5:
            self.history_bboxes.pop(0)
        self.age += 1


class ReferringTracker_FBFL(ReferringTracker_noiser):
    """
    结合了[显式运动预测] 与 [外观特征相似度] 的双流自回归 Tracker。
    
    1. 完美继承自 ReferringTracker_noiser，共享所有 Transformer 层和特征映射层权重。
    2. 支持在循环内部进行 On-the-fly 的可微 Mask 和 Bbox 空间预测。
    3. 支持在联合关联后通过 Noiser 进行训练态噪声扰动。
    """
    def __init__(self, *args, motion_weight: float = 0.4, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion_weight = motion_weight  # 物理几何匹配的权重配比
        self.active_tracks: List[KalmanTrackState] = []
        self.next_track_id = 0

    def _mask_to_normalized_bbox(self, mask: torch.Tensor) -> torch.Tensor:
        """
        [可微算子]：将高分辨率 Mask 投影并提取为归一化的坐标框特征。
        """
        H, W = mask.shape
        device = mask.device
        if mask.sum() < 1e-4:
            return torch.tensor([0.0, 0.0, 1.0, 1.0], device=device, dtype=torch.float32)
            
        y_indices = torch.arange(H, dtype=torch.float32, device=device)
        x_indices = torch.arange(W, dtype=torch.float32, device=device)
        
        y_proj = mask.max(dim=1)[0].float()
        x_proj = mask.max(dim=0)[0].float()
        
        y_min = (y_proj * y_indices).min() / H
        y_max = (y_proj * y_indices).max() / H
        x_min = (x_proj * x_indices).min() / W
        x_max = (x_proj * x_indices).max() / W
        
        return torch.stack([x_min, y_min, x_max, y_max], dim=0)

    def forward(
        self, 
        frame_embeds, 
        mask_features, 
        resume=False,
        return_indices=False, 
        frame_classes=None,
        frame_embeds_no_norm=None
    ):
        """
        重构的追踪前向传播。在循环迭代时建立并融合显式运动轨迹。
        """
        # 原版 mask_feature 映射
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(*mask_features_shape) # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        if frame_embeds_no_norm is not None:
            frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        else:
            frame_embeds_no_norm = frame_embeds
            
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []
        all_frames_references = []

        # 帧级自回归循环开始
        for i in range(n_frame):
            ms_output = []
            single_frame_embeds = frame_embeds[i]  # [q, b, c]
            single_frame_embeds_no_norm = frame_embeds_no_norm[i]
            single_frame_classes = None if frame_classes is None else frame_classes[i]

            frame_key = single_frame_embeds_no_norm

            # -----------------------------------------------------------------
            # 🌟 核心：On-the-fly 预测当前帧的所有候选实例 Mask (用于几何匹配)
            # -----------------------------------------------------------------
            current_norm = self.decoder_norm(single_frame_embeds_no_norm.transpose(0, 1)) # [b, q, c]
            mask_embed_cur = self.mask_embed(current_norm) # [b, q, mask_dim]
            # 计算当前时刻的概率 Mask -> [b, q, h, w]
            masks_cur = torch.einsum("bqc,bchw->bqhw", mask_embed_cur, mask_features[:, i])
            masks_cur_sigmoid = masks_cur.sigmoid()

            # 首帧初始化
            if i == 0 and resume is False:
                self._clear_memory()
                self.active_tracks = []
                self.next_track_id = 0

                # 首帧没有历史时序，使用原版 Noiser 的外观机制匹配（通常为 identity 顺序）
                indices, noised_init = self.noiser(
                    single_frame_embeds,
                    single_frame_embeds,
                    cur_embeds_no_norm=single_frame_embeds_no_norm,
                    activate=False,
                    cur_classes=single_frame_classes,
                )
                
                # 初始化轨迹物理模型
                for idx in indices:
                    # 获取首帧该 Query 预测出的 Mask 几何并绑定
                    mask_idx = masks_cur_sigmoid[0, idx] # 针对 bs=0
                    bbox_idx = self._mask_to_normalized_bbox(mask_idx)
                    track = KalmanTrackState(
                        track_id=self.next_track_id,
                        bbox=bbox_idx,
                        query_embd=single_frame_embeds[idx, 0] # bs=0
                    )
                    self.active_tracks.append(track)
                    self.next_track_id += 1

            # -----------------------------------------------------------------
            # 🌟 后续帧：执行显式几何 + 语义外观关联匹配
            # -----------------------------------------------------------------
            else:
                N_active = len(self.active_tracks)
                
                # 1. 语义外观特征距离 (Cosine Distance)
                active_embds = torch.stack([t.query_embd for t in self.active_tracks], dim=0) # [N_active, c]
                current_embds = single_frame_embeds[:, 0, :] # [q, c], 对齐第一个 batch
                
                norm_active = active_embds / (active_embds.norm(dim=-1, keepdim=True) + 1e-6)
                norm_current = current_embds / (current_embds.norm(dim=-1, keepdim=True) + 1e-6)
                C_appearance = 1.0 - torch.mm(norm_current, norm_active.T) # [q, N_active]
                
                # 2. 物理运动几何预测距离 (L1 Box 偏差度量)
                C_motion = torch.zeros_like(C_appearance)
                predicted_bboxes = [t.predict_next_bbox() for t in self.active_tracks]
                
                for q_idx in range(n_q):
                    cand_mask = masks_cur_sigmoid[0, q_idx]
                    cand_bbox = self._mask_to_normalized_bbox(cand_mask)
                    for t_idx, pred_bbox in enumerate(predicted_bboxes):
                        # 计算预测框与候选框的空间 L1 距离
                        C_motion[q_idx, t_idx] = torch.dist(pred_bbox, cand_bbox, p=1)
                        
                # 3. 联合代价矩阵配比
                C_total = (1.0 - self.motion_weight) * C_appearance + self.motion_weight * C_motion
                C_total_np = C_total.detach().cpu().numpy()
                
                # 4. 匈牙利匹配
                matched_indices_np = linear_sum_assignment(C_total_np)
                matched_indices = list(matched_indices_np[0]) # 关联当前帧对应的索引顺序
                
                # 5. 支持 Noiser 时序特征噪声注入（在训练时提供 Scheduled Denoising）
                if self.training and random.random() < self.noiser.noise_ratio:
                    if self.noiser.mode == 'rs':
                        indices, noised_init = self.noiser._rs_noise_forward(single_frame_embeds_no_norm)
                    elif self.noiser.mode == 'wa':
                        indices, noised_init = self.noiser._wa_noise_forward(single_frame_embeds_no_norm)
                    elif self.noiser.mode == 'cc':
                        indices, noised_init = self.noiser._cc_noise_forward(single_frame_embeds_no_norm)
                    else:
                        indices, noised_init = matched_indices, single_frame_embeds_no_norm[matched_indices]
                else:
                    indices, noised_init = matched_indices, single_frame_embeds_no_norm[matched_indices]
                
                # 6. 更新物理时序状态机内的空间坐标与语义
                for cur_idx, trk_idx in zip(matched_indices_np[0], matched_indices_np[1]):
                    track = self.active_tracks[trk_idx]
                    cand_mask = masks_cur_sigmoid[0, cur_idx]
                    cand_bbox = self._mask_to_normalized_bbox(cand_mask)
                    track.update(
                        bbox=cand_bbox,
                        query_embd=single_frame_embeds[cur_idx, 0] # bs=0
                    )

            # -----------------------------------------------------------------
            # 🌟 走标准的 Transformer Layer 映射流程 (继承原有参数运算)
            # -----------------------------------------------------------------
            if i == 0 and resume is False:
                # 第一帧的前向 Decoder
                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, self.ref_proj(frame_key),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None, memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
                        )
                        output = self.transformer_ffn_layers[j](output)
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], self.ref_proj(ms_output[-1]),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None, memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
                        )
                        output = self.transformer_ffn_layers[j](output)
                        ms_output.append(output)
                self.last_reference = self.ref_proj(frame_key)
            else:
                # 后续帧的前向 Decoder
                reference = self.ref_proj(self.last_outputs[-1])
                self.last_reference = reference

                for j in range(self.num_layers):
                    if j == 0:
                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, reference, frame_key,
                            single_frame_embeds_no_norm,
                            memory_mask=None, memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
                        )
                        output = self.transformer_ffn_layers[j](output)
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], reference, frame_key,
                            single_frame_embeds_no_norm,
                            memory_mask=None, memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )
                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=None
                        )
                        output = self.transformer_ffn_layers[j](output)
                        ms_output.append(output)

            all_frames_references.append(self.last_reference)

            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
            
        outputs = torch.stack(outputs, dim=0)  # (t, l, q, b, c)
        all_frames_references = torch.stack(all_frames_references, dim=0)  # (t, q, b, c)

        mask_features_ = mask_features
        if not self.training:
            outputs = outputs[:, -1:]
            del mask_features
            
        # 批量生成最终的高精度 Mask/类别预测结果（对齐 DVIS++ 的损失计算和评测结构）
        outputs_class, outputs_masks = self.prediction(outputs, mask_features_, all_frames_references)
        
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1),  # (b, c, t, q),
           'pred_references': all_frames_references.permute(2, 3, 0, 1),  # (b, c, t, q),
        }
        
        if return_indices:
            return out, ret_indices
        else:
            return out