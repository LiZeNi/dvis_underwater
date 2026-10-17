# video_mask2former_transformer_decoder_fbfl.py
# Copyright (c) Facebook, Inc. and its affiliates.
# Modified for FBFL video tracking segmentation.

import torch
from torch.nn import functional as F
import einops

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

# 动态兼容导入机制：从你原本注册 DVIS_Plus Decoder 的位置导入父类
try:
    from dvis_Plus.transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_dvisPlus
except ImportError:
    try:
        from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_dvisPlus
    except ImportError:
        # 如果由于模块路径未加入环境变量导致报错，可以直接用相对路径导入
        from .video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_dvisPlus


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_FBFL(VideoMultiScaleMaskedTransformerDecoder_dvisPlus):
    """
    专门针对 FBFL (Frame-by-Frame Loop) 框架重构的时序 Transformer Decoder。
    
    1. 完美继承自 VideoMultiScaleMaskedTransformerDecoder_dvisPlus。
    2. 无需重构任何初始化结构与参数，100% 保持官方权重加载路径。
    3. 重构了 forward 逻辑，无缝打通 1D 稀疏 Prompt (Query 层面) 与 2D 稠密 Prompt (特征图层面) 注入通道。
    """

    def forward(
        self, 
        x, 
        mask_features, 
        mask=None, 
        sparse_prompts: Optional[torch.Tensor] = None, 
        dense_prompts: Optional[torch.Tensor] = None
    ):
        """
        重写的前向传播，新增对 FBFL 时序和几何先验 Prompts 的支持。

        参数:
          x (list[Tensor]): 多尺度图像特征图 [res2, res3, res4, res5]
          mask_features (Tensor): 卷积分割掩膜特征 [B, C, H, W]
          mask (Tensor or None): 忽略遮挡图 (通常为 None)
          sparse_prompts (Tensor or None): 形状为 [N_active, C] 的 1D 实例特征，代表上一帧目标的位置与特征
          dense_prompts (Tensor or None): 形状为 [1, C, H_prompt, W_prompt] 的 2D 特征图，代表全图活跃目标的物理热力分布
        """
        # ---------------------------------------------------------------------
        # 🌟 1. 2D 稠密先验特征图注入 (Pixel Feature Level Add)
        # ---------------------------------------------------------------------
        if dense_prompts is not None:
            new_x = []
            for i in range(self.num_feature_levels):
                h_t, w_t = x[i].shape[-2:]
                # 动态对齐 dense_prompt 特征的空间分辨率并直接累加
                scaled_prompt = F.interpolate(
                    dense_prompts, size=(h_t, w_t), mode="bilinear", align_corners=False
                )
                new_x.append(x[i] + scaled_prompt)
            x = new_x

        # 验证输入尺度层数是否正确 (通常为 3 层)
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask  # 屏蔽 DVIS 默认未启用的 mask 控制

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # 将特征图展平为标准的 Transformer [HW, Batch, Channels] 格式
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # 初始化当前帧的基本 Queries [Q, Batch, Channels]
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # ---------------------------------------------------------------------
        # 🌟 2. 1D 稀疏目标 Tokens 注入 (Query Feature & Positional Level Add)
        # ---------------------------------------------------------------------
        if sparse_prompts is not None:
            # 截取有效的目标数目，防止其超出设定的 query 上限
            num_active = min(sparse_prompts.shape[0], self.num_queries)
            
            # 格式化其维度至 [Q, bs, C]
            sparse_prompts_formatted = sparse_prompts[:num_active].unsqueeze(1).repeat(1, bs, 1)
            
            # 同时将特征加到前 num_active 个 Query 的特征 (Content) 与位置编码 (Position) 上
            output[:num_active] = output[:num_active] + sparse_prompts_formatted
            query_embed[:num_active] = query_embed[:num_active] + sparse_prompts_formatted

        predictions_class = []
        predictions_mask = []

        # 3. 初始 learnable queries 特征的分类与 Mask 产生 (第 0 层)
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # 4. 执行多层 Transformer Decoder 解码交互
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # 交叉注意力 (Cross-Attention)
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            # 自注意力 (Self-Attention)
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # 前馈层 (FFN)
            output = self.transformer_ffn_layers[i](
                output
            )

            # 对应层的特征预测
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # 5. 展开 BT 序列为 (B, T) 时序格式，准备输出
        bt = predictions_mask[-1].shape[0]
        bs_split = bt // self.num_frames if self.training else 1
        t = bt // bs_split
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', t=t)

        pred_embds_without_norm = einops.rearrange(output, 'q (b t) c -> b c t q', t=t)
        pred_embds = self.decoder_norm(output)
        reid_embed = self.reid_embed(pred_embds)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)
        reid_embed = einops.rearrange(reid_embed, 'q (b t) c -> b c t q', t=t)

        # 6. 整理输出字典并拼接 ReID 辅助特征
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'pred_embds': torch.cat([pred_embds, reid_embed], dim=1),
            'pred_embds_without_norm': torch.cat([pred_embds_without_norm, reid_embed], dim=1),
            'pred_reid_embed': reid_embed,
            'mask_features': mask_features
        }
        return out