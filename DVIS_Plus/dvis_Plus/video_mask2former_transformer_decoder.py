import torch
from torch.nn import functional as F
from detectron2.config import configurable
from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder, MLP
import einops
import fvcore.nn.weight_init as weight_init

def debug_tensor(name, tensor):
    if tensor is None:
        print(f"{name}: None")
        return
    if torch.isnan(tensor).any():
        print(f"!!! {name} contains NaN !!!")
    if tensor.dtype == torch.bool:
        # 对于布尔张量，打印 True 比例和形状
        true_ratio = tensor.float().mean().item() if tensor.numel() > 0 else 0.0
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, True_ratio={true_ratio:.4f}")
    else:
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"min={tensor.min().item() if tensor.numel()>0 else 'N/A'}, "
              f"max={tensor.max().item() if tensor.numel()>0 else 'N/A'}, "
              f"mean={tensor.mean().item() if tensor.numel()>0 else 'N/A'}")


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_dvis(VideoMultiScaleMaskedTransformerDecoder):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames: int,
    ):
        super().__init__(
            in_channels=in_channels, 
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            num_frames=num_frames,
        )

        # use 2D positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # expand BT to B, T  
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', t=t)

        pred_embds_without_norm = einops.rearrange(output, 'q (b t) c -> b c t q', t=t)
        pred_embds = self.decoder_norm(output)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'pred_embds': pred_embds,
            'pred_embds_without_norm': pred_embds_without_norm,
            'mask_features': mask_features
        }
        
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_minvis(VideoMultiScaleMaskedTransformerDecoder_dvis):

    def forward(self, x, mask_features, mask=None):
        out = super(VideoMultiScaleMaskedTransformerDecoder_minvis, self).forward(x, mask_features, mask=mask)
        del out['mask_features']
        return out


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_dvisPlus(VideoMultiScaleMaskedTransformerDecoder):

    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            pre_norm: bool,
            mask_dim: int,
            enforce_input_project: bool,
            # video related
            num_frames: int,
            # ct related
            num_reid_head_layers,
            reid_hidden_dim,
    ):
        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            num_frames=num_frames,
        )

        # use 2D positional embedding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # add ct head
        if num_reid_head_layers > 0:
            self.reid_embed = MLP(
                hidden_dim, reid_hidden_dim, hidden_dim, num_reid_head_layers)
            for layer in self.reid_embed.layers:
                weight_init.c2_xavier_fill(layer)
        else:
            self.reid_embed = torch.nn.Identity()  # do nothing

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["num_frames"] = cfg.INPUT.SAMPLING_FRAME_NUM

        ret["reid_hidden_dim"] = cfg.MODEL.MASK_FORMER.REID_HIDDEN_DIM
        ret["num_reid_head_layers"] = cfg.MODEL.MASK_FORMER.NUM_REID_HEAD_LAYERS
        return ret


    def forward(self, x, mask_features, mask=None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # expand BT to B, T
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b t) q c -> b t q c', t=t)

        pred_embds_without_norm = einops.rearrange(output, 'q (b t) c -> b c t q', t=t)
        pred_embds = self.decoder_norm(output)
        reid_embed = self.reid_embed(pred_embds)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)
        reid_embed = einops.rearrange(reid_embed, 'q (b t) c -> b c t q', t=t)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            # 'pred_embds': pred_embds,
            # 'pred_embds_without_norm': pred_embds_without_norm,
            'pred_embds': torch.cat([pred_embds, reid_embed], dim=1),
            'pred_embds_without_norm': torch.cat([pred_embds_without_norm, reid_embed], dim=1),
            'pred_reid_embed': reid_embed,
            'mask_features': mask_features
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0,
                                                                                                         1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_dvisPlusFeedback(VideoMultiScaleMaskedTransformerDecoder_dvisPlus):
    """
    带有在线轨迹反馈机制（Query Prompting）的 DVIS-Plus Transformer 解码器。
    
    允许在 forward 阶段接收前序追踪器提纯后的轨迹向量（Denoised Tracklet Queries）作为 Prompt，
    强行替换或引导前 K 个 Object Queries 的注意力，从而实现密集特征的时序一致性增强。
    """

    # 🟢 动态复用父类的 __init__ 和 @classmethod from_config，无需在此重复编写

    def forward(self, x, mask_features, mask=None, query_prompts=None, query_embed_prompts=None):
        """
        Args:
            x: 多尺度特征图列表 (list[Tensor])
            mask_features: 密集特征图
            mask: 掩码 (默认为 None)
            query_prompts: 🟢 新增。上一个时间步中，已被追踪目标的黄金去噪特征 (Tensor)。
                           支持 [K, B, C] 或 [B, K, C] 格式。
            query_embed_prompts: 🟢 新增。对应的轨迹位置编码 Prompt (Tensor)。
        """
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # Q x N x C  (即 num_queries x Batch_Size x Hidden_Dim)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # ===================================================================
        # 🟢 【核心突破：在线时序闭环反馈 / Query Prompt 注入】
        # ===================================================================
        if query_prompts is not None:
            # A. 兼容性维度检查。如果是 Batch-First [B, K, C] 格式，自动转换为 [K, B, C] 满足 Decoder 要求
            if query_prompts.dim() == 3 and query_prompts.shape[0] == bs:
                query_prompts = query_prompts.permute(1, 0, 2)
            
            K = query_prompts.shape[0]
            assert K <= self.num_queries, f"Prompt queries count {K} exceeds max queries {self.num_queries}"
            
            # B. 覆盖策略：用上一帧时序滤波去噪后的轨迹特征，替换掉原有的前 K 个静态 learnable 初始特征
            # 这使分割器在这一帧的交叉注意力中获得了“历史视觉和空间记忆”
            output = torch.cat([query_prompts, output[K:]], dim=0)
            
            # C. 如果同时传入了上一帧该物体的三维位置编码 Prompt，同步进行覆盖对齐
            if query_embed_prompts is not None:
                if query_embed_prompts.dim() == 3 and query_embed_prompts.shape[0] == bs:
                    query_embed_prompts = query_embed_prompts.permute(1, 0, 2)
                query_embed = torch.cat([query_embed_prompts, query_embed[K:]], dim=0)
        # ===================================================================

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # 逐层进行 Transformer 交叉注意力、自注意力和前馈网络计算
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # 交叉注意力 (Cross-Attention)：在这里，包含了历史轨迹记忆的 output 就会开始主动检索特征图
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # 展开视频的 Batch-Time (BT) 格式回 B, T 空间
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i, pred_class in enumerate(predictions_class):
            predictions_class[i] = einops.rearrange(pred_class, '(b t) q c -> b t q c', t=t)

        pred_embds_without_norm = einops.rearrange(output, 'q (b t) c -> b c t q', t=t)
        pred_embds = self.decoder_norm(output)
        reid_embed = self.reid_embed(pred_embds)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)
        reid_embed = einops.rearrange(reid_embed, 'q (b t) c -> b c t q', t=t)

        # 输出格式与 DVIS-Plus 完美保持对齐
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


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_dvisPlusFeedbackConcat(VideoMultiScaleMaskedTransformerDecoder_dvisPlus):
    """
    支持时序轨迹提示“拼接机制”（Query Concatenation）的反馈版 DVIS-Plus 解码器。
    
    不同于替换前 K 个，该版本直接将上一帧去噪后的 K 个 Track 提示拼接在原装 N 个 Query 的最前方，
    使得总 Query 数量变为 N + K。这能够保障原装 N 个 learnable 探测器对于新游入画面的目标的检测完整度。
    """

    # 自动复用父类的 __init__ 和 from_config，无需重复编写代码

    def forward(self, x, mask_features, mask=None, query_prompts=None, query_embed_prompts=None):
        """
        Args:
            x: 多尺度特征图列表 (list[Tensor])
            mask_features: 密集特征图
            mask: 掩码 (默认为 None)
            query_prompts: 🟢 新增。上一个时间步中，已被追踪目标的去噪 Prompt (Tensor)。
                           支持 [K, B, C] 或 [B, K, C] 格式。
            query_embed_prompts: 🟢 新增。对应的轨迹位置编码 Prompt (Tensor)。
        """
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # 初始 Q x N x C  (即 num_queries x Batch_Size x Hidden_Dim)
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # ===================================================================
        # 🟢 【核心设计：在线时序反馈 / Query Prompt 直接拼接（Concat）】
        # ===================================================================
        if query_prompts is not None:
            # A. 维度兼容转换。如果是 Batch-First [B, K, C] 格式，自动转换为 [K, B, C]
            if query_prompts.dim() == 3 and query_prompts.shape[0] == bs:
                query_prompts = query_prompts.permute(1, 0, 2)
            
            K = query_prompts.shape[0]
            
            # B. 🟢 拼接策略：将去噪特征直接加到最前端，当前输入总 Queries 数量自动扩充为 K + N 个
            output = torch.cat([query_prompts, output], dim=0)
            
            # C. 🟢 拼接对应位置编码（P.E.）以保持形状完全对齐
            if query_embed_prompts is not None:
                if query_embed_prompts.dim() == 3 and query_embed_prompts.shape[0] == bs:
                    query_embed_prompts = query_embed_prompts.permute(1, 0, 2)
                query_embed = torch.cat([query_embed_prompts, query_embed], dim=0)
            else:
                # 🟢 防御性设计：如果 Tracker 没输出对应位置编码，复制前 K 个位置编码，防止形状冲突崩溃
                query_embed = torch.cat([query_embed[:K], query_embed], dim=0)
        # ===================================================================

        predictions_class = []
        predictions_mask = []

        # 初始预测（自适应处理 K + N 维度）
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output,
            mask_features,
            attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        # 逐层运行 Transformer 结构
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # 交叉注意力：包含时序记忆的 output 在此进行密集检索
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # 展开视频的 Batch-Time (BT) 格式回 B, T 空间
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)

        for i, pred_class in enumerate(predictions_class):
            predictions_class[i] = einops.rearrange(pred_class, '(b t) q c -> b t q c', t=t)

        pred_embds_without_norm = einops.rearrange(output, 'q (b t) c -> b c t q', t=t)
        pred_embds = self.decoder_norm(output)
        reid_embed = self.reid_embed(pred_embds)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)
        reid_embed = einops.rearrange(reid_embed, 'q (b t) c -> b c t q', t=t)

        # 完美兼容原本 DVIS-Plus 的数据返回结构（仅在 Q 轴上自适应扩宽了 K 个）
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


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_dvisPlusAttnMask(VideoMultiScaleMaskedTransformerDecoder_dvisPlus):
    """
    支持将轨迹 mask 作为额外注意力掩码的 DVIS-Plus 解码器。
    用法：在 forward 时传入 track_masks (形状为 (B, Q, T, H, W) 的 logits)，
    它将在每一层交叉注意力中与原始预测掩码生成的 attn_mask 进行 OR 操作，
    强制解码器关注被轨迹标记为前景的区域。
    """

    def _prepare_track_masks_for_attention(self, pred_masks):
        B, Q, T, H, W = pred_masks.shape
        track_masks = torch.full_like(pred_masks, -1e4)

        for q in range(Q):
            others = pred_masks.clone()
            others[:, q, ...] = -1e4
            max_other_logits, _ = others.max(dim=1, keepdim=False)
            track_masks[:, q, ...] = max_other_logits
        return track_masks

    def forward(self, x, mask_features, mask=None, track_masks=None):
        # 与父类相同的前期处理
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask  # mask 不使用

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape   # bs = B*T

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # 预处理 track_masks：为每个特征级别生成对应的 bool mask
        track_masks_per_level = []


        if track_masks is not None:
            track_mask=sself._prepare_track_masks_for_attention(track_masks)
            # track_masks 形状 (B, Q, T, H, W) -> (B*T, Q, H, W)
            B, Q, T, H, W = track_masks.shape
            track_masks_bt = track_masks.permute(0, 2, 1, 3, 4).reshape(bs, Q, H, W)
            # 对每个特征级别下采样并转为 bool mask
            for lvl_size in size_list:
                tm = F.interpolate(track_masks_bt, size=lvl_size, mode="bilinear", align_corners=False)
                # 使用 sigmoid 将 logits 转为概率，阈值 0.5 生成 bool mask (True 表示被遮挡)
                tm_bool = (tm.sigmoid() > 0.5).bool()
                # 展平 HW -> (bs, Q, HW)
                tm_bool = tm_bool.flatten(2)
                # 扩展多头：重复 num_heads 次并 reshape 为 (bs*num_heads, Q, HW)
                tm_bool = tm_bool.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
                track_masks_per_level.append(tm_bool)

        else:
            track_masks_per_level = [None] * self.num_feature_levels


        predictions_class = []
        predictions_mask = []

        # 初始预测
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )

        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            # 原有处理：将完全无响应的 attn_mask 置为 False (允许所有位置)
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # 融合轨迹 mask
            if track_masks_per_level[level_index] is not None:
                # attn_mask 形状: (bs*num_heads, Q, HW)，track_mask 形状一致
                # 使用 OR 操作：任何一方为 True 则位置被遮挡
                attn_mask = attn_mask | track_masks_per_level[level_index]
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # 交叉注意力
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            # 自注意力
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            # 生成下一层使用的 attn_mask（注意：这里不再需要融合 track_mask，会在下一次循环开始时融合）
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )


            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # 后处理与父类相同
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)
        for i, pred_class in enumerate(predictions_class):
            predictions_class[i] = einops.rearrange(pred_class, '(b t) q c -> b t q c', t=t)

        pred_embds_without_norm = einops.rearrange(output, 'q (b t) c -> b c t q', t=t)
        pred_embds = self.decoder_norm(output)
        reid_embed = self.reid_embed(pred_embds)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)
        reid_embed = einops.rearrange(reid_embed, 'q (b t) c -> b c t q', t=t)

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


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_dvisPlusPositiveAttnMask(
    VideoMultiScaleMaskedTransformerDecoder_dvisPlus
):
    """
    高强健性正向注意力引导版本：使用轨迹掩码作为正偏置，引导查询关注自身实例区域。
    
    🌟【双通道隔离 + 精度对齐 + 哨兵行熔断方案】：
    1. 完美补回被遗漏的 attn_mask 全掩膜安全重置行（第 923 行起），彻底拦截第 0 步因空 Query 产生的 NaN 崩溃！
    2. 当 track_masks 为 None（第一阶段训练）时：100% 保持使用原生 Boolean Mask。
       完全保留 PyTorch 官方高速硬件加速路径（如 cuDNN/FlashAttention），确保一阶段训练与原装 Baseline 绝对一致。
    3. 当 track_masks 激活时：使用与特征输入相同的动态 Dtype (dtype=src[level_index].dtype) 初始化 float 偏置，
       彻底消除混合精度（AMP）下的类型晋升冲突，确保多卡反向传播 100% 稳定通畅。
    """

    def forward(self, x, mask_features, mask=None, track_masks=None, attn_bias_scale=1.0):
        # 与父类相同的前期处理
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        del mask  # mask 不使用

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape   # bs = B*T

        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # 预处理 track_masks：为每个特征级别生成加性偏置（float mask）
        track_masks_bias_per_level = []
        has_positive_bias = False
        
        if track_masks is not None:
            has_positive_bias = True
            # track_masks 形状 (B, Q, T, H, W) -> (B*T, Q, H, W)
            B, Q, T, H, W = track_masks.shape
            track_masks_bt = track_masks.permute(0, 2, 1, 3, 4).reshape(bs, Q, H, W)
            for lvl_size in size_list:
                tm = F.interpolate(track_masks_bt, size=lvl_size, mode="bilinear", align_corners=False)
                
                # 🌟 强健性安全保护 1：对输入的 logits 限制在安全阈值 [-15.0, 15.0] 内
                # 防止极大或极小的输入 logits 导致 sigmoid 在 FP16 下发生数值饱和崩溃（underflow）
                tm = torch.clamp(tm, min=-15.0, max=15.0)
                
                # 将 logits 转为概率
                prob = tm.sigmoid()
                
                # 计算加性偏置：在安全的区间内计算 log 投影
                bias = attn_bias_scale * torch.log(prob + 1e-8)
                # 展平 HW -> (bs, Q, HW)
                bias = bias.flatten(2)
                # 扩展多头：重复 num_heads 次并 reshape 为 (bs*num_heads, Q, HW)
                bias = bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
                track_masks_bias_per_level.append(bias)

        predictions_class = []
        predictions_mask = []

        # 初始预测
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
            output, mask_features, attn_mask_target_size=size_list[0]
        )
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            
            # 🌟 核心修复 1：补回原生被遗漏的最关键安全行！
            # 必须在每一层循环开始时，对原生的 bool attn_mask 进行全掩膜判定与熔断重置！
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            
            # 🌟 动态双通道分流控制与精度安全对齐 🌟
            if has_positive_bias:
                # 🌟 通道 A（反馈模式）：转换为与特征图输入 [src] 完全对齐的动态 Dtype，避免混合精度类型碰撞
                if attn_mask.dtype == torch.bool:
                    attn_mask_float = torch.zeros_like(attn_mask, dtype=src[level_index].dtype) # 🌟 动态对齐精度
                    attn_mask_float[attn_mask] = -10000.0
                else:
                    attn_mask_float = attn_mask
                
                # 叠加正向轨迹偏置
                if track_masks_bias_per_level[level_index] is not None:
                    # 确保 bias 与特征图精度对齐
                    attn_mask_float = attn_mask_float + track_masks_bias_per_level[level_index].to(attn_mask_float.dtype)
                
                # 后置全掩膜安全拦截
                all_masked = (attn_mask_float <= -9999.0).all(dim=-1, keepdim=True)
                attn_mask_float = torch.where(all_masked, torch.zeros_like(attn_mask_float), attn_mask_float)
                
                current_attn_mask = attn_mask_float
            else:
                # 🌟 通道 B（第一阶段无损原生模式）：
                # 100% 保持使用原生布尔 attn_mask，不进行任何浮点数转换
                # 这可以让 PyTorch 直接在底层调用硬件加速通路，保持一阶段的绝对平顺与稳定！
                current_attn_mask = attn_mask

            # 交叉注意力（current_attn_mask 在一阶段是原装 bool，在二阶段是绝对精度的 float）
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=current_attn_mask,
                memory_key_padding_mask=None,
                pos=pos[level_index], query_pos=query_embed
            )

            # 自注意力
            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            # 生成下一层使用的 attn_mask（原始 bool）
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(
                output,
                mask_features,
                attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels]
            )
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # 后处理与父类相同
        bt = predictions_mask[-1].shape[0]
        bs = bt // self.num_frames if self.training else 1
        t = bt // bs
        for i in range(len(predictions_mask)):
            predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b t) q h w -> b q t h w', t=t)
        for i, pred_class in enumerate(predictions_class):
            predictions_class[i] = einops.rearrange(pred_class, '(b t) q c -> b t q c', t=t)

        pred_embds_without_norm = einops.rearrange(output, 'q (b t) c -> b c t q', t=t)
        pred_embds = self.decoder_norm(output)
        reid_embed = self.reid_embed(pred_embds)
        pred_embds = einops.rearrange(pred_embds, 'q (b t) c -> b c t q', t=t)
        reid_embed = einops.rearrange(reid_embed, 'q (b t) c -> b c t q', t=t)

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