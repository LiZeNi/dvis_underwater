# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from ..pixel_decoder.fpn import build_pixel_decoder


@SEM_SEG_HEADS_REGISTRY.register()
class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        return_transformer_feature: bool = False,
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
        # 新增：解码器名称，用于判断是否需要传递额外参数
        transformer_decoder_name: str = "MultiScaleMaskedTransformerDecoder",
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight
        self.return_transformer_feature = return_transformer_feature

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

        # 判断是否使用需要额外参数的解码器
        # 需要 track_masks 的解码器：VideoMultiScaleMaskedTransformerDecoder_dvisPlusAttnMask
        self.use_track_masks = (
            transformer_decoder_name == "VideoMultiScaleMaskedTransformerDecoder_dvisPlusAttnMask"
        or transformer_decoder_name =="VideoMultiScaleMaskedTransformerDecoder_dvisPlusPositiveAttnMask")

        self.use_query_prompts = (
            transformer_decoder_name == "VideoMultiScaleMaskedTransformerDecoder_dvisPlusFeedback"
        )


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # 计算 predictor 输入通道
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "return_transformer_feature": cfg.MODEL.SEM_SEG_HEAD.RETURN_TRANSFORMER_FEATURE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
            "transformer_decoder_name": cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME,  # 关键新增
        }

    def forward(self, features, mask=None, clip_size=None, query_prompts=None, query_embed_prompts=None, track_masks=None):
        return self.layers(
            features, mask,
            clip_size=clip_size,
            query_prompts=query_prompts,
            query_embed_prompts=query_embed_prompts,
            track_masks=track_masks,
        )

    def layers(self, features, mask=None, clip_size=None, query_prompts=None, query_embed_prompts=None, track_masks=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)

        # 构建传递给 predictor 的参数字典，根据标志包含可选参数
        extra_args = {}
        if self.use_track_masks:
            extra_args["track_masks"] = track_masks
        if self.use_query_prompts:
            extra_args["query_prompts"] = query_prompts
            extra_args["query_embed_prompts"] = query_embed_prompts

        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            if clip_size is not None:
                predictions = self.predictor(
                    multi_scale_features, mask_features, mask,
                    clip_size=clip_size,
                    **extra_args,
                )
            else:
                predictions = self.predictor(
                    multi_scale_features, mask_features, mask,
                    **extra_args,
                )
        elif self.transformer_in_feature == "transformer_encoder":
            assert transformer_encoder_features is not None, "Please use the TransformerEncoderPixelDecoder."
            if clip_size is not None:
                predictions = self.predictor(
                    transformer_encoder_features, mask_features, mask,
                    clip_size=clip_size,
                    **extra_args,
                )
            else:
                predictions = self.predictor(
                    transformer_encoder_features, mask_features, mask,
                    **extra_args,
                )
        elif self.transformer_in_feature == "pixel_embedding":
            if clip_size is not None:
                predictions = self.predictor(
                    mask_features, mask_features, mask,
                    clip_size=clip_size,
                    **extra_args,
                )
            else:
                predictions = self.predictor(
                    mask_features, mask_features, mask,
                    **extra_args,
                )
        else:
            if clip_size is not None:
                predictions = self.predictor(
                    features[self.transformer_in_feature], mask_features, mask,
                    clip_size=clip_size,
                    **extra_args,
                )
            else:
                predictions = self.predictor(
                    features[self.transformer_in_feature], mask_features, mask,
                    **extra_args,
                )

        if self.return_transformer_feature:
            if transformer_encoder_features is not None:
                predictions['transformer_features'] = transformer_encoder_features
            else:
                predictions['transformer_features'] = multi_scale_features[0]
        return predictions

@SEM_SEG_HEADS_REGISTRY.register()
class FCCLIPHead(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            raise NotImplementedError

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask,
                                        text_classifier=features["text_classifier"], num_templates=features["num_templates"])
        else:
            raise NotImplementedError
        return predictions