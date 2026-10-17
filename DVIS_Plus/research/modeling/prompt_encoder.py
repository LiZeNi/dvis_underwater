# fbfl_prompt_encoder.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from typing import Optional, Tuple
import torch
from torch.nn import functional as F

# 直接继承原版的 PromptEncoder
from research.modeling.sam2_utils.prompt_encoder import PromptEncoder


class FBFLPromptEncoder(PromptEncoder):
    """
    FBFLPromptEncoder 继承自 SAM 2 的原生 PromptEncoder。
    
    保持了父类的所有初始化逻辑和参数接口，可以直接加载 SAM 2 的预训练权重。
    仅通过新增方法，实现了将多目标 Box 和 Mask 转换为 DVIS++ 时序跟踪分割所需的单目标 Token。
    """
    
    # 提示：由于没有新增的初始化变量，我们甚至不需要重写 __init__，
    # Python 会自动调用父类 PromptEncoder 的 __init__，确保权重加载无误。

    def _embed_boxes_to_single_token(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        将 Bbox 投影为 1 个 Token。
        调用父类的 _embed_boxes (生成2个角点的 Token [N, 2, C])，
        通过均值池化将其降维为 Mask2Former 适配的单目标 [N, C] 向量。
        """
        # corner_embedding 形状: [N, 2, embed_dim]
        corner_embedding = self._embed_boxes(boxes)
        # 融合左右上/下两个角点的空间信息 -> [N, embed_dim]
        box_tokens = corner_embedding.mean(dim=1)
        return box_tokens

    def _embed_masks_to_tokens(self, masks: torch.Tensor) -> torch.Tensor:
        """
        利用父类 mask_downscaling 的卷积特征提取能力，
        将多目标 Mask [N, 1, H, W] 汇聚为各自实例对应的 1D 空间先验 Token [N, embed_dim]。
        """
        # dense_mask_feats 形状: [N, embed_dim, H_feat, W_feat]
        dense_mask_feats = self._embed_masks(masks)
        # 通过自适应均值池化，将其汇聚为 instance Token -> [N, embed_dim]
        mask_tokens = F.adaptive_avg_pool2d(dense_mask_feats, (1, 1)).flatten(1)
        return mask_tokens

    def forward_fbfl(
        self,
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        专门针对 FBFL 自回归环路优化的前向接口。
        
        输入：
          boxes (torch.Tensor or None): [N_active, 4], 目标框坐标
          masks (torch.Tensor or None): [N_active, 1, H, W], 实例级别的 Mask 掩膜

        输出：
          sparse_queries (torch.Tensor): [N_active, embed_dim] 的 1D 特征，
                                         Element-wise 直接加在 Mask2Former 对应的 Object Queries 上。
          dense_embeddings (torch.Tensor): [1, embed_dim, H_feat, W_feat] 的 2D 稠密先验，
                                           直接加在主干提取的 Pixel Features 图像特征图上。
        """
        N_active = 0
        if boxes is not None:
            N_active = boxes.shape[0]
        elif masks is not None:
            N_active = masks.shape[0]

        # 1. 若当前没有追踪的目标 (例如首帧)，返回占位向量
        if N_active == 0:
            sparse_queries = torch.zeros((1, self.embed_dim), device=self._get_device())
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
            return sparse_queries, dense_embeddings

        # 2. 生成多目标 1D 空间先验 Tokens (整合 Box 和 Mask)
        query_prompts = torch.zeros((N_active, self.embed_dim), device=self._get_device())
        if boxes is not None:
            query_prompts = query_prompts + self._embed_boxes_to_single_token(boxes)
        if masks is not None:
            query_prompts = query_prompts + self._embed_masks_to_tokens(masks)

        # 3. 空间 2D 稠密激活先验图 (合并全图活跃区域)
        if masks is not None:
            # 取多目标 Mask 的空间最大值进行合并 -> [1, 1, H, W]
            combined_mask, _ = torch.max(masks, dim=0, keepdim=True)
            dense_embeddings = self._embed_masks(combined_mask)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return query_prompts, dense_embeddings