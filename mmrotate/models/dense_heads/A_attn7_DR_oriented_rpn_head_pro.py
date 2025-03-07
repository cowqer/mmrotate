# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import torch.nn as nn
from mmcv.ops import batched_nms
from mmdet.core import anchor_inside_flags, unmap
import torch.nn.functional as F

from mmrotate.core import obb2xyxy
from ..builder import ROTATED_HEADS
from .rotated_rpn_head import RotatedRPNHead
from ..utils import AdaptiveRotatedConv2d
from ..utils import RountingFunction_attn7,Gatedpconv
from . import MSCA1_ADRPOrientedRPNHead


@ROTATED_HEADS.register_module()
class MSCA7_ADRPOrientedRPNHead(MSCA1_ADRPOrientedRPNHead):
    """Oriented RPN head for Oriented R-CNN."""

    def _init_layers(self):
        super()._init_layers()
        """Initialize layers of the head."""
        self.arconv = AdaptiveRotatedConv2d(
            in_channels=self.feat_channels,
            out_channels=self.feat_channels,
            kernel_size=3, 
            padding=1,
            groups=1,
            rounting_func=RountingFunction_attn7(
                in_channels=self.feat_channels,
                kernel_number=self.kernel_number,
            ),
            kernel_number=self.kernel_number,
        )
        
    def forward_single(self, x):
        return super().forward_single(x)

    # 继承父类的其他方法
    def _get_targets_single(self, *args, **kwargs):
        return super()._get_targets_single(*args, **kwargs)

    def loss_single(self, *args, **kwargs):
        return super().loss_single(*args, **kwargs)

    def _get_bboxes_single(self, *args, **kwargs):
        return super()._get_bboxes_single(*args, **kwargs)

