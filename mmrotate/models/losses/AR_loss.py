# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from mmdet.models.utils import weighted_loss


@mmcv.jit(derivate=True, coderize=True)

@weighted_loss
def smooth_l1_loss(pred, target, beta=1.0):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    
        pred_w, pred_h = torch.max(bbox_pred[:, 2], bbox_pred[:, 3]), torch.min(bbox_pred[:, 2], bbox_pred[:, 3])
        target_w, target_h = torch.max(bbox_targets[:, 2], bbox_targets[:, 3]), torch.min(bbox_targets[:, 2], bbox_targets[:, 3])
        
        pred_aspect_ratio = pred_w / (pred_h + 1e-6)  
        target_aspect_ratio = target_w / (target_h + 1e-6)
        
        aspect_ratio_diff = torch.abs(pred_aspect_ratio - target_aspect_ratio)
        aspect_ratio_weight = (2.0*aspect_ratio_diff + 2.0) / (aspect_ratio_diff + 2.0)
        # aspect_ratio_weight = 2.0 - torch.exp(-1 * aspect_ratio_diff)
    """
    print(f"Shape of pred: {pred.shape}")
    print(f"Shape of target: {target.shape}")
    assert beta > 0
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    pred_w, pred_h =(pred[:, 2], pred[:, 3])
    target_w, target_h = (target[:, 2], target[:, 3])
    ###########真实的长和宽的比和
    pred_aspect_ratio = pred_w / (pred_h + 1e-6)
    target_aspect_ratio = target_w / (target_h + 1e-6)
    
    aspect_ratio_diff = torch.abs(pred_aspect_ratio - target_aspect_ratio)
    aspect_ratio_weight = (2.0*aspect_ratio_diff + 2.0) / (aspect_ratio_diff + 2.0)
    
    ###########定义为长边比短边
    aspect_ratios_pre = torch.max(pred_w, pred_h) / torch.min(pred_w, pred_h)
    aspect_ratios_tar = torch.max(target_w, target_h) / torch.min(target_w, target_h)
    
    ar = (aspect_ratios_pre + aspect_ratios_tar) / 2.0
    ar_weight = 2.0 * ar / (ar + 1.0)
    
    alpha = 0.9
    loss = loss * (alpha * aspect_ratio_weight + (1.0 - alpha) * ar_weight)
    
    return loss 
 

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def l1_loss(pred, target):
    """L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.

    Returns:
        torch.Tensor: Calculated loss
    """
    if target.numel() == 0:
        return pred.sum() * 0

    assert pred.size() == target.size()
    loss = torch.abs(pred - target)
    return loss



@LOSSES.register_module()
class arSmoothL1Loss(nn.Module):
    """Smooth L1 loss.

    Args:
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, beta=1.0, reduction='mean', loss_weight=1.0):
        super(arSmoothL1Loss, self).__init__()
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ar_weight = 1.0


    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * smooth_l1_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_bbox




@LOSSES.register_module()
class L1Loss(nn.Module):
    """L1 loss.

    Args:
        reduction (str, optional): The method to reduce the loss.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of loss.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super(L1Loss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight


    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * l1_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss_bbox
