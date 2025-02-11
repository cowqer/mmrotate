import warnings

import torch
import torch.nn as nn
from mmdet.models.losses.utils import weighted_loss

from ..builder import ROTATED_LOSSES

try:
    from mmcv.ops import diff_iou_rotated_2d
except:  # noqa: E722
    diff_iou_rotated_2d = None


@weighted_loss
def rotated_ciou_loss(pred, target, eps=1e-6):
    """Rotated CIoU loss.

    Computing the CIoU loss between a set of predicted rbboxes and target
    rbboxes.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x, y, h, w, angle),
            shape (n, 5).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 5).
        eps (float): Eps to avoid log(0).
    Return:
        torch.Tensor: Loss tensor.
    """
    if diff_iou_rotated_2d is None:
        raise ImportError('Please install mmcv-full >= 1.5.0.')

    ious = diff_iou_rotated_2d(pred.unsqueeze(0), target.unsqueeze(0))
    ious = ious.squeeze(0).clamp(min=eps)

    # Calculate center distance
    pred_ctr_x = pred[:, 0]
    pred_ctr_y = pred[:, 1]
    target_ctr_x = target[:, 0]
    target_ctr_y = target[:, 1]
    center_distance = (pred_ctr_x - target_ctr_x) ** 2 + (pred_ctr_y - target_ctr_y) ** 2

    # Calculate aspect ratio
    pred_w = pred[:, 2]
    pred_h = pred[:, 3]
    target_w = target[:, 2]
    target_h = target[:, 3]
    aspect_ratio = 4 / (torch.pi ** 2) * (torch.atan(target_w / (target_h + eps)) - torch.atan(pred_w / (pred_h + eps))) ** 2

    # Calculate v and alpha
    v = aspect_ratio / (1 - ious + aspect_ratio)
    alpha = v / (1 - ious + v)

    # Calculate CIoU loss
    ciou_loss = 1 - ious + (center_distance / (pred_w ** 2 + pred_h ** 2 + eps)) + alpha * v

    return ciou_loss


@ROTATED_LOSSES.register_module()
class RotatedCIoULoss(nn.Module):
    """RotatedCIoULoss.

    Computing the CIoU loss between a set of predicted rbboxes and
    target rbboxes.
    Args:
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(self,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0):
        super(RotatedCIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

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
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 5) to (n,) to match the
            # ciou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * rotated_ciou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss