import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(pred, target, alpha=0.25, gamma=2.0, pos_weight=10):
    """
    Focal Loss for heatmap classification
    Args:
        pred: [B, C, H, W] 预测的heatmap
        target: [B, C, H, W] 目标heatmap (0: background, 1: positive, -1: ignore)
        alpha: 平衡正负样本的权重
        pos_weight: 正样本过少的时候我们希望增加正样本的loss权重
        gamma: 聚焦参数
    """
    # 将target转换为one-hot编码
    # target: 0=background, 1=positive, -1=ignore
    valid_mask = (target != -1).float()
    target_positive = (target == 1).float()
    # 计算sigmoid
    pred_sigmoid = torch.sigmoid(pred)
    # 计算focal loss
    pt = pred_sigmoid * target_positive + (1 - pred_sigmoid) * (1 - target_positive)
    focal_weight = (1 - pt) ** gamma
    # 计算BCE loss
    bce_loss = F.binary_cross_entropy_with_logits(pred, target_positive, reduction='none')
    # 应用focal weight和alpha
    alpha_weight = alpha * target_positive * pos_weight + (1 - alpha) * (1 - target_positive)
    focal_loss = alpha_weight * focal_weight * bce_loss
    # 只计算有效区域的loss
    focal_loss = focal_loss * valid_mask
    
    # 计算平均loss
    pos_pixels = target_positive.sum()
    if pos_pixels > 0:
        focal_loss = focal_loss.sum() / pos_pixels
    else:
        focal_loss = focal_loss.sum()
    return focal_loss

# def focal_loss(pred, target, alpha=0.25, gamma=2.0):
#     """
#     Focal Loss for heatmap classification
#     Args:
#         pred: [B, C, H, W] 预测的heatmap
#         target: [B, C, H, W] 目标heatmap (0: background, 1: positive, -1: ignore)
#         alpha: 平衡正负样本的权重
#         gamma: 聚焦参数
#     """
#     # 将target转换为one-hot编码
#     # target: 0=background, 1=positive, -1=ignore
#     valid_mask = (target != -1).float()
#     target_positive = (target == 1).float()
#     # 计算sigmoid
#     pred_sigmoid = torch.sigmoid(pred)
#     # 计算focal loss
#     pt = pred_sigmoid * target_positive + (1 - pred_sigmoid) * (1 - target_positive)
#     focal_weight = (1 - pt) ** gamma
#     # 计算BCE loss
#     bce_loss = F.binary_cross_entropy_with_logits(pred, target_positive, reduction='none')
#     # 应用focal weight和alpha
#     alpha_weight = alpha * target_positive + (1 - alpha) * (1 - target_positive)
#     focal_loss = alpha_weight * focal_weight * bce_loss
#     # 只计算有效区域的loss
#     focal_loss = focal_loss * valid_mask
    
#     # 计算平均loss
#     pos_pixels = target_positive.sum()
#     if pos_pixels > 0:
#         focal_loss = focal_loss.sum() / pos_pixels
#     else:
#         focal_loss = focal_loss.sum()
#     return focal_loss


def soft_focal_loss(pred_logits, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pred = pred_logits.sigmoid()          # 
  pos_inds = gt.eq(1).float()           # 等于1的设置为正样本
  neg_inds = gt.lt(1).float()           # 小于1的设置为负样本，即使是软正样本也设置为负样本

  neg_weights = torch.pow(1 - gt, 4)    # 计算后硬负样本权重为1, 软正样本的权重设置的很小

  loss = 0

  pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss + neg_loss
  else:
    loss = loss + (pos_loss + neg_loss) / num_pos
  return loss


class FocalLossWithHeat(nn.Module):
    def __init__(self, channel_weight, positive_weight):
        super().__init__()
        weight = torch.tensor(channel_weight).view(-1, 1, 1)
        pos_weight = torch.tensor(positive_weight).view(-1, 1, 1)
        self.bce1 = nn.BCEWithLogitsLoss(reduction='none')
        self.bce2 = nn.BCEWithLogitsLoss(weight=weight, reduction='none', pos_weight=pos_weight)

    def forward(self, pd_map, gt_map, mask, reduction='sum'):
        gt_map_ = torch.where(gt_map.eq(1), 1.0, 0.0)
        p = self.bce1(pd_map, gt_map_)
        p = torch.exp(-p)
        bce_loss = self.bce2(pd_map, gt_map_)
        focal_loss = torch.pow(1 - p, 2) * bce_loss
        heat_weight = torch.pow(1 - gt_map, 4)
        heat_weight[gt_map == 1] = 1
        focal_loss = heat_weight * focal_loss
        
        if reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss


class FocalLossWithSeg(nn.Module):
    def __init__(self, channel_weight, positive_weight):
        super().__init__()
        weight = torch.tensor(channel_weight).view(-1, 1, 1)
        pos_weight = torch.tensor(positive_weight).view(-1, 1, 1)
        self.bce1 = nn.BCEWithLogitsLoss(reduction='none')
        self.bce2 = nn.BCEWithLogitsLoss(weight=weight, reduction='none', pos_weight=pos_weight)

    def forward(self, pd_map, gt_map, channel_weight, reduction='sum'):
        gt_map_ = torch.where(gt_map > 0, 1.0, 0.0)
        
        p = self.bce1(pd_map, gt_map_)
        p = torch.exp(-p)
        bce_loss = self.bce2(pd_map, gt_map_)
        focal_loss = torch.pow(1 - p, 2) * bce_loss
        focal_loss *= channel_weight
        
        if reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss


def log_l1_loss(pred, target, mask=None, eps=1e-6, reduction='none'):#这个函数写了但是没有调用，用的还是原来的l1_loss
    """
    Log-space L1 loss for dimensions (length, width, height).

    Args:
        pred: predicted lwh tensor (B, 3, H, W), raw output from the network
        target: ground truth lwh tensor (B, 3, H, W), must be positive
        mask: optional mask tensor (B, 1, H, W), 1 for valid boxes
        eps: small value to avoid log(0)
        reduction: 'sum' | 'mean' | 'none'
    """
    # log-space ground truth
    target_log = torch.log(target.clamp(min=eps))
    
    # compute L1 loss in log-space
    loss = F.l1_loss(pred, target_log, reduction='none')

    if mask is not None:
        loss = loss * mask

    if reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
