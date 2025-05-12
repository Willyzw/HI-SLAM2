#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from math import exp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable


def mse(img1, img2):
    return ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)


def psnr(img1, img2):
    mse = ((img1 - img2) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l1_loss_weight(network_output, gt):
    image = gt.detach().cpu().numpy().transpose((1, 2, 0))
    rgb_raw_gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    sobelx = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(rgb_raw_gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel_merge = np.sqrt(sobelx * sobelx + sobely * sobely) + 1e-10
    sobel_merge = np.exp(sobel_merge)
    sobel_merge /= np.max(sobel_merge)
    sobel_merge = torch.from_numpy(sobel_merge)[None, ...].to(gt.device)

    return torch.abs((network_output - gt) * sobel_merge).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    )
    sigma12 = (
        F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def recolor_semantic_img(rendered_seg, gt_seg, color_map=None):
    """Adjust the semantic color by assigning to the closest color refer to
       the ground truth semantic image or color dict.
    """
    rendered_seg = rendered_seg.permute(1, 2, 0) # (3, H, W) -> (H, W, 3)
    gt_seg = gt_seg.permute(1, 2, 0)
    img_shape = gt_seg.shape
    rendered_seg = rendered_seg.reshape(-1, 1, 3).type(torch.float32) # (H*W, 1, 3)

    if color_map is None:
        gt_seg = gt_seg.reshape(-1, 3)
        # Find unique colors
        color_map, _ = torch.unique(gt_seg, dim=0, return_inverse=True)
    refer_color = color_map.reshape(1, -1, 3).type(torch.float32).to(gt_seg.device) # (1, H*W, 3)

    # l1_distances = torch.sum(torch.abs(rendered_seg - refer_color), axis=2)
    l1_distances = torch.sqrt(torch.sum((rendered_seg - refer_color) ** 2, axis=2))
    # Find the index of the minimum distance for each pixel
    closest_indices = torch.argmin(l1_distances, axis=1)
    del l1_distances

    # Assign the closest color to the rendered semantic image
    rendered_seg[:, 0, :] = refer_color.squeeze(0)[closest_indices]
    rendered_seg = rendered_seg.reshape(img_shape) # (H*W, 1, 3) -> (H, W, 3)
    rendered_seg = rendered_seg.permute(2, 0, 1) # (H, W, 3) -> (3, H, W)
    
    return rendered_seg


def miou(recolored_img_pred, gt_img):
    """
    Input : 
        recolored_img_pred: torch tensor of the colored semantic image, shape (C, H, W)
        gt_img: torch tensor of the colored semantic image, shape (C, H, W)
    """
    recolored_img = recolor_semantic_img(recolored_img_pred, gt_img)
    gt_flat = gt_img.permute(1, 2, 0).view(-1, 3)
    pred_flat = recolored_img.permute(1, 2, 0).view(-1, 3)

    # Filter out [0, 0, 0] (unlabeled) pixels
    labeled_pixels = (gt_flat != torch.tensor([0, 0, 0], dtype=torch.uint8).cuda()).any(dim=1)
    gt_flat = gt_flat[labeled_pixels]
    pred_flat = pred_flat[labeled_pixels]

    unique_colors = torch.unique(gt_flat, dim=0)
    iou_per_color = []

    for color in unique_colors:
        # Skip the unlabeled color
        if torch.equal(color, torch.tensor([0, 0, 0], dtype=torch.uint8).cuda()):
            continue

        gt_matches = torch.all(gt_flat == color, dim=1)
        pred_matches = torch.all(pred_flat == color, dim=1)

        # Calculate intersection and union
        intersection = torch.logical_and(gt_matches, pred_matches).sum().item()
        union = torch.logical_or(gt_matches, pred_matches).sum().item()

        if union == 0:
            continue

        iou = intersection / union
        iou_per_color.append(iou)

    # Calculate mean IoU
    miou = sum(iou_per_color) / len(iou_per_color) if iou_per_color else 0
    return miou
