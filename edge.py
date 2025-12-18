from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import GaussianBlur
from kornia.filters import canny, spatial_gradient
from kornia.morphology import dilation
import numpy as np
import torch
import os

def get_name(path):
    return os.path.splitext(os.path.basename(path))[0]

def get_bbox(img_size, center, crop_size):
    h, w, c = img_size
    x1 = int(center[1] - crop_size[0] / 2)
    y1 = int(center[0] - crop_size[1] / 2)

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(x1 + crop_size[0], w)
    y2 = min(y1 + crop_size[1], h)


    return [x1, y1, x2, y2]


def edge_metric(orig_path,  ai_path_list, classic_path_list, device='cpu', kernel_size=128, sigma=1, low_thresh=0.1, high_thresh=0.2):
    bboxes = []
    metric_values = []
    image_orig = read_image(orig_path, ImageReadMode.GRAY) / 255
    _, image_height, image_width = image_orig.shape
    tensors_list = []
    for classic_path, ai_path in zip(classic_path_list, ai_path_list):
        image_ai = read_image(ai_path, ImageReadMode.GRAY) / 255
        image_classic = read_image(classic_path, ImageReadMode.GRAY) / 255
        tensors_list.append(torch.stack((image_orig, image_ai, image_classic), dim=1))
    T = torch.stack(tensors_list, dim=0).squeeze()
    if T.dim() == 3:
        T = T[None, ...]
    T = T.to(device)
    image_orig = image_orig.to(device)
    canny_map = canny(image_orig[None, ...], low_threshold=low_thresh, high_threshold=high_thresh)[1]
    
    grads = spatial_gradient(T)
    transform = GaussianBlur(3, sigma=sigma).to(device)
    grads = torch.stack((transform(grads[:, :, 0]), transform(grads[:, :, 1])), dim=2)
    
    norms = torch.linalg.norm(grads, dim=2)
    norm_products = torch.stack((norms[:, 0] * norms[:, 1] + 1e-7, norms[:, 0] * norms[:, 2] + 1e-7), dim=1)
    dot_products = torch.abs(torch.stack(((grads[:, 0] * grads[:, 1]).sum(dim=1), (grads[:, 0] * grads[:, 2]).sum(dim=1)), dim=1))
    cos = dot_products / norm_products
    p_diffs = cos[:, 1] - cos[:, 0]
    p_diffs = torch.where(canny_map > 0, p_diffs, 0).transpose(0, 1)

 
    stride = 1
    conv = torch.nn.Conv2d(1, 1, (kernel_size, kernel_size), bias=False,  padding="same", padding_mode="zeros", stride=stride)
    conv.weight.data = (torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2).to(device)
    with torch.no_grad():
        cos_acum = conv(p_diffs)

    metrics = cos_acum
    metrics = metrics.cpu().squeeze(dim=1)
    metric_values = []
    bboxes = []
    crop_size = (300)

    img_name = get_name(orig_path)
    res_dict = {img_name: []}

    height, width = image_orig.shape[1], image_orig.shape[2]
    image_size = (height, width, 1)

    for triplet_idx in range(metrics.size(dim=0)):
        window = metrics[triplet_idx]

        max_value = window.max().item()
        center_y_conv, center_x_conv = torch.where(window == max_value)
        center_x = center_x_conv[0].item()
        center_y = center_y_conv[0].item()


        left = max(0, center_x - crop_size // 2)
        top = max(0, center_y - crop_size // 2)

        right = left + crop_size
        bottom = top + crop_size

        if right > image_width:
            left = max(0, image_width - crop_size)
            right = image_width

        if bottom > image_height:
            top = max(0, image_height - crop_size)
            bottom = image_height

        bbox = [left, top, right, bottom]

        ai_path = ai_path_list[triplet_idx]
        classic_path = classic_path_list[triplet_idx]

      
        res_dict[img_name].append( {
                'metric_value': max_value,
                'bbox': bbox,
                'ai_path': ai_path,
                'classic_path': classic_path,
                'gt_path': orig_path
        })

    return res_dict

