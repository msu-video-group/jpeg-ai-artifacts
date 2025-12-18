import os

import cv2
import numpy as np
import torch
from PIL import Image


def rgb2xyz(rgb_image, device):
    mt = torch.tensor([[0.4124, 0.3576, 0.1805],
                       [0.2126, 0.7152, 0.0722],
                       [0.0193, 0.1192, 0.9504]]).to(device)
    mask1 = (rgb_image > 0.0405).float()
    mask1_no = 1-mask1
    temp_img = mask1 * (((rgb_image + 0.055) / 1.055) ** 2.4)
    temp_img = temp_img+mask1_no * (rgb_image / 12.92)
    temp_img = 100 * temp_img

    res = torch.matmul(mt, temp_img.permute(1, 0, 2, 3).contiguous().view(3, -1)).view(3,
                                                                                       rgb_image.size(0), rgb_image.size(2), rgb_image.size(3)).permute(1, 0, 2, 3)
    return res


def xyz_lab(xyz_image, device):
    mask_value_0 = (xyz_image == 0).float().to(device)
    mask_value_0_no = 1-mask_value_0
    xyz_image = xyz_image+0.0001*mask_value_0
    mask1 = (xyz_image > 0.008856).float()
    mask1_no = 1-mask1
    res = mask1 * (xyz_image) ** (1 / 3)
    res = res+mask1_no * ((7.787 * xyz_image) + (16 / 116))
    res = res*mask_value_0_no
    return res


def rgb2lab_diff(rgb_image, device):
    '''
    Function to convert a batch of image tensors from RGB space to CIELAB space.    
    parameters: xn, yn, zn are the CIE XYZ tristimulus values of the reference white point. 
    Here use the standard Illuminant D65 with normalization Y = 100.
    '''
    rgb_image = rgb_image.to(device)
    res = torch.zeros_like(rgb_image)
    xyz_image = rgb2xyz(rgb_image, device)

    xn = 95.0489
    yn = 100
    zn = 108.8840

    x = xyz_image[:, 0, :, :]
    y = xyz_image[:, 1, :, :]
    z = xyz_image[:, 2, :, :]

    L = 116*xyz_lab(y/yn, device) - 16
    a = 500*(xyz_lab(x/xn, device) - xyz_lab(y/yn, device))
    b = 200*(xyz_lab(y/yn, device) - xyz_lab(z/zn, device))
    res[:, 0, :, :] = L
    res[:, 1, :, :] = a
    res[:, 2, :, :] = b

    return res


def degrees(n): return n * (180. / np.pi)
def radians(n): return n * (np.pi / 180.)


def hpf_diff(x, y):
    mask1 = ((x == 0) * (y == 0)).float()
    mask1_no = 1-mask1

    tmphp = degrees(torch.atan2(x*mask1_no, y*mask1_no))
    tmphp1 = tmphp * (tmphp >= 0).float()
    tmphp2 = (360+tmphp) * (tmphp < 0).float()

    return tmphp1+tmphp2


def dhpf_diff(c1, c2, h1p, h2p):

    mask1 = ((c1 * c2) == 0).float()
    mask1_no = 1-mask1
    res1 = (h2p - h1p)*mask1_no*(torch.abs(h2p - h1p) <= 180).float()
    res2 = ((h2p - h1p) - 360) * ((h2p - h1p) > 180).float()*mask1_no
    res3 = ((h2p - h1p)+360) * ((h2p - h1p) < -180).float()*mask1_no

    return res1+res2+res3


def ahpf_diff(c1, c2, h1p, h2p):

    mask1 = ((c1 * c2) == 0).float()
    mask1_no = 1-mask1
    mask2 = (torch.abs(h2p - h1p) <= 180).float()
    mask2_no = 1-mask2
    mask3 = (torch.abs(h2p + h1p) < 360).float()
    mask3_no = 1-mask3

    res1 = (h1p + h2p) * mask1_no * mask2
    res2 = (h1p + h2p + 360.) * mask1_no * mask2_no * mask3
    res3 = (h1p + h2p - 360.) * mask1_no * mask2_no * mask3_no
    res = (res1+res2+res3)+(res1+res2+res3)*mask1
    return res*0.5


def ciede2000_diff(lab1, lab2, device):
    '''
    CIEDE2000 metric to claculate the color distance map for a batch of image tensors defined in CIELAB space

    '''

    lab1 = lab1.to(device)
    lab2 = lab2.to(device)

    L1 = lab1[:, 0, :, :]
    A1 = lab1[:, 1, :, :]
    B1 = lab1[:, 2, :, :]
    L2 = lab2[:, 0, :, :]
    A2 = lab2[:, 1, :, :]
    B2 = lab2[:, 2, :, :]
    kL = 1
    kC = 1
    kH = 1

    mask_value_0_input1 = ((A1 == 0)*(B1 == 0)).float()
    mask_value_0_input2 = ((A2 == 0)*(B2 == 0)).float()
    mask_value_0_input1_no = 1-mask_value_0_input1
    mask_value_0_input2_no = 1-mask_value_0_input2
    B1 = B1+0.0001*mask_value_0_input1
    B2 = B2+0.0001*mask_value_0_input2

    C1 = torch.sqrt((A1 ** 2.) + (B1 ** 2.))
    C2 = torch.sqrt((A2 ** 2.) + (B2 ** 2.))

    aC1C2 = (C1 + C2) / 2.
    G = 0.5 * (1. - torch.sqrt((aC1C2 ** 7.) / ((aC1C2 ** 7.) + (25 ** 7.))))
    a1P = (1. + G) * A1
    a2P = (1. + G) * A2
    c1P = torch.sqrt((a1P ** 2.) + (B1 ** 2.))
    c2P = torch.sqrt((a2P ** 2.) + (B2 ** 2.))

    h1P = hpf_diff(B1, a1P)
    h2P = hpf_diff(B2, a2P)
    h1P = h1P*mask_value_0_input1_no
    h2P = h2P*mask_value_0_input2_no

    dLP = L2 - L1
    dCP = c2P - c1P
    dhP = dhpf_diff(C1, C2, h1P, h2P)
    dHP = 2. * torch.sqrt(c1P * c2P) * torch.sin(radians(dhP) / 2.)
    mask_0_no = 1-torch.max(mask_value_0_input1, mask_value_0_input2)
    dHP = dHP*mask_0_no

    aL = (L1 + L2) / 2.
    aCP = (c1P + c2P) / 2.
    aHP = ahpf_diff(C1, C2, h1P, h2P)
    T = 1. - 0.17 * torch.cos(radians(aHP - 39)) + 0.24 * torch.cos(radians(2. * aHP)
                                                                    ) + 0.32 * torch.cos(radians(3. * aHP + 6.)) - 0.2 * torch.cos(radians(4. * aHP - 63.))
    dRO = 30. * torch.exp(-1. * (((aHP - 275.) / 25.) ** 2.))
    rC = torch.sqrt((aCP ** 7.) / ((aCP ** 7.) + (25. ** 7.)))
    sL = 1. + ((0.015 * ((aL - 50.) ** 2.)) / torch.sqrt(20. + ((aL - 50.) ** 2.)))

    sC = 1. + 0.045 * aCP
    sH = 1. + 0.015 * aCP * T
    rT = -2. * rC * torch.sin(radians(2. * dRO))

    res_square = ((dLP / (sL * kL)) ** 2.) + ((dCP / (sC * kC)) ** 2.)*mask_0_no + ((dHP / (sH * kH))
                                                                                    ** 2.)*mask_0_no + rT * (dCP / (sC * kC)) * (dHP / (sH * kH))*mask_0_no
    mask_0 = (res_square <= 0).float()
    mask_0_no = 1-mask_0
    res_square = res_square+0.0001*mask_0
    res = torch.sqrt(res_square)
    res = res*mask_0_no

    return res


def process_image2(image: np.array):
    image = image.astype(np.float64) / 255
    image = image.transpose(2, 0, 1)
    return torch.from_numpy(image).float()


def get_ciede2000_map(gt_path, sample_path, device):
    gt = cv2.imread(gt_path)
    sample = cv2.imread(sample_path)
    gt_torch2 = process_image2(gt)
    sample_torch2 = process_image2(sample)

    gt_lab = rgb2lab_diff(gt_torch2.unsqueeze(0), device=device)
    sample_lab = rgb2lab_diff(sample_torch2.unsqueeze(0), device=device)
    ciede2000_heat_map = ciede2000_diff(gt_lab, sample_lab, device=device)

    return ciede2000_heat_map

def process_map(ciede2000_heat_map,t1, t2, device, kernel_size):
    ciede2000_heat_map[ciede2000_heat_map > t2] = 0
    ciede2000_heat_map[ciede2000_heat_map < t1] = 0

    conv = torch.nn.Conv2d(1, 1, (kernel_size, kernel_size), bias=False, padding='same', padding_mode='zeros')
    conv.weight.data = (torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2).to(device)

    with torch.no_grad():
        ciede2000_heat_map = conv(ciede2000_heat_map)
    ciede2000_heat_map = ciede2000_heat_map.squeeze(0)
    ciede2000_heat_map[ciede2000_heat_map < t1] = 0

    ciede2000_heat_map = ciede2000_heat_map
    return ciede2000_heat_map / t2


def conv_map(ciede2000_heat_map,kernel_size, stride, device):
    conv = torch.nn.Conv2d(1, 1, (kernel_size, kernel_size), bias=False, stride = stride)
    conv.weight.data = (torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2).to(device)

    with torch.no_grad():
        ciede2000_heat_map = conv(ciede2000_heat_map)
    ciede2000_heat_map = ciede2000_heat_map.squeeze(0)
    return ciede2000_heat_map 


def build_heatmap(gt_path, sample_path,  device, t1 = 3, t2 = 8, kernel_size = 64):
    ciede2000_heat_map = get_ciede2000_map(gt_path, sample_path, device)
    ciede2000_heat_map = process_map(ciede2000_heat_map, t1, t2, device, kernel_size)
    ciede2000_heat_map = ciede2000_heat_map.unsqueeze(0).cpu().numpy().transpose(1, 2, 0)
    return ciede2000_heat_map

def get_bbox(delta_heatmap, img):
    image_width, image_height = img.size
    crop_size = 300
    max_value = delta_heatmap.max()
    center_y_conv, center_x_conv, _ = np.where(delta_heatmap == max_value)

    center_x = int(center_x_conv[0])
    center_y = int(center_y_conv[0])
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

    return [left, top, right, bottom]


def color_metric(orig_path, ai_path_list, classic_path_list, device='cpu'):
    def get_name(path):
        return os.path.splitext(os.path.basename(path))[0]

    img_name = get_name(orig_path)
    res_dict = {img_name: []}
    im_orig = Image.open(orig_path).convert("RGB")

    for i in range(len(classic_path_list)):
        heatmap_neur = build_heatmap(orig_path, ai_path_list[i], device)
        heatmap_vtm = build_heatmap(orig_path, classic_path_list[i], device)

        delta_heatmap = heatmap_neur - heatmap_vtm

        delta = (delta_heatmap).max().item()
        bbox = get_bbox(delta_heatmap, im_orig)

        res_dict[img_name].append({
            'metric_value': delta,
            'bbox': bbox,
            'ai_path': ai_path_list[i],
            'classic_path': classic_path_list[i],
            'gt_path': orig_path,
        })

    return res_dict
