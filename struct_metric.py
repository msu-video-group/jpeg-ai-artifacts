import cv2
import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from functools import wraps
from time import time

# Try to import MSSSIM from piqa, fallback to skimage SSIM if not available
try:
    from piqa import MS_SSIM
    HAS_PIQA_MSSSIM = True
except ImportError:
    HAS_PIQA_MSSSIM = False

# Try to import SSIM from skimage
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te-ts))
        return result
    return wrap


def compute_local_msssim(gt_image, sample_image, window_size=11, stride=1, device='cpu'):
    """
    Compute local MSSSIM heatmap using sliding window approach.
    
    Args:
        gt_image: ground truth image tensor (1, 3, H, W) or (3, H, W)
        sample_image: sample image tensor (1, 3, H, W) or (3, H, W)
        window_size: size of sliding window
        stride: stride for sliding window
        device: device for computations
    
    Returns:
        Heatmap tensor (1, 1, H, W) with local MSSSIM values
    """
    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)
    if sample_image.dim() == 3:
        sample_image = sample_image.unsqueeze(0)
    
    gt_image = gt_image.to(device)
    sample_image = sample_image.to(device)
    
    _, _, H, W = gt_image.shape
    
    # Ensure images have 3 channels for MS_SSIM
    # If grayscale, replicate to 3 channels
    if gt_image.shape[1] == 1:
        gt_image = gt_image.repeat(1, 3, 1, 1)
    if sample_image.shape[1] == 1:
        sample_image = sample_image.repeat(1, 3, 1, 1)
    
    # Initialize heatmap
    heatmap = torch.zeros((1, 1, H, W), device=device)
    
    # Use SSIM from skimage for local map computation (most reliable and efficient)
    if HAS_SKIMAGE:
        # Convert to grayscale for skimage SSIM
        if gt_image.shape[1] == 3:
            # Convert RGB to grayscale
            gt_gray = (0.299 * gt_image[:, 0:1] + 0.587 * gt_image[:, 1:2] + 0.114 * gt_image[:, 2:3])
            sample_gray = (0.299 * sample_image[:, 0:1] + 0.587 * sample_image[:, 1:2] + 0.114 * sample_image[:, 2:3])
        else:
            gt_gray = gt_image
            sample_gray = sample_image
        
        # Convert to numpy for skimage
        gt_np = gt_gray[0, 0].cpu().numpy()
        sample_np = sample_gray[0, 0].cpu().numpy()
        
        # Ensure values are in [0, 1] range
        gt_np = np.clip(gt_np, 0, 1)
        sample_np = np.clip(sample_np, 0, 1)
        
        # Compute SSIM map with local windows
        # full=True returns the full SSIM image instead of just the mean
        win_size = min(window_size, min(H, W))
        if win_size % 2 == 0:
            win_size -= 1  # win_size must be odd
        if win_size < 3:
            win_size = 3
        
        _, ssim_map = ssim(gt_np, sample_np, full=True, data_range=1.0, win_size=win_size)
        
        # Convert to difference map (1 - similarity, so higher values = more difference)
        ssim_map = 1.0 - ssim_map
        
        # Convert back to torch tensor
        heatmap[0, 0] = torch.from_numpy(ssim_map).float().to(device)
    elif HAS_PIQA_MSSSIM:
        # Fallback: compute MS_SSIM for full image (slower but works)
        # Convert to 3 channels for MS_SSIM
        if gt_image.shape[1] == 1:
            gt_3ch = gt_image.repeat(1, 3, 1, 1)
            sample_3ch = sample_image.repeat(1, 3, 1, 1)
        else:
            gt_3ch = gt_image
            sample_3ch = sample_image
        
        msssim = MS_SSIM().to(device).eval()
        with torch.no_grad():
            score = msssim(gt_3ch, sample_3ch)
        
        # For full image, create uniform heatmap based on global score
        # This is not ideal but works as fallback
        heatmap.fill_(1.0 - score.item())
    else:
        # Simple fallback: use mean squared error
        # Convert to grayscale if needed
        if gt_image.shape[1] == 3:
            gt_gray = (0.299 * gt_image[:, 0:1] + 0.587 * gt_image[:, 1:2] + 0.114 * gt_image[:, 2:3])
            sample_gray = (0.299 * sample_image[:, 0:1] + 0.587 * sample_image[:, 1:2] + 0.114 * sample_image[:, 2:3])
        else:
            gt_gray = gt_image
            sample_gray = sample_image
        diff = (gt_gray - sample_gray) ** 2
        heatmap = diff.mean(dim=1, keepdim=True)
    
    return heatmap


def compute_si_heatmap(image_path, window_size=11, device='cpu'):
    """
    Compute Structural Information (SI) heatmap.
    SI measures local structural complexity/variation.
    
    Args:
        image_path: path to image
        window_size: size of window for computing local variance
        device: device for computations
    
    Returns:
        SI heatmap tensor (1, 1, H, W)
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to tensor
    img_tensor = torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    
    H, W = img_tensor.shape[2], img_tensor.shape[3]
    
    # Compute local variance as a proxy for structural information
    # Higher variance = more structural information
    kernel = torch.ones(1, 1, window_size, window_size, device=device) / (window_size ** 2)
    pad_size = window_size // 2
    
    # Pad image with reflect padding
    img_padded = torch.nn.functional.pad(img_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Compute local mean
    local_mean = torch.nn.functional.conv2d(img_padded, kernel, padding=0)
    
    # Pad the squared difference for variance computation
    diff_squared = (img_tensor - local_mean) ** 2
    diff_padded = torch.nn.functional.pad(diff_squared, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    
    # Compute local variance (structural information)
    local_variance = torch.nn.functional.conv2d(diff_padded, kernel, padding=0)
    
    # Normalize to [0, 1]
    local_variance = (local_variance - local_variance.min()) / (local_variance.max() - local_variance.min() + 1e-8)
    
    # Return as (1, 1, H, W) for consistency with other heatmaps
    return local_variance


def vqmt_heatmap2torch(gt_path, sample_path, metric="msssim", color_space="Y", save_dir="./vqmt", device='cpu'):
    """
    Compute MSSSIM heatmap without vqmt.
    
    Args:
        gt_path: path to ground truth image
        sample_path: path to sample image
        metric: metric name (currently only supports "msssim")
        color_space: color space (currently unused, kept for compatibility)
        save_dir: unused, kept for compatibility
        device: device for computations
    
    Returns:
        Heatmap tensor (1, 1, H, W)
    """
    # Load images
    gt_img = cv2.imread(gt_path)
    sample_img = cv2.imread(sample_path)
    
    if gt_img is None or sample_img is None:
        return None
    
    # Convert BGR to RGB
    gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
    sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor and normalize to [0, 1]
    transform = transforms.ToTensor()
    gt_tensor = transform(gt_img).unsqueeze(0).to(device)
    sample_tensor = transform(sample_img).unsqueeze(0).to(device)
    
    # Compute local MSSSIM heatmap
    heatmap = compute_local_msssim(gt_tensor, sample_tensor, window_size=11, stride=4, device=device)
    
    return heatmap


def get_si_heatmap(gt_path, metric="si", color_space="Y", save_dir="./vqmt", device='cpu'):
    """
    Compute SI (Structural Information) heatmap without vqmt.
    
    Args:
        gt_path: path to ground truth image
        metric: metric name (currently only supports "si")
        color_space: color space (currently unused, kept for compatibility)
        save_dir: unused, kept for compatibility
        device: device for computations
    
    Returns:
        SI heatmap tensor (1, 1, H, W)
    """
    return compute_si_heatmap(gt_path, window_size=11, device=device)


def process_si_heatmap(heatmap, kernel_size=4, t=0.3, device='cpu'):
    """
    Process SI heatmap with convolution and thresholding.
    
    Args:
        heatmap: SI heatmap tensor
        kernel_size: size of convolution kernel
        t: threshold value
        device: device for computations
    
    Returns:
        Processed heatmap
    """
    conv1 = torch.nn.Conv2d(
        1, 1, (kernel_size, kernel_size),
        bias=False, padding="same", padding_mode="zeros"
    )
    conv1.weight.data = (
        torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)
        / (kernel_size ** 2)
    )
    conv1.to(device)

    with torch.no_grad():
        heatmap = conv1(heatmap)
        heatmap = conv1(heatmap)
        heatmap = torch.where(heatmap < t, torch.zeros_like(heatmap), torch.ones_like(heatmap))
    return heatmap


def process_heatmap(si_heatmap, trad_heatmap, neural_heatmap, t=0.1, kernel_size=128, device="cpu", save_dir='./'):
    """
    Process heatmaps to compute final difference map.
    
    Args:
        si_heatmap: structural information heatmap
        trad_heatmap: traditional method heatmap
        neural_heatmap: neural method heatmap
        t: threshold value
        kernel_size: size of convolution kernel
        device: device for computations
        save_dir: unused, kept for compatibility
    
    Returns:
        Processed difference heatmap
    """
    hm_diff = neural_heatmap - trad_heatmap
    hm_diff = torch.where(hm_diff < t, torch.zeros_like(hm_diff), hm_diff)
    
    conv1 = torch.nn.Conv2d(
        1, 1, (kernel_size, kernel_size),
        bias=False, padding="same", padding_mode="zeros"
    )
    conv1.weight.data = (
        torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.float32)
        / (kernel_size ** 2)
    )
    conv1.to(device)
    
    with torch.no_grad():
        hm_diff = conv1(hm_diff)
        hm_diff = torch.where(hm_diff < t, torch.zeros_like(hm_diff), hm_diff)
    
    return hm_diff


def get_final_heatmap(gt_path, ai_path_list, classic_path_list, t1=0.0, t2=0.0, color_space='Y', device='cpu', base_metric='MSSSIM', kernel_size=32):
    """
    Compute final heatmap combining SI and MSSSIM differences.
    
    Args:
        gt_path: path to ground truth image
        ai_path_list: list of paths to AI-processed images
        classic_path_list: list of paths to classically-processed images
        t1: threshold for final heatmap
        t2: threshold for SI heatmap
        color_space: color space (unused, kept for compatibility)
        device: device for computations
        base_metric: base metric name (should be 'MSSSIM')
        kernel_size: kernel size for processing
    
    Returns:
        Final heatmap tensor (batch_size, 1, H, W)
    """
    si_batch = []
    neural_batch = []
    trad_batch = []

    si_heatmap = get_si_heatmap(gt_path, color_space=color_space, device=device)
    
    if si_heatmap is None:
        print(f"Warning: Could not compute SI heatmap for {gt_path}")
        return None

    for ai_path, classic_path in zip(ai_path_list, classic_path_list):
        neural_heatmap = vqmt_heatmap2torch(gt_path, ai_path, device=device, metric=base_metric)
        trad_heatmap = vqmt_heatmap2torch(gt_path, classic_path, device=device, metric=base_metric)
        
        if neural_heatmap is None or trad_heatmap is None:
            print(f"Warning: Could not compute heatmaps for {ai_path} or {classic_path}")
            continue

        si_batch.append(si_heatmap)
        neural_batch.append(neural_heatmap)
        trad_batch.append(trad_heatmap)

    if not trad_batch:
        return None

    # All heatmaps have shape (1, 1, H, W), we need to remove the first batch dimension
    # before stacking, so we get (batch_size, 1, H, W) instead of (batch_size, 1, 1, H, W)
    trad_batch = torch.stack([h.squeeze(0) for h in trad_batch], dim=0)
    neural_batch = torch.stack([h.squeeze(0) for h in neural_batch], dim=0)
    si_batch = torch.stack([si.squeeze(0) for si in si_batch], dim=0)

    si_batch2 = process_si_heatmap(si_batch, t=t2, device=device)
    res_heat_map = process_heatmap(si_batch2, trad_batch, neural_batch, t=t1, kernel_size=kernel_size, device=device)
    return res_heat_map


def get_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def get_bbox(img_size, center, crop_size):
    """
    Get bounding box from center point.
    
    Args:
        img_size: image size (h, w, c)
        center: center point (y, x)
        crop_size: crop size (height, width)
    
    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    h, w, c = img_size

    x1 = int(center[1] - crop_size[0] / 2)
    y1 = int(center[0] - crop_size[1] / 2)

    x1 = max(0, min(x1, w - crop_size[0]))
    y1 = max(0, min(y1, h - crop_size[1]))

    x2 = x1 + crop_size[0]
    y2 = y1 + crop_size[1]

    return [x1, y1, x2, y2]


CROP_SIZE = (300, 300)


def struct_metric(gt_path, ai_path_list, classic_path_list, device='cpu'):
    """
    Compute structural metric for artifact detection.
    
    Args:
        gt_path: path to ground truth image
        ai_path_list: list of paths to AI-processed images
        classic_path_list: list of paths to classically-processed images
        device: device for computations
    
    Returns:
        Dictionary with results containing metric values and bounding boxes
    """
    img_name = get_name(gt_path)
    res_dict = {img_name: []}
    
    heatmap = get_final_heatmap(gt_path, ai_path_list, classic_path_list, device=device)
    
    if heatmap is None:
        print(f"Warning: Could not compute heatmap for {gt_path}")
        return res_dict

    batch_size, c, h, w = heatmap.shape
    values, indices = torch.max(heatmap.view(batch_size, -1), dim=1)
    indices = torch.stack([indices // w, indices % w], dim=-1)

    imsize = cv2.imread(gt_path).shape

    for max_index, value, ai_path, classic_path in zip(indices, values, ai_path_list, classic_path_list):
        bbox = get_bbox(imsize, max_index, CROP_SIZE)
        res_dict[img_name].append({
            'metric_value': value.item(),
            'bbox': bbox,
            'ai_path': ai_path,
            'classic_path': classic_path,
            'gt_path': gt_path
        })
    
    return res_dict
