#!/usr/bin/env python3
"""
A script for running metrics for detecting artifacts and visualizing the results.
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch


def get_image_paths(data_dir):
    """
    Gets paths to images from test_data.
    Returns dictionary: {image_name: {'orig': path, 'ai': [paths], 'classic': [paths]}}
    """
    data_dir = Path(data_dir)
    orig_dir = data_dir / 'orig'
    ai_dir = data_dir / 'ai'
    classic_dir = data_dir / 'classic'
    
    orig_files = sorted(orig_dir.glob('*.png')) + sorted(orig_dir.glob('*.jpg'))
    
    images_dict = {}
    
    for orig_path in orig_files:
        img_name = orig_path.stem
        
        ai_files = sorted(ai_dir.glob(f'{img_name}.*'))
        classic_files = sorted(classic_dir.glob(f'{img_name}.*'))
        
        if ai_files and classic_files:
            images_dict[img_name] = {
                'orig': str(orig_path),
                'ai': [str(f) for f in ai_files],
                'classic': [str(f) for f in classic_files]
            }
    
    return images_dict


def draw_bbox_with_label(image_path, bbox, metric_value, output_path, label_prefix=''):
    """
    Draws a bounding box and signs the metric value on the image.
    
    Args:
        image_path: path to the original image
        bbox: [left, top, right, bottom]
        metric_value: the metric value to display
        output_path: the path to save the result.
        label_prefix: a prefix for the caption (for example, the name of the metric)
    """

    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    left, top, right, bottom = bbox
    
    draw.rectangle([left, top, right, bottom], outline='red', width=3)
    
    if label_prefix:
        label = f'{label_prefix}: {metric_value:.4f}'
    else:
        label = f'{metric_value:.4f}'
    
    try:
        font_size = max(20, int(img.width / 40))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    bbox_text = draw.textbbox((0, 0), label, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    text_x = left
    text_y = top - text_height - 5 if top > text_height + 5 else bottom + 5
    if text_y + text_height > img.height:
        text_y = max(0, top - text_height - 5)
    
    draw.rectangle(
        [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
        fill='yellow',
        outline='red',
        width=2
    )
    
    draw.text((text_x, text_y), label, fill='black', font=font)
    
    img.save(output_path)


def visualize_results(results, output_dir, metric_name):
    """
    Visualizes the metric results: draws a bbox and signs the metric values.
    
    Args:
        results: dictionary of results from yandex.metrica
        output_dir: directory for saving results
        metric_name: the name of the metric
    """
    output_dir = Path(output_dir)
    metric_output_dir = output_dir / metric_name
    metric_output_dir.mkdir(parents=True, exist_ok=True)
    
    for img_name, result_list in results.items():
        for idx, result in enumerate(result_list):
            metric_value = result['metric_value']
            bbox = result['bbox']

            # Use AI image path instead of original (metrics are measured for AI-compressed images)
            ai_image_path = result['ai_path']
            
            output_filename = f'{img_name}_{idx}.png'
            output_path = metric_output_dir / output_filename
            
            draw_bbox_with_label(
                ai_image_path,
                bbox,
                metric_value,
                output_path,
                label_prefix=metric_name
            )
            
            print(f'Saved: {output_path}')


def run_metric(metric_name, images_dict, device='cpu', **kwargs):
    """
    Runs the specified metric on all images.
    
    Args:
        metric_name: the name of the metric ('color', 'edge', 'struct', 'text')
        images_dict: dictionary with paths to images
        device: a device for computing
        **kwargs: additional parameters for yandex.metrica
    
    Returns:
        Dictionary of results from all images
    """
    all_results = {}
    
    # Lazy import of metrics - only import the one that's needed
    available_metrics = ['color', 'edge', 'struct', 'text']
    if metric_name not in available_metrics:
        raise ValueError(f'Unknown metric: {metric_name}. Available: {available_metrics}')
    
    # Import only the required metric
    if metric_name == 'color':
        from color import color_metric as metric_func
    elif metric_name == 'edge':
        from edge import edge_metric as metric_func
    elif metric_name == 'struct':
        from struct_metric import struct_metric as metric_func
    elif metric_name == 'text':
        from text import text_metric as metric_func
    
    print(f'Launching metric: {metric_name}')
    
    if metric_name == 'text':
        # MMOCR import will happen inside text_metric function
        # We'll catch errors when calling the metric function
        pass
    
    for img_name, paths in images_dict.items():
        print(f'Image Processing: {img_name}')
        
        orig_path = paths['orig']
        ai_paths = paths['ai']
        classic_paths = paths['classic']
        
        min_len = min(len(ai_paths), len(classic_paths))
        ai_paths = ai_paths[:min_len]
        classic_paths = classic_paths[:min_len]
        
        try:
            result = metric_func(orig_path, ai_paths, classic_paths, device=device, **kwargs)

            for key, value in result.items():
                if key in all_results:
                    all_results[key].extend(value)
                else:
                    all_results[key] = value
                    
        except (RuntimeError, AssertionError, AttributeError) as e:
            error_msg = str(e)
            if 'mmcv' in error_msg.lower() or 'MMCV' in error_msg or '__version__' in error_msg:
                print(f'Error: MMOCR requires properly installed mmcv. {error_msg}')
                if metric_name == 'text':
                    print('Please install compatible mmcv version: mmcv>=2.0.0rc4,<2.1.0')
                    print('Try: pip uninstall mmcv mmcv-full && pip install "mmcv>=2.0.0rc4,<2.1.0"')
                    print('Or skip the text metric if not needed.')
                break  # Stop processing this metric entirely
            else:
                print(f'Error during processing {img_name}: {e}')
                import traceback
                traceback.print_exc()
                continue
        except Exception as e:
            print(f'Error during processing {img_name}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Launching artifact detection metrics')
    parser.add_argument('--metric', type=str, choices=['color', 'edge', 'struct', 'text'],
                       help='The name of the metric to launch')
    parser.add_argument('--all', action='store_true',
                       help='Run all metrics')
    parser.add_argument('--data_dir', type=str, default='./test_data',
                       help='The path to the test data directory (default: ./test_data)')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='The path to the directory to save the results (default: ./results)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Computing device (cpu/cuda) (default: cpu)')
    
    args = parser.parse_args()
    
    if not args.metric and not args.all:
        parser.print_help()
        print('\nError: specify --metric or --all')
        return
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f'Error: the {data_dir} directory does not exist')
        return
    
    print(f'Uploading images from {data_dir}...')
    images_dict = get_image_paths(data_dir)
    
    if not images_dict:
        print('Error: no images found in test_data')
        return
    
    print(f'{len(images_dict)} images found')
    
    if args.all:
        metrics_to_run = ['color', 'edge', 'struct', 'text']
    else:
        metrics_to_run = [args.metric]
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Warning: CUDA is unavailable, CPU is in use')
        args.device = 'cpu'

    for metric_name in metrics_to_run:
        print(f'\n{"="*50}')
        print(f'Metric: {metric_name}')
        print(f'{"="*50}')
        
        try:
            results = run_metric(metric_name, images_dict, device=args.device)
            
            if results:
                print(f'\nVisualization of the results for {metric_name}...')
                visualize_results(results, args.output_dir, metric_name)
                print(f'The results are saved in {args.output_dir}/{metric_name}/')
            else:
                print(f'No results for the metric {metric_name}')
                
        except Exception as e:
            print(f'Error when executing the metric {metric_name}: {e}')
            import traceback
            traceback.print_exc()
            continue
    
    print(f'\n{"="*50}')
    print('Done!')


if __name__ == '__main__':
    main()

