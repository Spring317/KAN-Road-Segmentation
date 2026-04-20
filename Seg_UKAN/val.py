#! /data/cxli/miniconda3/envs/th200/bin/python
import argparse
import os
from glob import glob
import random
import numpy as np
import matplotlib.pyplot as plt

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations import Normalize
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from collections import OrderedDict
from ultralytics import YOLO

import archs

from dataset import BDD100KDataset, BDD100K_NUM_CLASSES, BDD100K_COLOR_DICT, BDD100K_CLASSES, colorize_mask, onehot_to_mask
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time

from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='bdd100k_UKAN', help='model name')
    parser.add_argument('--output_dir', default='outputs', help='ouput dir')
    parser.add_argument('--num_vis', default=10, type=int, help='number of images to visualize')
    parser.add_argument('--yolo_exp', default=None, help='YOLO experiment name (e.g., yolo_exp2, yolo_exp3, yolo_exp4). If provided, validates this YOLO model.')
    parser.add_argument('--cpu', action='store_true', help='Use CPU for evaluation')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size for evaluation (default: 1)')
            
    args = parser.parse_args()

    return args

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def plot_results(images, gt_masks, pred_masks, img_ids, save_dir, num_vis=10):
    """
    Plot and save visualization of original image, ground truth mask, and predicted mask.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_images = min(num_vis, len(images))
    
    for idx in range(num_images):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        img = images[idx]
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Ground truth mask (colorized)
        gt_colored = colorize_mask(gt_masks[idx])
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        # Predicted mask (colorized)
        pred_colored = colorize_mask(pred_masks[idx])
        axes[2].imshow(pred_colored)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{img_ids[idx]}_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Saved {num_images} visualization images to {save_dir}")


def plot_class_legend(save_dir):
    """Plot and save a legend showing all classes and their colors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color patches for legend
    for class_id, class_name in BDD100K_CLASSES.items():
        color = BDD100K_COLOR_DICT[class_id]
        ax.barh(class_id, 1, color=color, edgecolor='black', linewidth=0.5)
        ax.text(1.1, class_id, f'{class_id}: {class_name}', va='center', fontsize=10)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, 19.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title('BDD100K Segmentation Classes', fontsize=14)
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'class_legend.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved class legend to {save_dir}/class_legend.png")


def main():
    seed_torch()
    args = parse_args()

    with open(f'{args.output_dir}/{args.name}/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    device = torch.device('cpu' if args.cpu else 'cuda')

    if args.yolo_exp:
        yolo_model_path = os.path.join('runs', args.yolo_exp, 'weights', 'best.pt')
        print(f"Loading YOLO model from {yolo_model_path}")
        model = YOLO(yolo_model_path)
    else:
        # Create model with no_kan flag if present in config
        no_kan = config.get('no_kan', False)
        kan_type = config.get('kan_type', 'FasterKAN')
        model = archs.__dict__[config['arch']](config['num_classes'], config['input_channels'], config['deep_supervision'], embed_dims=config['input_list'], no_kan=no_kan, kan_type=kan_type)
        model = model.to(device)

    # BDD100K dataset paths
    bdd100k_base = '/storage/student11/bdd100k_seg/bdd100k/seg'
    
    # Get image IDs from the BDD100K validation set - use masks as the source of truth
    val_mask_paths = sorted(glob(os.path.join(bdd100k_base, 'labels', 'val', '*.png')))
    val_img_ids = []
    for p in val_mask_paths:
        mask_name = os.path.splitext(os.path.basename(p))[0]
        img_id = mask_name.replace('_train_id', '')
        val_img_ids.append(img_id)
    
    print(f"Validation samples: {len(val_img_ids)}")

    if not args.yolo_exp:
        # Load model weights
        model_path = f'{args.output_dir}/{args.name}/checkpoint_best.pth'
        print(f"Loading model from {model_path}")
        ckpt = torch.load(model_path)

        try:        
            model.load_state_dict(ckpt)
        except:
            print("Pretrained model keys:", ckpt.keys())
            print("Current model keys:", model.state_dict().keys())

            pretrained_dict = {k: v for k, v in ckpt.items() if k in model.state_dict()}
            current_dict = model.state_dict()
            diff_keys = set(current_dict.keys()) - set(pretrained_dict.keys())

            print("Difference in model keys:")
            for key in diff_keys:
                print(f"Key: {key}")

            model.load_state_dict(ckpt, strict=False)
            
        model.eval()

    val_transform = Compose([
        Resize(config['input_h'], config['input_w']),
        Normalize(),
    ])

    val_dataset = BDD100KDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join(bdd100k_base, 'images', 'val'),
        mask_dir=os.path.join(bdd100k_base, 'labels', 'val'),
        img_ext='.jpg',
        mask_ext='.png',
        num_classes=BDD100K_NUM_CLASSES,
        transform=val_transform,
        mask_suffix='_train_id'
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    iou_avg_meter = AverageMeter()
    dice_avg_meter = AverageMeter()
    map_avg_meter = AverageMeter()
    
    # Per-class IoU tracking
    class_iou_sum = np.zeros(BDD100K_NUM_CLASSES)
    class_iou_count = np.zeros(BDD100K_NUM_CLASSES)

    # Store images for visualization
    vis_images = []
    vis_gt_masks = []
    vis_pred_masks = []
    vis_img_ids = []
    
    timer = []
    output_dir = os.path.join(args.output_dir, config['name'], 'out_val')
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.to(device)
            target = target.to(device)
            start = time.perf_counter()
            
            if args.yolo_exp:
                # YOLO Inference (needs raw image paths)
                raw_img_paths = [os.path.join(bdd100k_base, 'images', 'val', f"{img_id}.jpg") for img_id in meta['img_id']]
                results = model(raw_img_paths, verbose=False, device=device)
                end = time.perf_counter()
                
                # Convert YOLO results to semantic segmentation format (B, C, H, W)
                H, W = config['input_h'], config['input_w']
                output_pseudo_batch = []
                pred_masks_batch = []
                
                for res in results:
                    semantic_mask = np.zeros((H, W), dtype=np.uint8)
                    pseudo_output = np.zeros((config['num_classes'], H, W), dtype=np.float32)
                    
                    if res.masks is not None:
                        masks = res.masks.data.cpu().numpy()
                        classes = res.boxes.cls.cpu().numpy().astype(int)
                        conf = res.boxes.conf.cpu().numpy()
                        
                        sorted_idx = np.argsort(conf)
                        for idx in sorted_idx:
                            c = classes[idx]
                            mask = masks[idx]
                            if mask.shape != (H, W):
                                mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
                            valid_mask = mask > 0.5
                            semantic_mask[valid_mask] = c
                            pseudo_output[c, valid_mask] = 1.0
                            
                    pred_masks_batch.append(semantic_mask)
                    output_pseudo_batch.append(pseudo_output)
                    
                output = torch.tensor(np.array(output_pseudo_batch)).to(device)
                output_sigmoid = output
                pred_masks = np.array(pred_masks_batch)
            else:
                model = model.to(device)
                output = model(input)
                end = time.perf_counter()
                output_sigmoid = torch.sigmoid(output)
                pred_masks = torch.argmax(output_sigmoid, dim=1).cpu().numpy()  # (B, H, W)

            iou, dice, _ = iou_score(output, target)
            iou_avg_meter.update(iou, input.size(0))
            dice_avg_meter.update(dice, input.size(0))
            infer_time = end - start 
            timer.append(infer_time)
            
            # Calculate mAP
            batch_map = 0
            valid_classes = 0
            
            target_np = target.cpu().numpy()
            output_sigmoid_np = output_sigmoid.cpu().numpy()
            
            for c in range(config.get('num_classes', BDD100K_NUM_CLASSES)):
                y_true = target_np[:, c, ...].flatten()
                y_scores = output_sigmoid_np[:, c, ...].flatten()
                if y_true.sum() > 0:
                    ap = average_precision_score(y_true, y_scores)
                    batch_map += ap
                    valid_classes += 1
            if valid_classes > 0:
                map_avg_meter.update(batch_map / valid_classes, 1)

            # Get ground truth masks (argmax over one-hot)
            gt_masks = torch.argmax(target, dim=1).cpu().numpy()  # (B, H, W)
            
            # Get original images for visualization (denormalize)
            input_np = input.cpu().numpy()
            
            # Save predictions and collect for visualization
            for i, (pred, gt, img_id) in enumerate(zip(pred_masks, gt_masks, meta['img_id'])):
                # Save prediction as colored mask
                pred_colored = colorize_mask(pred)
                cv2.imwrite(os.path.join(output_dir, f'{img_id}_pred.png'), cv2.cvtColor(pred_colored, cv2.COLOR_RGB2BGR))
                
                # Save prediction as class indices
                cv2.imwrite(os.path.join(output_dir, f'{img_id}_pred_class.png'), pred.astype(np.uint8))
                
                # Collect for visualization (only first num_vis samples)
                if len(vis_images) < args.num_vis:
                    # Denormalize image for visualization
                    img = input_np[i].transpose(1, 2, 0)  # (H, W, C)
                    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    img = np.clip(img, 0, 1)
                    
                    vis_images.append(img)
                    vis_gt_masks.append(gt)
                    vis_pred_masks.append(pred)
                    vis_img_ids.append(img_id)
	
    fps = len(val_dataset) / sum(timer) if sum(timer) > 0 else 0

    # Print overall metrics
    print('\n' + '='*50)
    print(f'Model: {config["name"]}' if not args.yolo_exp else f'YOLO Exp: {args.yolo_exp}')
    print('='*50)
    print(f'Overall IoU: {iou_avg_meter.avg:.4f}')
    print(f'Overall Dice: {dice_avg_meter.avg:.4f}')
    print(f'Overall mAP: {map_avg_meter.avg:.4f}')
    print(f'Average FPS: {fps:.2f}')
    print('='*50)
    
    # Plot visualization results
    vis_dir = os.path.join(args.output_dir, config['name'], 'visualizations')
    plot_results(vis_images, vis_gt_masks, vis_pred_masks, vis_img_ids, vis_dir, args.num_vis)
    
    # Plot class legend
    plot_class_legend(vis_dir)
    
    # Create a summary plot with multiple samples
    num_summary = min(6, len(vis_images))
    fig, axes = plt.subplots(num_summary, 3, figsize=(15, 5 * num_summary))
    
    for idx in range(num_summary):
        # Original image
        axes[idx, 0].imshow(vis_images[idx])
        axes[idx, 0].set_title(f'Image: {vis_img_ids[idx]}' if idx == 0 else '')
        axes[idx, 0].axis('off')
        
        # Ground truth
        gt_colored = colorize_mask(vis_gt_masks[idx])
        axes[idx, 1].imshow(gt_colored)
        axes[idx, 1].set_title('Ground Truth' if idx == 0 else '')
        axes[idx, 1].axis('off')
        
        # Prediction
        pred_colored = colorize_mask(vis_pred_masks[idx])
        axes[idx, 2].imshow(pred_colored)
        axes[idx, 2].set_title('Prediction' if idx == 0 else '')
        axes[idx, 2].axis('off')
    
    plt.suptitle(f'BDD100K Val Results - IoU: {iou_avg_meter.avg:.4f}, Dice: {dice_avg_meter.avg:.4f}, mAP: {map_avg_meter.avg:.4f}, FPS: {fps:.2f}' if not args.yolo_exp else f'YOLO {args.yolo_exp} Val - IoU: {iou_avg_meter.avg:.4f}, mAP: {map_avg_meter.avg:.4f}, FPS: {fps:.2f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'summary_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary plot to {vis_dir}/summary_results.png")
    
    # Save metrics to file
    metrics_path = os.path.join(args.output_dir, config['name'], 'val_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {config['name'] if not args.yolo_exp else args.yolo_exp}\n")
        f.write(f"Overall IoU: {iou_avg_meter.avg:.4f}\n")
        f.write(f"Overall Dice: {dice_avg_meter.avg:.4f}\n")
        f.write(f"Overall mAP: {map_avg_meter.avg:.4f}\n")
        f.write(f"Average FPS: {fps:.2f}\n")
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()
