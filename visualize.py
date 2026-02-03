import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
from tqdm import tqdm


def create_overlay(image, mask, alpha=0.5):
    """Create overlay of mask on image"""
    # Ensure image is RGB
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3 and image.dtype == np.uint8:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create colored mask (red for segmentation)
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = mask  # Red channel
    
    # Blend
    overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def create_error_map(pred, target):
    """Create error visualization map"""
    # True Positive (green), False Positive (red), False Negative (blue)
    error_map = np.zeros((*pred.shape, 3), dtype=np.uint8)
    
    tp = (pred > 0.5) & (target > 0.5)
    fp = (pred > 0.5) & (target <= 0.5)
    fn = (pred <= 0.5) & (target > 0.5)
    
    error_map[tp, 1] = 255  # Green for True Positive
    error_map[fp, 0] = 255  # Red for False Positive
    error_map[fn, 2] = 255  # Blue for False Negative
    
    return error_map


def create_comparison_grid(predictions, num_samples=16, save_path='outputs/visualizations/comparison_grid.png'):
    """Create grid comparing original, ground truth, and predictions"""
    
    # Select random samples
    np.random.seed(42)
    samples = np.random.choice(predictions, min(num_samples, len(predictions)), replace=False)
    
    # Calculate grid size
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    fig = plt.figure(figsize=(20, rows*3))
    gs = GridSpec(rows, cols*3, figure=fig)
    
    for idx, sample in enumerate(samples):
        row = idx // cols
        col_group = (idx % cols) * 3
        
        # Load images
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        mask_path = sample['image_path'].replace('.png', '-mask.png')
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        pred_mask = sample['pred_binary']
        
        # Original image
        ax1 = fig.add_subplot(gs[row, col_group])
        ax1.imshow(image)
        ax1.set_title('Original', fontsize=8)
        ax1.axis('off')
        
        # Ground truth
        ax2 = fig.add_subplot(gs[row, col_group + 1])
        ax2.imshow(gt_mask, cmap='gray')
        ax2.set_title('Ground Truth', fontsize=8)
        ax2.axis('off')
        
        # Prediction
        ax3 = fig.add_subplot(gs[row, col_group + 2])
        ax3.imshow(pred_mask, cmap='gray')
        ax3.set_title('Prediction', fontsize=8)
        ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison grid saved to {save_path}")


def create_overlay_visualizations(predictions, num_samples=8, save_path='outputs/visualizations/overlay_samples.png'):
    """Create overlay visualizations"""
    
    # Select random samples
    np.random.seed(42)
    samples = np.random.choice(predictions, min(num_samples, len(predictions)), replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
    
    for idx, sample in enumerate(samples):
        # Load images
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        mask_path = sample['image_path'].replace('.png', '-mask.png')
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        pred_mask = sample['pred_binary']
        
        # Create overlays
        gt_overlay = create_overlay(image.copy(), gt_mask)
        pred_overlay = create_overlay(image.copy(), pred_mask)
        
        # Plot
        axes[0, idx].imshow(gt_overlay)
        axes[0, idx].set_title('GT Overlay', fontsize=8)
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(pred_overlay)
        axes[1, idx].set_title('Pred Overlay', fontsize=8)
        axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overlay visualizations saved to {save_path}")


def create_error_visualizations(predictions, num_samples=8, save_path='outputs/visualizations/error_maps.png'):
    """Create error map visualizations"""
    
    # Select random samples
    np.random.seed(42)
    samples = np.random.choice(predictions, min(num_samples, len(predictions)), replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
    
    for idx, sample in enumerate(samples):
        # Load ground truth mask
        mask_path = sample['image_path'].replace('.png', '-mask.png')
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Normalize
        gt_mask = (gt_mask > 127).astype(np.uint8)
        pred_mask = (sample['pred_binary'] > 127).astype(np.uint8)
        
        # Create error map
        error_map = create_error_map(pred_mask, gt_mask)
        
        # Plot prediction
        axes[0, idx].imshow(pred_mask, cmap='gray')
        axes[0, idx].set_title('Prediction', fontsize=8)
        axes[0, idx].axis('off')
        
        # Plot error map
        axes[1, idx].imshow(error_map)
        axes[1, idx].set_title('Error Map', fontsize=8)
        axes[1, idx].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='True Positive'),
        Patch(facecolor='red', label='False Positive'),
        Patch(facecolor='blue', label='False Negative')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, frameon=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Error maps saved to {save_path}")


def create_case_studies(predictions, num_samples=4, save_dir='outputs/visualizations/case_studies'):
    """Create detailed case studies"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Select samples from different categories
    carries_samples = [p for p in predictions if 'Carries' in p['category']]
    normal_samples = [p for p in predictions if 'Normal' in p['category']]
    
    np.random.seed(42)
    carries_selected = np.random.choice(carries_samples, min(num_samples//2, len(carries_samples)), replace=False)
    normal_selected = np.random.choice(normal_samples, min(num_samples//2, len(normal_samples)), replace=False)
    
    samples = list(carries_selected) + list(normal_selected)
    
    for idx, sample in enumerate(samples):
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        # Load images
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth mask
        mask_path = sample['image_path'].replace('.png', '-mask.png')
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        pred_mask = sample['pred_binary']
        
        # Normalize masks
        gt_mask_norm = (gt_mask > 127).astype(np.uint8)
        pred_mask_norm = (pred_mask > 127).astype(np.uint8)
        
        # Create visualizations
        gt_overlay = create_overlay(image.copy(), gt_mask)
        pred_overlay = create_overlay(image.copy(), pred_mask)
        error_map = create_error_map(pred_mask_norm, gt_mask_norm)
        
        # Plot
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gt_mask, cmap='gray')
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(pred_mask, cmap='gray')
        axes[0, 2].set_title('Predicted Mask')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(gt_overlay)
        axes[1, 0].set_title('Ground Truth Overlay')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(pred_overlay)
        axes[1, 1].set_title('Prediction Overlay')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(error_map)
        axes[1, 2].set_title('Error Map\n(Green=TP, Red=FP, Blue=FN)')
        axes[1, 2].axis('off')
        
        plt.suptitle(f"Case Study {idx+1}: {sample['category']} - {sample['name']}", fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'case_study_{idx+1}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Case study {idx+1} saved to {save_path}")


def main(args):
    # Create output directory
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Load predictions
    print("Loading predictions...")
    with open(os.path.join(args.output_dir, 'predictions_info.pkl'), 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    print("\n1. Creating comparison grid...")
    create_comparison_grid(predictions, 
                          num_samples=16,
                          save_path=os.path.join(viz_dir, 'comparison_grid.png'))
    
    print("\n2. Creating overlay visualizations...")
    create_overlay_visualizations(predictions,
                                 num_samples=8,
                                 save_path=os.path.join(viz_dir, 'overlay_samples.png'))
    
    print("\n3. Creating error maps...")
    create_error_visualizations(predictions,
                               num_samples=8,
                               save_path=os.path.join(viz_dir, 'error_maps.png'))
    
    print("\n4. Creating case studies...")
    create_case_studies(predictions,
                       num_samples=6,
                       save_dir=os.path.join(viz_dir, 'case_studies'))
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETED!")
    print(f"All visualizations saved to: {viz_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory containing predictions')
    
    args = parser.parse_args()
    main(args)