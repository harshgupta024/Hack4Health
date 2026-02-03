import os
import argparse
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from utils.metrics import MetricsCalculator


def evaluate_predictions(predictions):
    """Evaluate all predictions and calculate metrics"""
    
    metrics_calc = MetricsCalculator()
    
    print("\nCalculating metrics...")
    for sample in tqdm(predictions):
        # Load ground truth mask
        mask_path = sample['image_path'].replace('.png', '-mask.png')
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Normalize masks
        gt_mask = (gt_mask > 127).astype(np.float32) / 255.0
        pred_mask = sample['pred']
        
        # Update metrics
        metrics_calc.update(pred_mask, gt_mask)
    
    return metrics_calc


def plot_metrics(metrics_calc, save_path):
    """Plot metrics as bar chart"""
    
    avg_metrics = metrics_calc.get_average_metrics()
    std_metrics = metrics_calc.get_std_metrics()
    
    # Prepare data
    metric_names = list(avg_metrics.keys())
    metric_values = list(avg_metrics.values())
    metric_stds = list(std_metrics.values())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metric_names))
    bars = ax.bar(x, metric_values, yerr=metric_stds, capsize=5, 
                   color='skyblue', edgecolor='navy', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Segmentation Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val, std) in enumerate(zip(bars, metric_values, metric_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics plot saved to {save_path}")


def plot_metric_distributions(metrics_calc, save_path):
    """Plot distributions of all metrics"""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    metrics_data = {
        'Dice Score': metrics_calc.dice_scores,
        'IoU': metrics_calc.iou_scores,
        'Precision': metrics_calc.precisions,
        'Recall': metrics_calc.recalls,
        'F1-Score': metrics_calc.f1_scores,
        'Pixel Accuracy': metrics_calc.pixel_accuracies,
        'Sensitivity': metrics_calc.sensitivities,
        'Specificity': metrics_calc.specificities,
        'Hausdorff Distance': metrics_calc.hausdorff_distances
    }
    
    for idx, (name, data) in enumerate(metrics_data.items()):
        if len(data) > 0:
            axes[idx].hist(data, bins=30, color='skyblue', edgecolor='navy', alpha=0.7)
            axes[idx].axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.3f}')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(name)
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metric distributions saved to {save_path}")


def create_confusion_matrix(predictions, save_path):
    """Create confusion matrix visualization"""
    
    # Calculate pixel-wise confusion matrix
    tp = fp = tn = fn = 0
    
    for sample in tqdm(predictions, desc="Calculating confusion matrix"):
        # Load ground truth mask
        mask_path = sample['image_path'].replace('.png', '-mask.png')
        if os.path.exists(mask_path):
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = np.zeros((256, 256), dtype=np.uint8)
        
        # Normalize
        gt_mask = (gt_mask > 127).astype(np.uint8)
        pred_mask = (sample['pred_binary'] > 127).astype(np.uint8)
        
        # Calculate
        tp += np.sum((pred_mask == 1) & (gt_mask == 1))
        fp += np.sum((pred_mask == 1) & (gt_mask == 0))
        tn += np.sum((pred_mask == 0) & (gt_mask == 0))
        fn += np.sum((pred_mask == 0) & (gt_mask == 1))
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Absolute values
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix (Absolute)')
    
    # Normalized values
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=ax2,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    ax2.set_title('Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}")


def save_metrics_report(metrics_calc, save_path):
    """Save detailed metrics report"""
    
    avg_metrics = metrics_calc.get_average_metrics()
    std_metrics = metrics_calc.get_std_metrics()
    
    with open(save_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DENTAL X-RAY SEGMENTATION - EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("SUMMARY METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<25} {'Mean':<15} {'Std Dev':<15}\n")
        f.write("-"*80 + "\n")
        
        for metric_name in avg_metrics.keys():
            avg = avg_metrics[metric_name]
            std = std_metrics[metric_name]
            f.write(f"{metric_name:<25} {avg:>6.4f}          {std:>6.4f}\n")
        
        f.write("="*80 + "\n\n")
        
        f.write("DETAILED METRICS\n")
        f.write("-"*80 + "\n\n")
        
        f.write("1. Dice Similarity Coefficient (DSC)\n")
        f.write(f"   - Measures overlap between prediction and ground truth\n")
        f.write(f"   - Range: [0, 1], Higher is better\n")
        f.write(f"   - Value: {avg_metrics['Dice Score']:.4f} ± {std_metrics['Dice Score']:.4f}\n\n")
        
        f.write("2. Intersection over Union (IoU / Jaccard Index)\n")
        f.write(f"   - Measures ratio of intersection to union\n")
        f.write(f"   - Range: [0, 1], Higher is better\n")
        f.write(f"   - Value: {avg_metrics['IoU']:.4f} ± {std_metrics['IoU']:.4f}\n\n")
        
        f.write("3. Precision\n")
        f.write(f"   - Ratio of true positives to predicted positives\n")
        f.write(f"   - Range: [0, 1], Higher is better\n")
        f.write(f"   - Value: {avg_metrics['Precision']:.4f} ± {std_metrics['Precision']:.4f}\n\n")
        
        f.write("4. Recall (Sensitivity)\n")
        f.write(f"   - Ratio of true positives to actual positives\n")
        f.write(f"   - Range: [0, 1], Higher is better\n")
        f.write(f"   - Value: {avg_metrics['Recall']:.4f} ± {std_metrics['Recall']:.4f}\n\n")
        
        f.write("5. F1-Score\n")
        f.write(f"   - Harmonic mean of precision and recall\n")
        f.write(f"   - Range: [0, 1], Higher is better\n")
        f.write(f"   - Value: {avg_metrics['F1-Score']:.4f} ± {std_metrics['F1-Score']:.4f}\n\n")
        
        f.write("6. Pixel-wise Accuracy\n")
        f.write(f"   - Ratio of correctly classified pixels\n")
        f.write(f"   - Range: [0, 1], Higher is better\n")
        f.write(f"   - Value: {avg_metrics['Pixel Accuracy']:.4f} ± {std_metrics['Pixel Accuracy']:.4f}\n\n")
        
        f.write("7. Specificity\n")
        f.write(f"   - Ratio of true negatives to actual negatives\n")
        f.write(f"   - Range: [0, 1], Higher is better\n")
        f.write(f"   - Value: {avg_metrics['Specificity']:.4f} ± {std_metrics['Specificity']:.4f}\n\n")
        
        f.write("8. Hausdorff Distance\n")
        f.write(f"   - Maximum distance between boundaries\n")
        f.write(f"   - Range: [0, ∞], Lower is better\n")
        f.write(f"   - Value: {avg_metrics['Hausdorff Distance']:.4f} ± {std_metrics['Hausdorff Distance']:.4f}\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"Metrics report saved to {save_path}")


def main(args):
    # Create output directory
    metrics_dir = os.path.join(args.output_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Load predictions
    print("Loading predictions...")
    with open(os.path.join(args.output_dir, 'predictions_info.pkl'), 'rb') as f:
        predictions = pickle.load(f)
    
    print(f"Loaded {len(predictions)} predictions")
    
    # Evaluate
    print("\nEvaluating predictions...")
    metrics_calc = evaluate_predictions(predictions)
    
    # Print metrics
    metrics_calc.print_metrics()
    
    # Save metrics report
    print("\nSaving metrics report...")
    save_metrics_report(metrics_calc, os.path.join(metrics_dir, 'metrics_summary.txt'))
    
    # Create visualizations
    print("\nCreating metric visualizations...")
    plot_metrics(metrics_calc, os.path.join(metrics_dir, 'metrics_bar_chart.png'))
    plot_metric_distributions(metrics_calc, os.path.join(metrics_dir, 'metric_distributions.png'))
    create_confusion_matrix(predictions, os.path.join(metrics_dir, 'confusion_matrix.png'))
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETED!")
    print(f"All metrics saved to: {metrics_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate segmentation results')
    
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory containing predictions')
    
    args = parser.parse_args()
    main(args)