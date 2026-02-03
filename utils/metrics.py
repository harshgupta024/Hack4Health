import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


def calculate_dice_score(pred, target, smooth=1e-6):
    """Calculate Dice Similarity Coefficient"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice


def calculate_iou(pred, target, smooth=1e-6):
    """Calculate Intersection over Union (Jaccard Index)"""
    pred = pred.flatten()
    target = target.flatten()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou


def calculate_precision_recall(pred, target, smooth=1e-6):
    """Calculate Precision and Recall"""
    pred = pred.flatten()
    target = target.flatten()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    
    return precision, recall


def calculate_f1_score(precision, recall):
    """Calculate F1-Score"""
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1


def calculate_pixel_accuracy(pred, target):
    """Calculate Pixel-wise Accuracy"""
    pred = pred.flatten()
    target = target.flatten()
    
    correct = (pred == target).sum()
    total = len(pred)
    
    accuracy = correct / total
    
    return accuracy


def calculate_sensitivity_specificity(pred, target, smooth=1e-6):
    """Calculate Sensitivity and Specificity"""
    pred = pred.flatten()
    target = target.flatten()
    
    tp = (pred * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    sensitivity = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    
    return sensitivity, specificity


def calculate_hausdorff_distance(pred, target):
    """Calculate Hausdorff Distance"""
    try:
        # Get coordinates of positive pixels
        pred_points = np.argwhere(pred > 0.5)
        target_points = np.argwhere(target > 0.5)
        
        if len(pred_points) == 0 or len(target_points) == 0:
            return float('inf')
        
        # Calculate directed Hausdorff distances
        hd1 = directed_hausdorff(pred_points, target_points)[0]
        hd2 = directed_hausdorff(target_points, pred_points)[0]
        
        # Return maximum
        return max(hd1, hd2)
    except:
        return float('inf')


class MetricsCalculator:
    """Class to calculate and store all metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.precisions = []
        self.recalls = []
        self.f1_scores = []
        self.pixel_accuracies = []
        self.sensitivities = []
        self.specificities = []
        self.hausdorff_distances = []
    
    def update(self, pred, target):
        """Update metrics with new prediction and target"""
        # Convert to numpy if needed
        if torch.is_tensor(pred):
            pred = pred.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()
        
        # Threshold predictions
        pred_binary = (pred > 0.5).astype(np.float32)
        target_binary = target.astype(np.float32)
        
        # Calculate all metrics
        dice = calculate_dice_score(pred_binary, target_binary)
        iou = calculate_iou(pred_binary, target_binary)
        precision, recall = calculate_precision_recall(pred_binary, target_binary)
        f1 = calculate_f1_score(precision, recall)
        pixel_acc = calculate_pixel_accuracy(pred_binary, target_binary)
        sensitivity, specificity = calculate_sensitivity_specificity(pred_binary, target_binary)
        hd = calculate_hausdorff_distance(pred_binary, target_binary)
        
        # Store metrics
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.precisions.append(precision)
        self.recalls.append(recall)
        self.f1_scores.append(f1)
        self.pixel_accuracies.append(pixel_acc)
        self.sensitivities.append(sensitivity)
        self.specificities.append(specificity)
        if hd != float('inf'):
            self.hausdorff_distances.append(hd)
    
    def get_average_metrics(self):
        """Get average of all metrics"""
        metrics = {
            'Dice Score': np.mean(self.dice_scores),
            'IoU': np.mean(self.iou_scores),
            'Precision': np.mean(self.precisions),
            'Recall': np.mean(self.recalls),
            'F1-Score': np.mean(self.f1_scores),
            'Pixel Accuracy': np.mean(self.pixel_accuracies),
            'Sensitivity': np.mean(self.sensitivities),
            'Specificity': np.mean(self.specificities),
            'Hausdorff Distance': np.mean(self.hausdorff_distances) if self.hausdorff_distances else 0
        }
        return metrics
    
    def get_std_metrics(self):
        """Get standard deviation of all metrics"""
        metrics = {
            'Dice Score': np.std(self.dice_scores),
            'IoU': np.std(self.iou_scores),
            'Precision': np.std(self.precisions),
            'Recall': np.std(self.recalls),
            'F1-Score': np.std(self.f1_scores),
            'Pixel Accuracy': np.std(self.pixel_accuracies),
            'Sensitivity': np.std(self.sensitivities),
            'Specificity': np.std(self.specificities),
            'Hausdorff Distance': np.std(self.hausdorff_distances) if self.hausdorff_distances else 0
        }
        return metrics
    
    def print_metrics(self):
        """Print all metrics"""
        avg_metrics = self.get_average_metrics()
        std_metrics = self.get_std_metrics()
        
        print("\n" + "="*60)
        print("SEGMENTATION METRICS")
        print("="*60)
        
        for metric_name in avg_metrics.keys():
            avg = avg_metrics[metric_name]
            std = std_metrics[metric_name]
            print(f"{metric_name:20s}: {avg:.4f} ± {std:.4f}")
        
        print("="*60 + "\n")
        
        return avg_metrics