import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from utils.dataset import create_dataloaders
from utils.model import get_model, CombinedLoss
from utils.metrics import MetricsCalculator


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    metrics = MetricsCalculator()
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks, _ in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate metrics
        pred = torch.sigmoid(outputs).detach()
        metrics.update(pred, masks)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    avg_metrics = metrics.get_average_metrics()
    
    return avg_loss, avg_metrics


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    metrics = MetricsCalculator()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks, _ in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            
            # Calculate metrics
            pred = torch.sigmoid(outputs)
            metrics.update(pred, masks)
            
            pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / len(dataloader)
    avg_metrics = metrics.get_average_metrics()
    
    return avg_loss, avg_metrics


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dice Score plot
    axes[0, 1].plot(history['train_dice'], label='Train Dice')
    axes[0, 1].plot(history['val_dice'], label='Val Dice')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Similarity Coefficient')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU plot
    axes[1, 0].plot(history['train_iou'], label='Train IoU')
    axes[1, 0].plot(history['val_iou'], label='Val IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('IoU')
    axes[1, 0].set_title('Intersection over Union')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # F1-Score plot
    axes[1, 1].plot(history['train_f1'], label='Train F1')
    axes[1, 1].plot(history['val_f1'], label='Val F1')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1-Score')
    axes[1, 1].set_title('F1-Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history saved to {save_path}")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader = create_dataloaders(
        carries_dir=args.carries_dir,
        normal_dir=args.normal_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = get_model(device)
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': []
    }
    
    best_dice = 0.0
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("="*60)
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['Dice Score'])
        history['val_dice'].append(val_metrics['Dice Score'])
        history['train_iou'].append(train_metrics['IoU'])
        history['val_iou'].append(val_metrics['IoU'])
        history['train_f1'].append(train_metrics['F1-Score'])
        history['val_f1'].append(val_metrics['F1-Score'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Dice: {train_metrics['Dice Score']:.4f} | Val Dice: {val_metrics['Dice Score']:.4f}")
        print(f"Train IoU: {train_metrics['IoU']:.4f} | Val IoU: {val_metrics['IoU']:.4f}")
        
        # Save best model
        if val_metrics['Dice Score'] > best_dice:
            best_dice = val_metrics['Dice Score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_score': best_dice,
                'val_metrics': val_metrics
            }, os.path.join(args.model_dir, 'best_model.pth'))
            print(f"✓ Best model saved! Dice: {best_dice:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.model_dir, f'checkpoint_epoch_{epoch+1}.pth'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'final_model.pth'))
    
    # Plot training history
    plot_training_history(history, os.path.join(args.output_dir, 'training_history.png'))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Models saved in: {args.model_dir}")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dental segmentation model')
    
    # Data arguments
    parser.add_argument('--carries_dir', type=str, default='data/Carries',
                        help='Directory containing caries images')
    parser.add_argument('--normal_dir', type=str, default='data/Normal',
                        help='Directory containing normal images')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size (default: 256)')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split (default: 0.2)')
    
    # Output arguments
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    
    args = parser.parse_args()
    main(args)