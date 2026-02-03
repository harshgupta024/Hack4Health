import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.model import UNet


def predict_image(model, image_path, device, image_size=256):
    """Predict segmentation mask for a single image"""
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Transform
    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Resize back to original size
    pred = cv2.resize(pred, (original_size[1], original_size[0]))
    
    # Threshold
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    
    return pred, pred_binary


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best Dice Score: {checkpoint.get('dice_score', 'unknown'):.4f}")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, 'predictions')
    os.makedirs(pred_dir, exist_ok=True)
    
    # Get all images
    all_images = []
    for category in ['Carries', 'Normal']:
        cat_dir = os.path.join(args.data_dir, category)
        if os.path.exists(cat_dir):
            images = [f for f in os.listdir(cat_dir) if not 'mask' in f.lower() and f.endswith('.png')]
            all_images.extend([(os.path.join(cat_dir, img), category, img) for img in images])
    
    print(f"\nFound {len(all_images)} images to predict")
    
    # Predict all images
    print("\nGenerating predictions...")
    predictions = []
    
    for img_path, category, img_name in tqdm(all_images):
        pred, pred_binary = predict_image(model, img_path, device, args.image_size)
        
        # Save prediction
        save_name = f"{category}_{img_name.replace('.png', '_pred.png')}"
        save_path = os.path.join(pred_dir, save_name)
        cv2.imwrite(save_path, pred_binary)
        
        # Store for later use
        predictions.append({
            'image_path': img_path,
            'category': category,
            'name': img_name,
            'pred_path': save_path,
            'pred': pred,
            'pred_binary': pred_binary
        })
    
    print(f"\nPredictions saved to: {pred_dir}")
    print(f"Total predictions: {len(predictions)}")
    
    # Save predictions info
    import pickle
    with open(os.path.join(args.output_dir, 'predictions_info.pkl'), 'wb') as f:
        pickle.dump(predictions, f)
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict dental segmentations')
    
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing data folders')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save predictions')
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size for prediction')
    
    args = parser.parse_args()
    main(args)