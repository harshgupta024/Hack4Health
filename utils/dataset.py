import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DentalDataset(Dataset):
    """Dataset class for dental X-ray images and segmentation masks"""
    
    def __init__(self, image_dir, transform=None, is_train=True):
        """
        Args:
            image_dir: Directory containing both images and masks
            transform: Albumentations transforms
            is_train: Whether this is training data
        """
        self.image_dir = image_dir
        self.transform = transform
        self.is_train = is_train
        
        # Get all image files (not masks)
        all_files = os.listdir(image_dir)
        self.images = sorted([f for f in all_files if not 'mask' in f.lower() and f.endswith('.png')])
        
        print(f"Found {len(self.images)} images in {image_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask - handle different naming conventions
        mask_name = img_name.replace('.png', '-mask.png')
        mask_path = os.path.join(self.image_dir, mask_name)
        
        if not os.path.exists(mask_path):
            # Try alternative naming
            base_name = img_name.split('.')[0]
            mask_name = f"{base_name}-mask.png"
            mask_path = os.path.join(self.image_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            print(f"Warning: Mask not found for {img_name}, using empty mask")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Normalize mask to binary (0 or 1)
        mask = (mask > 127).astype(np.uint8)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to float tensor
        mask = mask.unsqueeze(0).float()
        
        return image, mask, img_name


def get_transforms(image_size=256):
    """Get training and validation transforms"""
    
    train_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform


def create_dataloaders(carries_dir, normal_dir, batch_size=8, image_size=256, val_split=0.2):
    """Create train and validation dataloaders"""
    
    train_transform, val_transform = get_transforms(image_size)
    
    # Load datasets from both directories
    carries_dataset = DentalDataset(carries_dir, transform=train_transform, is_train=True)
    normal_dataset = DentalDataset(normal_dir, transform=train_transform, is_train=True)
    
    # Combine datasets
    combined_dataset = torch.utils.data.ConcatDataset([carries_dataset, normal_dataset])
    
    # Split into train and validation
    total_size = len(combined_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        combined_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    
    return train_loader, val_loader