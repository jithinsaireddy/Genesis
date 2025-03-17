"""Dataset classes for the GENESIS project."""

import os
import cv2
import glob
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

class MovieDataset(Dataset):
    """Dataset class for movie frames."""
    
    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[transforms.Compose] = None
    ):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        
        # Setup transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
        ])
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(list(self.root_dir.glob(f"**/{ext}")))
        
        # Filter out unwanted files
        self.image_files = [f for f in self.image_files if not str(f).endswith(('.0ipUU9', '.DS_Store'))]
        
        if not self.image_files:
            raise RuntimeError(f"No images found in {root_dir}")
            
        print(f"Found {len(self.image_files)} images in {root_dir}")
        
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = str(self.image_files[idx])
        
        try:
            # Load and convert image
            image = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
                
            return {'image': image}
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a random noise image as fallback
            return {'image': torch.randn(3, *self.image_size)}

class CelebAHQDataset(Dataset):
    """Dataset class for CelebA-HQ faces with CLIP text conditioning."""
    
    def __init__(
        self,
        root_dir: str,
        clip_processor=None,
        clip_model=None,
        max_text_length: int = 77,
        image_size: Tuple[int, int] = (512, 512),
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize the CelebA-HQ dataset.
        
        Args:
            root_dir (str): Path to the folder containing images
            clip_processor: CLIP processor for text encoding (optional)
            clip_model: CLIP model for text feature extraction (optional)
            max_text_length (int): Maximum length for text tokens (default: 77 for CLIP)
            image_size (Tuple[int, int]): Target image size for training (default: 512x512)
        """
        self.root_dir = Path(root_dir)
        self.clip_processor = clip_processor
        self.clip_model = clip_model
        self.max_text_length = max_text_length
        self.image_size = image_size
        
        # Support both jpg and png files in nested folders
        jpg_files = glob.glob(os.path.join(str(root_dir), "**/*.jpg"), recursive=True)
        png_files = glob.glob(os.path.join(str(root_dir), "**/*.png"), recursive=True)
        self.image_files = sorted(jpg_files + png_files)
        
        # Count images in each subfolder
        folder_counts = {}
        for img_path in self.image_files:
            folder = os.path.basename(os.path.dirname(img_path))
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
        
        if not self.image_files:
            raise RuntimeError(f"No images found in {root_dir}")
        else:
            print(f"Found {len(self.image_files)} total images in {root_dir}:")
            for folder, count in sorted(folder_counts.items()):
                print(f"  - Folder '{folder}': {count} images")
        
        # Setup transforms
        self.transform = transform or transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
        ])
    
    def extract_text_from_filename(self, filename: str) -> str:
        """Extract text description from filename.
        Example: 'young_male_with_glasses_001.jpg' -> 'Young male with glasses'
        """
        # Remove extension and number suffix
        text = os.path.splitext(filename)[0]
        text = '_'.join(text.split('_')[:-1]) if text.split('_')[-1].isdigit() else text
        
        # Convert underscores to spaces and capitalize first letter
        text = ' '.join(text.split('_')).capitalize()
        
        return text
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_files[idx]
        
        try:
            # Load and process image
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Extract text description and compute CLIP embeddings if available
            text = self.extract_text_from_filename(os.path.basename(img_path))
            text_embedding = None
            
            if self.clip_processor and self.clip_model:
                # Process text through CLIP
                inputs = self.clip_processor(
                    text=text,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_text_length,
                    truncation=True
                )
                
                # Get text features from CLIP model
                with torch.no_grad():
                    text_embedding = self.clip_model.get_text_features(**inputs)
                    text_embedding = text_embedding.squeeze(0)
            
            return {
                'image': image,
                'text': text,
                'text_embedding': text_embedding
            }
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a random noise image as fallback
            return {
                'image': torch.randn(3, 512, 512),
                'text': '',
                'text_embedding': torch.randn(512) if self.clip_model else None
            }
