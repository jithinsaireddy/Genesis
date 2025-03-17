"""Training script for the GENESIS project."""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

from config import TRAIN_CONFIG, get_device
from models import Generator, Discriminator
from datasets import MovieDataset, CelebAHQDataset
from utils import (
    seed_everything,
    clear_gpu_memory,
    save_checkpoint,
    load_checkpoint,
    AverageMeter
)

def initialize_clip(device):
    """Initialize CLIP model and processor."""
    try:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("Successfully initialized CLIP model and processor")
        return clip_model, processor
    except Exception as e:
        print(f"Error initializing CLIP: {str(e)}")
        return None, None

def train(args):
    """Main training function."""
    
    # Set random seed
    seed_everything(args.seed)
    
    # Setup device
    device = get_device() if args.device == 'auto' else torch.device(args.device)
    print(f"Using device: {device}")
    
    # Initialize CLIP
    clip_model, clip_processor = initialize_clip(device)
    
    # Setup datasets
    movie_dataset = MovieDataset(
        root_dir=args.video_dataset,
        image_size=(args.image_size, args.image_size)
    )
    
    celeba_dataset = CelebAHQDataset(
        root_dir=args.celeba_dataset,
        clip_processor=clip_processor,
        clip_model=clip_model,
        max_text_length=77,
        image_size=(args.image_size, args.image_size)
    )
    
    # Combine datasets
    train_dataset = ConcatDataset([movie_dataset, celeba_dataset])
    print(f"Total training images: {len(train_dataset)} (Movie: {len(movie_dataset)}, CelebA-HQ: {len(celeba_dataset)})")
    print(f"Training at resolution: {args.image_size}x{args.image_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize models
    generator = Generator(
        latent_dim=args.latent_dim,
        channels=3,
        image_size=args.image_size
    ).to(device)
    
    discriminator = Discriminator(
        channels=3,
        image_size=args.image_size
    ).to(device)
    
    # Initialize optimizers
    optimizer_g = optim.Adam(
        generator.parameters(),
        lr=args.g_lr,
        betas=(0.5, 0.999)
    )
    
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.d_lr,
        betas=(0.5, 0.999)
    )
    
    # Load checkpoint if fine-tuning
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = load_checkpoint(args.checkpoint, generator, discriminator)
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")
        
        if args.finetune:
            print("Fine-tuning mode: Adjusting learning rates")
            args.g_lr *= 0.1
            args.d_lr *= 0.1
    else:
        start_epoch = 0
    
    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        generator.train()
        discriminator.train()
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")
        
        # Metrics
        g_losses = AverageMeter()
        d_losses = AverageMeter()
        
        for batch_idx, batch in enumerate(pbar):
            real_images = batch['image'].to(device)
            batch_size = real_images.size(0)
            
            # Clear memory
            clear_gpu_memory()
            
            # Train discriminator
            optimizer_d.zero_grad()
            
            # Generate fake images
            z = torch.randn(batch_size, args.latent_dim).to(device)
            fake_images = generator(z)
            
            # Real images score
            real_validity = discriminator(real_images)
            # Fake images score
            fake_validity = discriminator(fake_images.detach())
            
            # Discriminator loss
            d_loss = (
                torch.mean(nn.ReLU()(1.0 - real_validity)) +
                torch.mean(nn.ReLU()(1.0 + fake_validity))
            )
            
            d_loss.backward()
            optimizer_d.step()
            
            # Train generator
            if batch_idx % 5 == 0:
                optimizer_g.zero_grad()
                
                # Generate fake images
                fake_validity = discriminator(fake_images)
                
                # Generator loss
                g_loss = -torch.mean(fake_validity)
                
                g_loss.backward()
                optimizer_g.step()
                
                g_losses.update(g_loss.item())
            
            d_losses.update(d_loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'g_loss': f"{g_losses.avg:.4f}",
                'd_loss': f"{d_losses.avg:.4f}"
            })
            
            # Save samples periodically
            if batch_idx % 100 == 0:
                save_path = Path("samples") / f"epoch_{epoch}_batch_{batch_idx}.png"
                save_image(fake_images, save_path, normalize=True)
        
        # Save checkpoint
        save_checkpoint(
            generator,
            discriminator,
            optimizer_g,
            optimizer_d,
            epoch,
            g_losses.avg,
            d_losses.avg,
            "checkpoints",
            'latest.pth'
        )

def main():
    parser = argparse.ArgumentParser(description='Train GENESIS model')
    
    # Dataset paths
    parser.add_argument('--video_dataset', type=str, required=True, help='Path to video frames dataset')
    parser.add_argument('--celeba_dataset', type=str, required=True, help='Path to CelebA-HQ dataset')
    
    # Training parameters
    parser.add_argument('--image_size', type=int, default=512, help='Image size for training (default: 512)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train (default: 100)')
    parser.add_argument('--g_lr', type=float, default=0.0002, help='Generator learning rate (default: 0.0002)')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='Discriminator learning rate (default: 0.0002)')
    parser.add_argument('--latent_dim', type=int, default=512, help='Latent dimension (default: 512)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
    
    # Device and checkpointing
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint for resuming/fine-tuning')
    parser.add_argument('--finetune', action='store_true', help='Enable fine-tuning mode with reduced learning rates')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Adjust batch size for higher resolutions
    if args.image_size >= 1024:
        args.batch_size = max(4, args.batch_size // 4)
        print(f"Adjusted batch size to {args.batch_size} for {args.image_size}x{args.image_size} training")
    
    train(args)

if __name__ == '__main__':
    main()
