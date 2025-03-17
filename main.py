import os
import argparse
import torch.optim as optim
from video_frame_generator import train_gan, Generator, Discriminator

def main():
    parser = argparse.ArgumentParser(description='Train GENESIS GAN model')
    parser.add_argument('--video_dataset', type=str, default='training_videos', help='Path to video dataset directory')
    parser.add_argument('--celeba_dataset', type=str, default='datasets/celeba_hq', help='Path to CelebA-HQ dataset directory')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda', 'mps'], default='cuda', help='Device to use for training')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('samples', exist_ok=True)
    
    print("Starting GENESIS GAN training...")
    
    # Initialize models
    latent_dim = 100
    generator = Generator(latent_dim=latent_dim, channels=3).to(args.device)
    discriminator = Discriminator(channels=3).to(args.device)
    
    # Initialize optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    train_gan(
        epochs=args.epochs,
        resume=args.resume,
        batch_size=args.batch_size,
        device=args.device,
        video_dataset=args.video_dataset,
        celeba_dataset=args.celeba_dataset,
        generator=generator,
        discriminator=discriminator,
        optimizer_G=optimizer_G,
        optimizer_D=optimizer_D
    )
    print("Training complete!")

if __name__ == '__main__':
    main()
