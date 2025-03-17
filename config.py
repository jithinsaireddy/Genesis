"""Configuration settings for the GENESIS project."""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "datasets"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
SAMPLES_DIR = PROJECT_ROOT / "samples"

# Dataset paths
CELEBA_DATASET = DATA_DIR / "celeba_hq"
VIDEO_DATASET = PROJECT_ROOT / "training_videos"

# Training configs
TRAIN_CONFIG = {
    'image_size': 256,  # Start with 256x256 for local training
    'latent_dim': 100,
    'batch_size': 16,   # Smaller batch size for local training
    'num_epochs': 100,
    'lr': 0.0002,
    'beta1': 0.5,
    'beta2': 0.999,
    'n_critic': 5,
    'clip_value': 0.01,
    'sample_interval': 100
}

# Create necessary directories
for directory in [DATA_DIR, CHECKPOINT_DIR, SAMPLES_DIR, CELEBA_DATASET, VIDEO_DATASET]:
    directory.mkdir(parents=True, exist_ok=True)

# Device configuration
def get_device():
    """Get the best available device for training."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# Model saving/loading paths
MODEL_PATHS = {
    'generator': CHECKPOINT_DIR / 'generator.pth',
    'discriminator': CHECKPOINT_DIR / 'discriminator.pth',
    'optimizer_g': CHECKPOINT_DIR / 'optimizer_g.pth',
    'optimizer_d': CHECKPOINT_DIR / 'optimizer_d.pth',
}
