"""Utility functions for the GENESIS project."""

import os
import gc
import torch
import random
import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any

def seed_everything(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def clear_gpu_memory(aggressive: bool = False) -> None:
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        if aggressive:
            gc.collect()

def save_checkpoint(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: torch.optim.Optimizer,
    epoch: int,
    loss_g: float,
    loss_d: float,
    checkpoint_dir: Union[str, Path],
    filename: str = 'checkpoint.pth'
) -> None:
    """Save model checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'loss_g': loss_g,
        'loss_d': loss_d
    }
    
    torch.save(checkpoint, checkpoint_dir / filename)

def load_checkpoint(
    checkpoint_path: Union[str, Path],
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    optimizer_g: Optional[torch.optim.Optimizer] = None,
    optimizer_d: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cpu')
) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    if optimizer_g is not None:
        optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    if optimizer_d is not None:
        optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    
    return {
        'epoch': checkpoint['epoch'],
        'loss_g': checkpoint['loss_g'],
        'loss_d': checkpoint['loss_d']
    }

def initialize_weights(model: torch.nn.Module) -> None:
    """Initialize model weights."""
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
