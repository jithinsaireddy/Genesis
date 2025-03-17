"""Model architectures for the GENESIS project."""

import torch
import torch.nn as nn
from typing import Tuple

class Generator(nn.Module):
    """Generator network."""
    
    def __init__(
        self,
        latent_dim: int = 100,
        channels: int = 3,
        image_size: int = 256
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.channels = channels
        self.image_size = image_size
        
        # Calculate initial size
        self.init_size = image_size // 32
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size ** 2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(32, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """Discriminator network."""
    
    def __init__(
        self,
        channels: int = 3,
        image_size: int = 256
    ):
        super().__init__()
        
        def discriminator_block(
            in_filters: int,
            out_filters: int,
            bn: bool = True
        ) -> nn.Sequential:
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return nn.Sequential(*block)
        
        self.model = nn.Sequential(
            discriminator_block(channels, 16, bn=False),
            discriminator_block(16, 32),
            discriminator_block(32, 64),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
        )
        
        # Calculate size of output features
        ds_size = image_size // 2**5
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity
