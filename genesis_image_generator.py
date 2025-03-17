import torch
from diffusers import (
    StableDiffusionXLPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    AutoencoderKL
)
from PIL import Image
import numpy as np
from pathlib import Path
import os
from typing import Optional, List, Union, Dict
from controlnet_pipeline import ControlNetPipeline

class GenesisImageGenerator:
    """
    Advanced image generation system for GENESIS AI using Juggernaut XL and ControlNet
    Focuses on Phase 1: AI Character & Background Generation
    """
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.models_dir = Path(__file__).parent / "models"
        self.base_model_path = str(self.models_dir / "juggernautXL_juggXIByRundiffusion.safetensors")
        self.controlnet = ControlNetPipeline(self.base_model_path)
        
        # Initialize base Juggernaut XL pipeline
        self.base_pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.float16
        ).to(device)
        
        # Enable optimizations
        self.base_pipeline.enable_attention_slicing()
        self.base_pipeline.enable_vae_slicing()

    def generate_character(
        self,
        character_prompt: str,
        pose_image: Optional[Image.Image] = None,
        control_type: str = "pose",
        negative_prompt: str = "low quality, bad anatomy, worst quality, low resolution",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8
    ) -> Image.Image:
        """
        Generate a high-quality character image with optional pose control
        """
        if pose_image is not None:
            return self.controlnet.generate(
                prompt=character_prompt,
                control_image=pose_image,
                control_type=control_type,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength
            )
        
        return self.base_pipeline(
            prompt=character_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

    def generate_scene(
        self,
        scene_prompt: str,
        depth_map: Optional[Image.Image] = None,
        composition_sketch: Optional[Image.Image] = None,
        control_type: str = "depth",
        negative_prompt: str = "low quality, blurry, worst quality",
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Generate a high-quality scene with optional depth or composition control
        """
        control_image = depth_map if control_type == "depth" else composition_sketch
        if control_image is not None:
            return self.controlnet.generate(
                prompt=scene_prompt,
                control_image=control_image,
                control_type=control_type,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            )
        
        return self.base_pipeline(
            prompt=scene_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

    def generate_cinematic_shot(
        self,
        prompt: str,
        shot_type: str,
        control_image: Optional[Image.Image] = None,
        control_type: Optional[str] = None,
    ) -> Image.Image:
        """
        Generate a cinematic shot with specific framing and composition
        """
        # Enhance prompt with cinematic elements
        cinematic_prompt = f"{shot_type}, {prompt}, cinematic lighting, dramatic composition, high resolution, professional photography, 8k"
        
        if control_image and control_type:
            return self.controlnet.generate(
                prompt=cinematic_prompt,
                control_image=control_image,
                control_type=control_type,
                negative_prompt="amateur, low quality, blurry, oversaturated",
                num_inference_steps=50,
                guidance_scale=8.0
            )
            
        return self.base_pipeline(
            prompt=cinematic_prompt,
            negative_prompt="amateur, low quality, blurry, oversaturated",
            num_inference_steps=50,
            guidance_scale=8.0
        ).images[0]

    def refine_image(
        self,
        image: Image.Image,
        prompt: str,
        mask_image: Optional[Image.Image] = None,
        strength: float = 0.7
    ) -> Image.Image:
        """
        Refine specific areas of an image using inpainting
        """
        if mask_image is not None:
            return self.controlnet.generate(
                prompt=prompt,
                control_image=image,
                control_type="inpaint",
                mask_image=mask_image,
                strength=strength
            )
        return image

    def save_image(self, image: Image.Image, filename: str, output_dir: str = "outputs"):
        """Save generated image with metadata"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        image.save(filepath, "PNG")
        return filepath
