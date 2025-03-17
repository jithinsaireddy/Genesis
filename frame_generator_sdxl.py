import torch
from diffusers import StableDiffusionXLPipeline
import os
from PIL import Image
import logging
import numpy as np

class SDXLFrameGenerator:
    def __init__(self, model_path, device=None):
        """Initialize the SDXL Frame Generator.
        
        Args:
            model_path (str): Path to Juggernaut XL model file
            device (str): Device to run the model on (defaults to 'mps' for Apple Silicon)
        """
        # Initialize logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # For M4 Max, we'll use MPS by default
        self.device = device or 'mps'
        if self.device == 'mps' and not torch.backends.mps.is_available():
            self.logger.warning("MPS not available, falling back to CPU")
            self.device = 'cpu'
            
        self.logger.info(f"Initializing SDXL Frame Generator on {self.device}")
        
        if not os.path.exists(model_path):
            raise ValueError(f"Model not found at {model_path}")
            
        self.logger.info(f"Loading Juggernaut XL model from: {model_path}")
        
        # Load the model using StableDiffusionXLPipeline with optimized settings
        self.pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float32,  # Start with float32
            use_safetensors=True,
            variant="fp16"
        )
        
        # Memory optimizations
        self.pipeline.enable_attention_slicing()
        self.pipeline.enable_vae_slicing()
        
        # Convert to float16 and move to device if using MPS
        if self.device == 'mps':
            self.pipeline.to(torch.device("cpu"))  # First move to CPU
            self.pipeline.to(torch_dtype=torch.float16)  # Convert to float16
            self.pipeline.to(torch.device(self.device))  # Then move to MPS
        else:
            self.pipeline.to(torch.device(self.device))
        
        # Set default Juggernaut XL parameters - optimized for v10
        self.default_positive_prefix = "(masterpiece:1.4), (photorealistic:1.4), (best quality:1.4), (ultra detailed:1.2), (sharp focus:1.2), cinematic lighting"
        self.default_negative_prompt = "(worst quality:2.0), (low quality:2.0), (normal quality:2.0), text, watermark, signature, blurry, art, drawing, painting, rendered, anime, manga"
        
        self.logger.info("SDXL Frame Generator initialized successfully")
        
    def generate_frame(self, prompt, width=1024, height=1024, num_inference_steps=30, guidance_scale=9.0, seed=None, output_dir="outputs", filename=None):
        """Generate a frame using the SDXL pipeline.
        
        Args:
            prompt (str): The prompt to generate the image from
            width (int): Width of the generated image
            height (int): Height of the generated image
            num_inference_steps (int): Number of denoising steps
            guidance_scale (float): How closely to follow the prompt
            seed (int): Random seed for reproducibility
            output_dir (str): Directory to save the generated image
            filename (str): Filename for the generated image
            
        Returns:
            tuple: (PIL.Image, str) The generated image and its save path
        """
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == 'mps':
                torch.mps.manual_seed(seed)
            
        # Prepare the prompt
        full_prompt = f"{self.default_positive_prefix}, {prompt}"
        
        # Generate the image
        self.logger.info(f"Generating image with prompt: {prompt}")
        
        # Generate with proper error handling
        try:
            # First attempt with normal settings
            with torch.inference_mode():
                output = self.pipeline(
                    prompt=full_prompt,
                    negative_prompt=self.default_negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                )
                image = output.images[0]
            
            # Verify image is not blank using a more robust check
            img_array = np.array(image)
            if img_array.std() < 1.0 or img_array.mean() < 1.0:
                self.logger.warning("Generated image appears to be blank, retrying with different settings...")
                # Retry with more aggressive settings
                with torch.inference_mode():
                    output = self.pipeline(
                        prompt=full_prompt,
                        negative_prompt=self.default_negative_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=40,  # More steps
                        guidance_scale=12.0,  # Much higher guidance scale
                    )
                    image = output.images[0]
                
                # If still blank, try one last time with maximum settings
                img_array = np.array(image)
                if img_array.std() < 1.0 or img_array.mean() < 1.0:
                    self.logger.warning("Still blank, trying one final time with maximum settings...")
                    with torch.inference_mode():
                        output = self.pipeline(
                            prompt=full_prompt,
                            negative_prompt=self.default_negative_prompt,
                            width=width,
                            height=height,
                            num_inference_steps=50,  # Maximum steps
                            guidance_scale=15.0,  # Maximum guidance
                        )
                        image = output.images[0]
        
        except Exception as e:
            self.logger.error(f"Error generating image: {str(e)}")
            raise
        
        # Save the image with absolute paths
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if filename is None:
            filename = f"frame_{seed if seed is not None else 'unseed'}.png"
        save_path = os.path.join(output_dir, filename)
        
        # Save with proper flushing
        print(f"DEBUG: Saving frame to {save_path}")
        image.save(save_path, format='PNG')
        
        # Force flush to disk
        with open(save_path, 'rb') as f:
            os.fsync(f.fileno())
        
        # Verify the file exists and has content
        if os.path.exists(save_path):
            size = os.path.getsize(save_path)
            print(f"DEBUG: Frame saved successfully, size: {size} bytes")
            if size == 0:
                raise RuntimeError(f"Generated frame file is empty: {save_path}")
        else:
            raise RuntimeError(f"Failed to save frame to: {save_path}")
        
        return image, save_path

    def generate_sequence(
        self,
        prompts,
        **kwargs
    ):
        """Generate a sequence of frames using SDXL.
        
        Args:
            prompts (list): List of prompts for each frame
            **kwargs: Additional arguments passed to generate_frame
            
        Returns:
            list: List of (image, path) tuples for each generated frame
        """
        return [self.generate_frame(prompt, **kwargs) for prompt in prompts]
