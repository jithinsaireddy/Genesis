import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler
from diffusers.utils import export_to_video
from transformers import CLIPTextModel, CLIPTokenizer
import imageio
from PIL import Image
import numpy as np
import os
import sys

# Add AnimateDiff to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "AnimateDiff"))
from animatediff.utils.util import load_weights

class MPSAnimateDiffPipeline(AnimateDiffPipeline):
    """Custom pipeline that handles MPS device placement correctly."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cpu_device = torch.device("cpu")
        self.mps_device = torch.device("mps") if torch.backends.mps.is_available() else self.cpu_device
    
    @torch.no_grad()
    def __call__(self, *args, **kwargs):
        # Move all input tensors to CPU before processing
        if "prompt" in kwargs:
            if isinstance(kwargs["prompt"], torch.Tensor):
                kwargs["prompt"] = kwargs["prompt"].to(self.cpu_device)
        if "negative_prompt" in kwargs:
            if isinstance(kwargs["negative_prompt"], torch.Tensor):
                kwargs["negative_prompt"] = kwargs["negative_prompt"].to(self.cpu_device)
                
        # Process the text on CPU
        text_encoder_device = self.cpu_device
        self.text_encoder = self.text_encoder.to(text_encoder_device)
        
        # Get the text embeddings
        text_inputs = self.tokenizer(
            kwargs["prompt"],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(text_encoder_device)
        
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        prompt_embeds = prompt_embeds.to(self.mps_device)
        
        # Handle negative prompt
        if kwargs.get("negative_prompt") is not None:
            neg_text_inputs = self.tokenizer(
                kwargs["negative_prompt"],
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_text_inputs.input_ids.to(text_encoder_device)
            negative_prompt_embeds = self.text_encoder(neg_input_ids)[0]
            negative_prompt_embeds = negative_prompt_embeds.to(self.mps_device)
        else:
            negative_prompt_embeds = None
            
        # Update kwargs with pre-computed embeddings
        kwargs["prompt_embeds"] = prompt_embeds
        kwargs["negative_prompt_embeds"] = negative_prompt_embeds
        kwargs.pop("prompt", None)
        kwargs.pop("negative_prompt", None)
        
        # Ensure UNet and VAE are on MPS
        self.unet = self.unet.to(self.mps_device)
        self.vae = self.vae.to(self.mps_device)
        
        return super().__call__(*args, **kwargs)

def convert_image_to_video(image_path, output_path="output_animation.mp4", num_frames=16):
    """Convert a single image into a video animation using AnimateDiff."""
    
    # Load and resize the image
    image = Image.open(image_path)
    image = image.resize((1280, 720), Image.Resampling.LANCZOS)
    
    # Set up devices
    cpu_device = torch.device("cpu")
    mps_device = torch.device("mps") if torch.backends.mps.is_available() else cpu_device
    print(f"Using MPS: {mps_device.type == 'mps'}")
    
    # Set up scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear"
    )
    
    try:
        # Initialize pipeline
        pipe = MPSAnimateDiffPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            scheduler=scheduler,
            torch_dtype=torch.float32,
            use_safetensors=True,
        )
        
        # Download and load motion module
        motion_module_path = "/Users/jithinpothireddy/CascadeProjects/GENESIS/models/v3_sd15_mm.ckpt"
        if not os.path.exists(motion_module_path):
            print("Downloading motion module...")
            os.makedirs(os.path.dirname(motion_module_path), exist_ok=True)
            os.system(f"curl -L https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt -o {motion_module_path}")
        
        # Load motion module weights
        load_weights(pipe, motion_module_path=motion_module_path)
        
        # Enable optimizations
        pipe.enable_vae_slicing()
        
        # Set up prompts
        prompt = ["A cinematic scene with smooth camera movement, dynamic lighting changes, atmospheric effects like subtle smoke or particles, maintaining the original image's style and composition, high quality, dslr, soft lighting"]
        negative_prompt = ["low quality, blurry, distorted, pixelated, artificial, bad composition, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, text, close up, cropped, out of frame, worst quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, deformed"]
        
        # Generate frames
        with torch.inference_mode():
            video_frames = pipe(
                image=image,
                num_frames=num_frames,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                guidance_scale=8.0,
            ).frames
        
        # Save the video
        export_to_video(video_frames, output_path, fps=8)
        print(f"Generated animation saved as '{output_path}'")
        
        # Save individual frames
        frames_dir = os.path.splitext(output_path)[0] + "_frames"
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame in enumerate(video_frames):
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            Image.fromarray(frame).save(frame_path)
        print(f"Individual frames saved in '{frames_dir}'")
        
    except Exception as e:
        print(f"Error during video generation: {str(e)}")
        raise

if __name__ == "__main__":
    input_image = "/Users/jithinpothireddy/CascadeProjects/GENESIS/test_outputs/test_image_2.png"
    output_video = "test_image_2_animation.mp4"
    convert_image_to_video(input_image, output_video)
