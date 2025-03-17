import os
import sys
import time
import logging
import numpy as np
from PIL import Image
from typing import Optional
from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler, StableDiffusionXLPipeline
from diffusers.models import MotionAdapter, ModelMixin, UNet2DConditionModel
from diffusers.utils import export_to_video
from transformers import CLIPTextModel, CLIPTokenizer, CLIPFeatureExtractor
import cv2
import numpy as np
import torch
from frame_generator_sdxl import SDXLFrameGenerator
from PIL import Image
from safetensors.torch import load_file as load_safetensors

# Add AnimateDiff to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "AnimateDiff"))
from animatediff.utils.util import load_file

def preprocess_image(image, device='cpu'):
    """Preprocess an image for the AnimateDiff pipeline.
    
    Args:
        image: PIL Image or file path
        device: Target device for the tensor
        
    Returns:
        torch.Tensor: Preprocessed image tensor in the format expected by AnimateDiff
    """
    if isinstance(image, str):
        image = Image.open(image)
    
    w, h = image.size
    image = image.convert("RGB")
    image = np.array(image)
    image = image[None].transpose(0, 3, 1, 2)  # NHWC -> NCHW
    image = torch.from_numpy(image).to(device=device, dtype=torch.float32) / 127.5 - 1.0  # Normalize to [-1, 1]
    
    return image

def export_to_video(frames, output_path, fps=8):
    """Export frames to MP4 video with specified fps.
    
    Args:
        frames: List or numpy array of frames
        output_path: Path to save the output video
        fps: Frames per second (default: 8)
    """
    print(f"DEBUG: Starting video export to {output_path} with {fps} fps")
    print(f"DEBUG: Frames shape before processing: {frames.shape if hasattr(frames, 'shape') else 'frames is list'}")
    
    if isinstance(frames, list):
        print(f"DEBUG: Converting frames list to array, list length: {len(frames)}")
        frames = np.array(frames)
        print(f"DEBUG: Frames shape after conversion: {frames.shape}")
    
    if frames.ndim == 4:
        print("DEBUG: Found 4D frames array")
        pass
    elif frames.ndim == 5:
        print("DEBUG: Found 5D frames array, taking first batch")
        frames = frames[0]
    
    # Ensure frames are uint8
    if frames.dtype != np.uint8:
        print(f"DEBUG: Converting frames from {frames.dtype} to uint8")
        frames = (frames * 255).round().astype(np.uint8)
    
    print(f"DEBUG: Final frames shape: {frames.shape}, dtype: {frames.dtype}")
    
    # Convert to absolute paths
    output_path = os.path.abspath(output_path)
    print(f"DEBUG: Using absolute path: {output_path}")
    
    # Create output directory
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        print(f"DEBUG: Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Create temp directory next to output directory
    temp_dir = os.path.join(output_dir, 'temp')
    if not os.path.exists(temp_dir):
        print(f"DEBUG: Creating temp directory: {temp_dir}")
        os.makedirs(temp_dir, exist_ok=True)
    
    temp_path = os.path.join(temp_dir, os.path.basename(output_path))
    print(f"DEBUG: Using temporary path: {temp_path}")
    
    try:
        # Get video dimensions
        height, width = frames[0].shape[:2]
        print(f"DEBUG: Video dimensions: {width}x{height}")
        
        # Try different codecs in order of preference
        codecs = [
            ('avc1', '.mp4'),
            ('mp4v', '.mp4'),
            ('mjpg', '.avi'),
            ('MJPG', '.avi')
        ]
        
        success = False
        for codec, ext in codecs:
            try:
                temp_file = temp_path.rsplit('.', 1)[0] + ext
                print(f"DEBUG: Trying codec {codec} with file {temp_file}")
                
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
                
                if not out.isOpened():
                    print(f"DEBUG: Failed to open VideoWriter with codec {codec}")
                    continue
                
                print(f"DEBUG: Successfully opened VideoWriter with codec {codec}")
                
                # Write frames
                for i, frame in enumerate(frames):
                    # OpenCV expects BGR format
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)
                
                # Release the video writer
                out.release()
                
                # Force flush the file
                with open(temp_file, 'rb') as f:
                    os.fsync(f.fileno())
                
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    print(f"DEBUG: Successfully wrote video with codec {codec}")
                    success = True
                    temp_path = temp_file  # Update temp_path to the successful file
                    break
                else:
                    print(f"DEBUG: Failed to write video with codec {codec}")
            except Exception as e:
                print(f"DEBUG: Error with codec {codec}: {str(e)}")
                continue
        
        if not success:
            raise RuntimeError("Failed to write video with any codec")
        
        print(f"DEBUG: Moving temporary file to final location")
        # Copy the file to final location with locking
        with open(temp_path, 'rb') as tmp:
            with open(output_path, 'wb') as final:
                # Get an exclusive lock
                import fcntl
                fcntl.flock(final.fileno(), fcntl.LOCK_EX)
                try:
                    # Copy the contents
                    final.write(tmp.read())
                    final.flush()
                    os.fsync(final.fileno())
                finally:
                    # Release the lock
                    fcntl.flock(final.fileno(), fcntl.LOCK_UN)
        
        # Verify final file
        if os.path.exists(output_path):
            size = os.path.getsize(output_path)
            print(f"DEBUG: Final file exists, size: {size} bytes")
            if size == 0:
                raise RuntimeError("Final file is empty")
        else:
            raise RuntimeError("Final file does not exist")
            
    except Exception as e:
        print(f"DEBUG: Error saving video: {str(e)}")
        raise
    finally:
        # Cleanup temp files
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
            print("DEBUG: Cleaned up temporary files")
        except Exception as e:
            print(f"DEBUG: Error during cleanup: {str(e)}")
    
    return output_path

from diffusers import AnimateDiffSDXLPipeline, DDIMScheduler, StableDiffusionXLPipeline
from diffusers.models import MotionAdapter, UNet2DConditionModel
import torch
import os
import logging
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
from pathlib import Path

class AnimateDiffGenerator:
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.logger = logging.getLogger(__name__)
        
        # Initialize motion generation parameters
        self.default_guidance_scale = 9.0  # Higher guidance for more pronounced motion
        
        # Initialize the SDXL frame generator for initial frame generation
        self.sdxl_generator = SDXLFrameGenerator(model_path=model_path, device=device)
        
        # Initialize base SDXL pipeline with JuggernautXL model
        base_pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float32,
            use_safetensors=True,
            variant="fp16" if device == "mps" else None  # Use fp16 for M4 Max GPU
        )
        
        # Load official AnimateDiff SDXL motion adapter with improved settings
        motion_adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-sdxl-beta",
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        # Initialize AnimateDiffSDXL pipeline with motion adapter
        self.pipeline = AnimateDiffSDXLPipeline(
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            text_encoder_2=base_pipeline.text_encoder_2,
            tokenizer=base_pipeline.tokenizer,
            tokenizer_2=base_pipeline.tokenizer_2,
            unet=base_pipeline.unet,
            scheduler=base_pipeline.scheduler,
            motion_adapter=motion_adapter
        ).to(device)
        
        # Enable memory optimizations
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        
        # Use EulerAncestralDiscreteScheduler for better motion consistency
        from diffusers import EulerAncestralDiscreteScheduler
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config,
            timestep_spacing="linspace",
            steps_offset=1,
            beta_schedule="scaled_linear"
        )
        
        # Enable memory optimizations for video generation
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()
        
        # Configure the scheduler for optimal motion generation
        self.pipeline.scheduler = DDIMScheduler.from_config(
            self.pipeline.scheduler.config,
            beta_schedule="scaled_linear",  # Better for temporal consistency
            timestep_spacing="linspace",  # Uniform temporal sampling
            steps_offset=1,
            clip_sample=False,
            prediction_type="epsilon"  # Standard noise prediction
        )
        
        # Configure motion-specific settings
        self.pipeline.scheduler.config.steps_offset = 1
        self.pipeline.scheduler.config.clip_sample = False

    def generate_video_from_image(
        self, 
        init_image, 
        prompt="", 
        negative_prompt="", 
        num_frames=16, 
        num_inference_steps=20, 
        guidance_scale=7.5, 
        motion_scale=1.0, 
        seed=None,
        output_path="output.mp4"
    ):
        generator = torch.Generator(self.device).manual_seed(seed) if seed else None

        frames = self.pipeline(
            image=init_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            motion_scale=motion_scale,
            generator=generator
        ).frames[0]

        export_to_video(frames, output_path=output_path, fps=8)
        return output_path

    def generate_video_from_text(
            self,
            prompt,
            negative_prompt="",
            num_frames=24,  # Increased for smoother motion
            num_inference_steps=50,  # More steps for quality
            guidance_scale=7.5,
            width=1024,
            height=1024,
            seed=None,
            output_dir="outputs",
            filename="generated_video",
            fps=12  # Higher FPS for smoother playback
        ):
        """Generate a video from a text prompt using JuggernautXL for initial frame and AnimateDiff for animation."""
        self.logger.info(f"Generating video for prompt: {prompt}")

        if not self.sdxl_generator:
            raise ValueError("SDXL generator not initialized. Cannot generate initial frame.")

        try:
            # Create output subdirectories
            frames_dir = os.path.join(output_dir, "frames")
            videos_dir = os.path.join(output_dir, "videos")
            os.makedirs(frames_dir, exist_ok=True)
            os.makedirs(videos_dir, exist_ok=True)

            # First generate the initial frame using JuggernautXL
            self.logger.info("Generating initial frame with JuggernautXL...")
            init_frame, init_frame_path = self.sdxl_generator.generate_frame(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                output_dir=frames_dir,
                filename=f"{filename}_init_frame.png"
            )

            self.logger.info(f"Initial frame saved at: {init_frame_path}")

            # Preprocess and encode the initial frame
            self.logger.info("Processing initial frame for AnimateDiff pipeline...")
            
            # Move pipeline components to MPS for GPU acceleration
            self.logger.info("Moving components to MPS for GPU acceleration...")
            
            # Move core components to MPS
            self.pipeline.vae = self.pipeline.vae.to(self.device)
            self.pipeline.text_encoder = self.pipeline.text_encoder.to(self.device)
            self.pipeline.text_encoder_2 = self.pipeline.text_encoder_2.to(self.device)
            self.pipeline.unet = self.pipeline.unet.to(self.device)
            
            # Preprocess frame and move to MPS
            preprocessed_frame = preprocess_image(init_frame)
            preprocessed_frame = preprocessed_frame.to(device=self.device, dtype=torch.float32)
            self.logger.info(f"Frame moved to {preprocessed_frame.device}")
            
            # Add batch dimension if needed
            if preprocessed_frame.ndim == 3:
                preprocessed_frame = preprocessed_frame.unsqueeze(0)
            
            # Encode using VAE on MPS
            with torch.no_grad():
                # Encode the frame
                init_latents = self.pipeline.vae.encode(preprocessed_frame).latent_dist.sample()
                init_latents = init_latents * self.pipeline.vae.config.scaling_factor
                
                # Add temporal dimension for AnimateDiff
                init_latents = init_latents.unsqueeze(2)  # [batch, channel, frames, height, width]
                init_latents = init_latents.repeat(1, 1, num_frames, 1, 1)
                init_latents = init_latents.to(self.device)
                
            self.logger.info(f"Latents prepared on {init_latents.device} with shape {init_latents.shape}")
            
            self.logger.info(f"Final latents shape: {init_latents.shape}, device: {init_latents.device}")
            
            # Enhanced motion-specific prompt engineering
            motion_prompt = (
                f"{prompt}, masterpiece quality, with smooth fluid motion, "
                "dynamic camera movement, cinematic scene with natural movement, "
                "waves gently rolling, clouds drifting in the wind, temporal coherence, "
                "fluid animation, 60fps smooth motion"
            )
            motion_negative = (
                f"{negative_prompt}, static, frozen, still image, low quality, "
                "poor animation, choppy movement, inconsistent lighting, stuttering motion"
            )
            
            self.logger.info("Generating video using AnimateDiff pipeline with enhanced motion...")
            
            # Configure generation parameters for optimal motion
            output = self.pipeline(
                prompt=motion_prompt,
                negative_prompt=motion_negative,
                num_inference_steps=50,
                guidance_scale=12.0,  # Higher guidance for stronger motion
                width=width,
                height=height,
                num_frames=num_frames,
                generator=torch.Generator(self.device).manual_seed(seed) if seed else None,
                latents=init_latents  # Use encoded JuggernautXL frame
            )

            # Process and save the generated video
            frames = output.frames[0]
            
            # Convert frames to numpy array if it's a list
            if isinstance(frames, list):
                frames = np.array(frames)
            elif isinstance(frames, torch.Tensor):
                frames = frames.cpu().numpy()
            
            # Convert from [-1,1] to [0,255] range if needed
            if frames.dtype != np.uint8:
                if frames.min() < 0 or frames.max() > 1:
                    frames = ((frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                elif frames.max() <= 1.0:
                    frames = (frames * 255).clip(0, 255).astype(np.uint8)
            
            self.logger.info(f"Frame stats - Min: {frames.min()}, Max: {frames.max()}, Shape: {frames.shape}")
            
            # Save a sample frame for verification at original resolution
            sample_frame_path = os.path.join(frames_dir, f"{filename}_sample_frame.png")
            Image.fromarray(frames[0]).save(sample_frame_path)
            self.logger.info(f"Sample frame saved at: {sample_frame_path}")
            
            # Export to video with higher FPS for smoother motion
            output_path = os.path.join(videos_dir, f"{filename}.mp4")
            export_to_video(frames, output_path, fps=fps)
            self.logger.info(f"Video exported to {output_path} at {fps} FPS")

            # Verify the video was saved correctly
            if os.path.exists(output_path):
                size = os.path.getsize(output_path)
                self.logger.info(f"Final verification - file exists at {output_path}, size: {size} bytes")
                if size == 0:
                    raise RuntimeError(f"Generated video file is empty: {output_path}")
            else:
                raise RuntimeError(f"Failed to save video to: {output_path}")

            return output_path

        except Exception as e:
            self.logger.error(f"Error in video generation pipeline: {str(e)}")
            raise

def test_video_generation():
    """Test the video generation pipeline."""
    try:
        # Initialize the generator with JuggernautXL model
        generator = AnimateDiffGenerator(
            model_path="/Users/jithinpothireddy/CascadeProjects/GENESIS/models/juggernautXL_juggXIByRundiffusion.safetensors",
            device="mps"  # Using MPS for M4 Max GPU
        )
        # Test parameters optimized for high-quality output
        prompt = "A beautiful sunset over a calm ocean, cinematic lighting, masterpiece, best quality, highly detailed"
        negative_prompt = "low quality, bad quality, blurry, distorted, ugly, duplicate, morbid, mutilated, deformed"
        
        # Generate video with optimized settings for high-quality output
        output_path = generator.generate_video_from_text(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=16,  # Standard length for short sequences
            num_inference_steps=30,  # More steps for quality
            guidance_scale=7.5,  # Balanced guidance
            width=256,  # Reduced for testing
            height=256,  # Reduced for testing
            output_dir="outputs",
            filename="test_video"
        )
        
        print(f"Video generated successfully at: {output_path}")
        
    except Exception as e:
        print(f"Error in test: {str(e)}")
        raise

if __name__ == "__main__":
    test_video_generation()
