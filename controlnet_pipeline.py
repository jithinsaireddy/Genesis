import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
from pathlib import Path
import os

class ControlNetPipeline:
    def __init__(self, base_model_path=None):
        self.models_dir = Path(__file__).parent / "models"
        self.base_model_path = base_model_path or str(self.models_dir / "juggernautXL_juggXIByRundiffusion.safetensors")
        self.controlnet_models = {
            "depth": str(self.models_dir / "diffusion_depthcontrol_pytorch_model.fp16.safetensors"),
            "pose": str(self.models_dir / "diffusion_posecontrol_pytorch_model.fp16.safetensors"),
            "scribble": str(self.models_dir / "diffusion_scribble_pytorch_model.safetensors"),
            "inpaint": str(self.models_dir / "diffusion_inpaint_pytorch_model.fp16.safetensors")
        }
        self.active_pipeline = None
        self.active_control_type = None

    def load_pipeline(self, control_type):
        """
        Load the pipeline with specified control type
        Args:
            control_type (str): One of 'depth', 'pose', 'scribble', or 'inpaint'
        """
        if control_type not in self.controlnet_models:
            raise ValueError(f"Control type must be one of {list(self.controlnet_models.keys())}")
        
        if self.active_control_type == control_type and self.active_pipeline is not None:
            return self.active_pipeline

        # Load the ControlNet model
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_models[control_type],
            torch_dtype=torch.float16
        )

        # Load the base model with ControlNet
        self.active_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.base_model_path,
            controlnet=controlnet,
            torch_dtype=torch.float16
        )

        if torch.cuda.is_available():
            self.active_pipeline.to("cuda")
        
        self.active_control_type = control_type
        return self.active_pipeline

    def generate(self, prompt, control_image, control_type, negative_prompt=None, **kwargs):
        """
        Generate an image using ControlNet
        Args:
            prompt (str): The text prompt for generation
            control_image: The control image (preprocessed according to control type)
            control_type (str): Type of control to use
            negative_prompt (str, optional): Negative prompt
            **kwargs: Additional arguments for the pipeline
        """
        pipe = self.load_pipeline(control_type)
        
        return pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
            **kwargs
        ).images[0]
