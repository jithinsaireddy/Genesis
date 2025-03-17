import os
from frame_generator_sdxl import SDXLFrameGenerator
import json
import requests
import time

def generate_keyframe():
    """Generate a high-quality keyframe using JuggernautXL"""
    model_path = os.path.abspath("models/juggernautXL_juggXIByRundiffusion.safetensors")
    generator = SDXLFrameGenerator(model_path=model_path)
    
    # Simple test prompt for animation
    prompt = "cinematic shot of a butterfly in a garden, detailed wings, photorealistic"
    
    # Generate at 256x256 for quick testing
    # Using default output path from frame generator
    image, path = generator.generate_frame(
        prompt=prompt,
        width=256,
        height=256,
        num_inference_steps=20,
        guidance_scale=7.0
    )
    return path

def animate_with_comfyui(keyframe_path):
    """Apply AnimateDiff to the keyframe using ComfyUI's API.
    This function handles the integration between JuggernautXL frames and AnimateDiff,
    ensuring proper frame format handling and style preservation.
    """
    workflow = {
        "1": {
            "inputs": {
                "image": keyframe_path
            },
            "class_type": "LoadImage"
        },
        "2": {
            "inputs": {
                "model_name": "v3_sd15_mm.ckpt",
                "beta_schedule": "linear"
            },
            "class_type": "StandardUniformContextOptions"
        },
        "3": {
            "inputs": {
                "text": "cinematic shot of a butterfly flying in a garden, smooth motion, detailed wings, natural movement, photorealistic"
            },
            "class_type": "CLIPTextEncode"
        },
        "4": {
            "inputs": {
                "text": "bad quality, blurry, static image, choppy motion, duplicate frames"
            },
            "class_type": "CLIPTextEncode"
        },
        "5": {
            "inputs": {
                "model_path": "animatediff_models/v3_sd15_mm.ckpt",
                "context_options": ["2", 0]
            },
            "class_type": "LoadAnimateDiffModelNode"
        },
        "6": {
            "inputs": {
                "model": ["5", 0],
                "image": ["1", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "context_options": ["2", 0],
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "motion_scale": 1.0,
                "frame_count": 16,
                "fps": 8
            },
            "class_type": "ApplyAnimateDiffModelNode"
        },
        "7": {
            "inputs": {
                "samples": ["6", 0]
            },
            "class_type": "VAEDecode"
        },
        "8": {
            "inputs": {
                "images": ["7", 0],
                "frame_rate": 8,
                "filename_prefix": "animation",
                "format": "GIF"
            },
            "class_type": "SaveAnimatedWEBP"
        }
    }
    
    # Queue the workflow
    response = requests.post("http://127.0.0.1:8188/prompt", json={
        "prompt": workflow,
        "client_id": "test_script"
    })
    
    if response.status_code == 200:
        print("Successfully queued animation workflow")
        prompt_id = response.json()['prompt_id']
        
        # Wait for completion
        while True:
            response = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
            if response.status_code == 200:
                data = response.json()[prompt_id]
                if "executing" in data and not data["executing"]:
                    print("Animation complete!")
                    print(f"Animation saved to: ComfyUI/output/{data.get('output_filenames', [''])[0]}")
                    break
            time.sleep(2)
    else:
        print(f"Failed to queue animation: {response.status_code}")

if __name__ == "__main__":
    print("Step 1: Generating keyframe with JuggernautXL...")
    keyframe = generate_keyframe()
    print(f"Keyframe generated at: {keyframe}")
    
    print("\nStep 2: Applying AnimateDiff motion...")
    animate_with_comfyui(keyframe)
