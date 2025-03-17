import os
from animatediff_generator import AnimateDiffGenerator

def test_animation_pipeline():
    # Initialize the generator with JuggernautXL model
    model_path = "models/juggernautXL_juggXIByRundiffusion.safetensors"
    generator = AnimateDiffGenerator(model_path=model_path, device="mps")  # Using MPS for M4 Max GPU
    
    # Test parameters
    prompt = "cinematic shot of a butterfly flying in a garden, masterpiece quality, detailed wings, natural movement"
    negative_prompt = "bad quality, blurry, static, distorted wings"
    
    # Generate video
    output_path = generator.generate_video_from_text(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=24,  # More frames for smoother motion
        num_inference_steps=50,  # More steps for quality
        guidance_scale=12.0,  # Higher guidance for stronger motion
        width=1024,
        height=1024,
        seed=42,  # Fixed seed for reproducibility
        output_dir="outputs",
        filename="butterfly_animation",
        fps=12  # Higher FPS for smoother playback
    )
    
    print(f"Animation generated successfully! Output saved to: {output_path}")

if __name__ == "__main__":
    test_animation_pipeline()
