import os
from frame_generator_sdxl import SDXLFrameGenerator

def test_frame_generation():
    # Create output directory
    os.makedirs("test_outputs", exist_ok=True)
    
    # Get absolute path to Juggernaut XL model
    model_path = os.path.abspath("models/juggernautXL_juggXIByRundiffusion.safetensors")
    print(f"Loading Juggernaut XL model from: {model_path}")
    
    print("\nInitializing SDXLFrameGenerator...")
    generator = SDXLFrameGenerator(model_path=model_path)
    
    # Test prompts
    test_prompts = [
        "hyper-realistic portrait of a female architect with 35 years old, mixed ethnicity, high cheekbones as distinctive feature, confident expression, professional attire, detailed facial features including hazel eyes and shoulder-length auburn hair, professional rembrandt lighting setup, shot with 85mm portrait lens, shallow depth of field, crisp focus on eyes, neutral studio background, photographic render, 8k resolution, highly detailed skin texture --ar 4:5 --v 6.0 --s 750 --q 2",
        # "a cinematic portrait of a 3 students having conversation with physics professor, soft rim lighting, 35mm film"
    ]
    
    for idx, prompt in enumerate(test_prompts):
        print(f"\nGenerating test image {idx + 1}...")
        print(f"Prompt: {prompt}")
        
        try:
            image, path = generator.generate_frame(
                prompt=prompt,
                width=1024,
                height=1024,
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=42 + idx,
                output_dir="test_outputs",
                filename=f"test_image_{idx + 1}.png"
            )
            print(f"Successfully generated image: {path}")
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            raise

if __name__ == "__main__":
    test_frame_generation()
