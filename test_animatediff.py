from diffusers import AnimateDiffPipeline, DiffusionPipeline
import torch
from PIL import Image
import imageio

def test_animatediff():
    # Load the base model and AnimateDiff
    pipe = AnimateDiffPipeline.from_pretrained(
        "guoyww/animatediff-motion-adapter-v1-5-2",
        torch_dtype=torch.float16
    ).to("cuda")
    
    # Generate a video
    prompt = "A beautiful butterfly flying in a garden with flowers, highly detailed, photorealistic"
    negative_prompt = "low quality, blurry, distorted"
    
    # Generate frames
    frames = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=25,
        num_frames=16
    ).frames
    
    # Save as GIF
    imageio.mimsave("butterfly_animation.gif", frames, fps=8)
    
    # Save first frame as image
    first_frame = Image.fromarray(frames[0])
    first_frame.save("butterfly_frame.png")
    
    print("Generated animation saved as 'butterfly_animation.gif'")
    print("First frame saved as 'butterfly_frame.png'")

if __name__ == "__main__":
    test_animatediff()
