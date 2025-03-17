from huggingface_hub import snapshot_download
import os

def download_base_model():
    print("Downloading base SDXL model...")
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model
    snapshot_download(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        local_dir="models/sdxl-base-1.0",
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md", "*.txt", "*.yaml"]
    )
    
    print("Download complete!")

if __name__ == "__main__":
    download_base_model()
