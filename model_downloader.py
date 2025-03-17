import os
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib
from typing import Dict, Optional
from huggingface_hub import hf_hub_download, login

class ModelDownloader:
    """Handles downloading and management of AI models for GENESIS"""
    
    MODEL_CONFIGS = {
        "animatediff": {
            "repo_id": "guoyww/AnimateDiff",
            "filename": "mm_sd_v15_v2.ckpt",
            "local_filename": "animatediff_model.pth"
        },
        "zeroscope": {
            "repo_id": "cerspense/zeroscope_v2_576w",
            "filename": "v2_576w.ckpt",
            "local_filename": "zeroscope_v2_576w.pth"
        }
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def login_to_hf(token: str):
        """Login to Hugging Face Hub"""
        login(token=token)

    def download_model(self, model_name: str) -> Path:
        """Download a model from Hugging Face Hub"""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
            
        config = self.MODEL_CONFIGS[model_name]
        local_path = self.models_dir / config["local_filename"]
        
        print(f"\nDownloading {model_name} from {config['repo_id']}")
        try:
            # Download file from Hugging Face Hub
            downloaded_path = hf_hub_download(
                repo_id=config["repo_id"],
                filename=config["filename"],
                local_dir=self.models_dir,
                local_dir_use_symlinks=False
            )
            
            # Rename to our desired filename if needed
            if Path(downloaded_path).name != config["local_filename"]:
                os.rename(downloaded_path, local_path)
            
            print(f"Successfully downloaded {model_name} to {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            raise

    def download_all_models(self):
        """Download all required models"""
        for model_name in self.MODEL_CONFIGS:
            try:
                self.download_model(model_name)
                print(f"Successfully downloaded {model_name}")
            except Exception as e:
                print(f"Error downloading {model_name}: {str(e)}")

    def get_model_path(self, model_name: str) -> Path:
        """Get the path to a downloaded model"""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}")
        return self.models_dir / self.MODEL_CONFIGS[model_name]["local_filename"]

if __name__ == "__main__":
    print("To download these models, you need to:")
    print("1. Create an account on https://huggingface.co")
    print("2. Generate an access token at https://huggingface.co/settings/tokens")
    print("3. Set your token as an environment variable:")
    print("   export HUGGINGFACE_TOKEN=your_token_here")
    print("\nOr you can login directly using:")
    print("from model_downloader import ModelDownloader")
    print("ModelDownloader.login_to_hf('your_token_here')")
    
    token = os.environ.get("HUGGINGFACE_TOKEN")
    if token:
        ModelDownloader.login_to_hf(token)
        downloader = ModelDownloader()
        downloader.download_all_models()
    else:
        print("\nNo HUGGINGFACE_TOKEN found in environment variables.")
        print("Please set your token and try again.")
