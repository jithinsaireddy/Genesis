import os
import requests
import shutil
from pathlib import Path

# Set up authentication
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
if not HF_TOKEN:
    print("Warning: HUGGINGFACE_TOKEN environment variable not set. Some downloads may fail.")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}" if HF_TOKEN else "",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# Create directories
BASE_DIR = Path("ComfyUI/models/animatediff")
BASE_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url, filename):
    print(f"\nDownloading {filename}...")
    target_path = BASE_DIR / filename
    
    try:
        response = requests.get(url, headers=HEADERS, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded_size = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size:
                        percent = (downloaded_size / total_size) * 100
                        print(f"Progress: {percent:.1f}% ({downloaded_size}/{total_size} bytes)", end='\r')
        
        print(f"\nSuccessfully downloaded to {target_path}")
        return True
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        if target_path.exists():
            target_path.unlink()
        return False

# Official AnimateDiff v3 URLs from the model repository
MOTION_MODULE_URL = "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_mm.ckpt"
ADAPTER_URL = "https://huggingface.co/guoyww/animatediff/resolve/main/v3_sd15_adapter.ckpt"

print("Starting downloads for AnimateDiff v3...")
print("This will download the motion module and adapter required for high-quality animation generation.")

# Download Motion Module
success_mm = download_file(MOTION_MODULE_URL, "v3_sd15_mm.ckpt")
if success_mm:
    print("✓ Motion module downloaded successfully")
else:
    print("✗ Failed to download motion module")

# Download Adapter
success_adapter = download_file(ADAPTER_URL, "v3_sd15_adapter.ckpt")
if success_adapter:
    print("✓ Adapter downloaded successfully")
else:
    print("✗ Failed to download adapter")

print("\nVerifying downloads...")
for file in ["v3_sd15_mm.ckpt", "v3_sd15_adapter.ckpt"]:
    path = BASE_DIR / file
    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"✓ {file}: {size_mb:.1f} MB")
    else:
        print(f"✗ {file} not found")

if success_mm and success_adapter:
    print("\n✓ Successfully downloaded all required AnimateDiff v3 components!")
    print("These files will enable high-quality animation generation with temporal consistency.")
else:
    print("\n✗ Some downloads failed. Please check the error messages above.")
