import json
import requests
import websocket
import time

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": "genesis_animation"}
    data = json.dumps(p)
    resp = requests.post('http://127.0.0.1:8188/prompt', data=data)
    if resp.status_code != 200:
        raise Exception(f"Error queuing prompt: {resp.text}")
    return resp.json()

def generate_animation(prompt, negative_prompt="bad quality, blurry, static image, choppy motion"):
    # Define our workflow nodes
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "juggernautXL_juggXIByRundiffusion.safetensors"
            }
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["1", 1]
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["1", 1]
            }
        },
        "4": {
            "class_type": "ADE_LoadAnimateDiffModel",
            "inputs": {
                "model": ["1", 0],
                "motion_model": "mm_sdxl_v10_beta.ckpt"
            }
        },
        "5": {
            "class_type": "ADE_AnimateDiffSamplingSettings",
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "motion_scale": 1.0,
                "frame_count": 16
            }
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "batch_size": 1,
                "height": 512,
                "width": 512
            }
        },
        "7": {
            "class_type": "ADE_UseEvolvedSampling",
            "inputs": {
                "model": ["4", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["6", 0],
                "sampling_settings": ["5", 0]
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["1", 2]
            }
        },
        "9": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["8", 0],
                "fps": 12,
                "filename_prefix": "butterfly_animation"
            }
        }
    }
    
    print("Queuing animation generation...")
    result = queue_prompt(workflow)
    
    # Connect to websocket to monitor progress
    ws = websocket.WebSocket()
    ws.connect("ws://127.0.0.1:8188/ws")
    
    try:
        while True:
            out = ws.recv()
            if not out:
                continue
            
            data = json.loads(out)
            if data['type'] == 'executing':
                node_id = data.get('data', {}).get('node', None)
                if node_id:
                    print(f"Processing node {node_id}...")
            elif data['type'] == 'executed':
                print("Animation generation complete!")
                break
            elif data['type'] == 'error':
                print(f"Error: {data.get('error', 'Unknown error')}")
                break
            
            time.sleep(0.1)
    finally:
        ws.close()

if __name__ == "__main__":
    prompt = "cinematic shot of a butterfly flying in a garden, smooth motion, detailed wings, natural movement, masterpiece quality"
    try:
        generate_animation(prompt)
        print("\nAnimation saved as butterfly_animation.webp")
    except Exception as e:
        print(f"\nError: {str(e)}")
