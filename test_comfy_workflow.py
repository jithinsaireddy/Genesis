import json
import requests
import time
import os

def check_server_status():
    try:
        response = requests.get("http://127.0.0.1:8188/object_info")
        if response.status_code == 200:
            print("ComfyUI server is running")
            object_info = response.json()
            print("Available nodes:")
            for node_name in object_info:
                if "AnimateDiff" in node_name:
                    print(f"  - {node_name}")
            return True
        else:
            print("ComfyUI server is not responding properly")
            return False
    except Exception as e:
        print(f"Error connecting to ComfyUI server: {str(e)}")
        return False

def check_model_paths():
    required_models = {
        "JuggernautXL": "models/checkpoints/juggernautXL_juggXIByRundiffusion.safetensors",
        "AnimateDiff": "models/animatediff_models/v3_sd15_mm.ckpt"
    }
    
    for model_name, model_path in required_models.items():
        full_path = os.path.join(os.getcwd(), "ComfyUI", model_path)
        if os.path.exists(full_path):
            print(f"{model_name} model found at: {model_path}")
        else:
            print(f"WARNING: {model_name} model not found at: {model_path}")

def queue_prompt(workflow_file):
    with open(workflow_file, 'r') as file:
        workflow = json.load(file)
    
    # API endpoint
    api_url = "http://127.0.0.1:8188/prompt"
    
    # Queue the prompt
    try:
        response = requests.post(api_url, json={
            "prompt": workflow,
            "client_id": "test_script"
        })
        
        if response.status_code == 200:
            print("Successfully queued prompt")
            prompt_id = response.json()['prompt_id']
            return prompt_id
        else:
            print(f"Failed to queue prompt: {response.status_code}")
            print("Response content:", response.text)
            return None
    except Exception as e:
        print(f"Error queuing prompt: {str(e)}")
        return None

def check_progress(prompt_id):
    while True:
        try:
            response = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}")
            if response.status_code == 200:
                data = response.json()[prompt_id]
                if "executing" in data and not data["executing"]:
                    if "error" in data:
                        print(f"Error during generation: {data['error']}")
                        if 'node_errors' in data:
                            for node_id, error in data['node_errors'].items():
                                print(f"Node {node_id} error: {error}")
                    else:
                        print("Generation complete!")
                    break
                print("Still generating...")
            else:
                print(f"Failed to check progress: {response.status_code}")
                break
        except Exception as e:
            print(f"Error checking progress: {str(e)}")
            break
        time.sleep(2)

if __name__ == "__main__":
    print("Checking ComfyUI server status...")
    if not check_server_status():
        print("Please ensure ComfyUI server is running on port 8188")
        exit(1)
        
    print("\nChecking required model paths...")
    check_model_paths()
    
    # Test with simple AnimateDiff workflow
    workflow_file = "ComfyUI/workflows/simple_animatediff_test.json"
    print("\nTesting simple AnimateDiff workflow...")
    
    # Queue the prompt
    prompt_id = queue_prompt(workflow_file)
    if prompt_id:
        # Monitor progress
        check_progress(prompt_id)
        print(f"Output will be saved in ComfyUI/output/")
