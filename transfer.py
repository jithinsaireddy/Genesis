import os
import subprocess
from pathlib import Path

def transfer_files(source_dir, remote_host, remote_path, key_path, chunk_size=1000):
    # Get all image files
    files = []
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        files.extend(Path(source_dir).rglob(ext))
    
    # Process in chunks
    for i in range(0, len(files), chunk_size):
        chunk = files[i:i + chunk_size]
        # Create rsync command
        files_str = ' '.join(str(f.relative_to(source_dir)) for f in chunk)
        cmd = f'cd {source_dir} && rsync -av --progress {files_str} -e "ssh -i {key_path}" {remote_host}:{remote_path}'
        
        try:
            subprocess.run(cmd, shell=True, check=True)
            print(f'Transferred chunk {i//chunk_size + 1} of {(len(files) + chunk_size - 1)//chunk_size}')
        except subprocess.CalledProcessError as e:
            print(f'Error transferring chunk {i//chunk_size + 1}: {e}')

if __name__ == '__main__':
    source_dir = 'training_videos'
    remote_host = 'ubuntu@192.222.57.228'
    remote_path = '~/GENESIS/training_videos/'
    key_path = '~/Downloads/my-ai-data.pem'
    
    transfer_files(source_dir, remote_host, remote_path, key_path)
