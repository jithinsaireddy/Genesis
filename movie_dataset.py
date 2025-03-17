import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import glob
import os
import logging
from typing import List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieShotDataset(Dataset):
    """
    Dataset loader for movie shots used in AI cinematography training.
    Supports loading video files from a specified directory and preprocessing
    frames for training.
    """
    
    def __init__(self, 
                 video_folder: str,
                 frame_size: Tuple[int, int] = (720, 1280),
                 max_frames: Optional[int] = None):
        """
        Initialize the MovieShotDataset.
        
        Args:
            video_folder (str): Path to folder containing movie scene videos
            frame_size (tuple): Target size for frames (height, width)
            max_frames (int, optional): Maximum number of frames to load per video
        """
        self.video_folder = video_folder
        self.frame_size = frame_size
        self.max_frames = max_frames
        
        # Supported video formats
        video_extensions = ['*.mp4', '*.avi', '*.mov']
        self.video_files = []
        
        # Collect all video files
        for ext in video_extensions:
            self.video_files.extend(glob.glob(os.path.join(video_folder, ext)))
        self.video_files = sorted(self.video_files)
        
        if not self.video_files:
            logger.warning(f"No video files found in {video_folder}")
            
        logger.info(f"Found {len(self.video_files)} video files in {video_folder}")

    def __len__(self) -> int:
        return len(self.video_files)

    def preprocess_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """
        Preprocess a single frame.
        
        Args:
            frame (torch.Tensor): Input frame
            
        Returns:
            torch.Tensor: Preprocessed frame
        """
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
        
        # Convert to tensor and normalize
        frame = torch.tensor(frame).permute(2, 0, 1) / 255.0
        
        return frame

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Load and preprocess frames from a video file.
        
        Args:
            idx (int): Index of the video file to load
            
        Returns:
            torch.Tensor: Stack of preprocessed frames from the video
        """
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise RuntimeError(f"Could not open video file: {video_path}")
            
        frames = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                frame = self.preprocess_frame(frame)
                frames.append(frame)
                
                frame_count += 1
                if self.max_frames and frame_count >= self.max_frames:
                    break
                    
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
            raise
            
        finally:
            cap.release()
            
        if not frames:
            logger.error(f"No frames extracted from video: {video_path}")
            raise RuntimeError(f"No frames could be extracted from: {video_path}")
            
        return torch.stack(frames)

def create_movie_dataloader(
    video_folder: str,
    batch_size: int = 1,
    frame_size: Tuple[int, int] = (720, 1280),
    max_frames: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True
) -> Tuple[MovieShotDataset, DataLoader]:
    """
    Create a DataLoader for movie shots.
    
    Args:
        video_folder (str): Path to folder containing movie scenes
        batch_size (int): Batch size for DataLoader
        frame_size (tuple): Target size for frames (height, width)
        max_frames (int, optional): Maximum number of frames per video
        num_workers (int): Number of worker processes for data loading
        shuffle (bool): Whether to shuffle the dataset
        
    Returns:
        tuple: (MovieShotDataset, DataLoader) instances
    """
    dataset = MovieShotDataset(
        video_folder=video_folder,
        frame_size=frame_size,
        max_frames=max_frames
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return dataset, dataloader

if __name__ == "__main__":
    # Example usage
    VIDEO_FOLDER = "movie_scenes"
    
    try:
        # Create dataset and dataloader
        dataset, dataloader = create_movie_dataloader(
            video_folder=VIDEO_FOLDER,
            batch_size=1,
            frame_size=(720, 1280),
            max_frames=100  # Limit frames per video for testing
        )
        
        logger.info(f"Successfully loaded {len(dataset)} movie scenes")
        logger.info(f"Frame size: {dataset.frame_size}")
        
        # Test loading a batch
        for batch in dataloader:
            logger.info(f"Batch shape: {batch.shape}")
            break
            
    except Exception as e:
        logger.error(f"Error in dataset loading: {str(e)}")
