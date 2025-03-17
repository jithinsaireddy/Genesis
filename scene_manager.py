from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
import json
import os

@dataclass
class Shot:
    """Represents a single shot in a scene"""
    shot_type: str  # e.g., "medium close-up", "wide shot", "over-the-shoulder"
    description: str
    camera_angle: str
    lighting: str
    character_positions: Optional[Dict[str, str]] = None
    control_type: Optional[str] = None
    control_image_path: Optional[str] = None

@dataclass
class Scene:
    """Represents a complete scene with multiple shots"""
    scene_id: str
    location: str
    time_of_day: str
    mood: str
    characters: List[str]
    shots: List[Shot]
    additional_notes: Optional[str] = None

class SceneManager:
    """
    Manages scene and shot information for GENESIS AI
    Will be integrated with AI Director and Cinematography systems in future phases
    """
    def __init__(self, output_dir: str = "genesis_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenes: List[Scene] = []
        
    def add_scene(self, scene: Scene):
        """Add a new scene to the manager"""
        self.scenes.append(scene)
        self._save_scene_metadata(scene)

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        """Retrieve a scene by its ID"""
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        return None

    def _save_scene_metadata(self, scene: Scene):
        """Save scene metadata for future reference"""
        scene_dir = self.output_dir / scene.scene_id
        scene_dir.mkdir(exist_ok=True)
        
        # Convert scene to dictionary for JSON serialization
        scene_data = {
            "scene_id": scene.scene_id,
            "location": scene.location,
            "time_of_day": scene.time_of_day,
            "mood": scene.mood,
            "characters": scene.characters,
            "shots": [
                {
                    "shot_type": shot.shot_type,
                    "description": shot.description,
                    "camera_angle": shot.camera_angle,
                    "lighting": shot.lighting,
                    "character_positions": shot.character_positions,
                    "control_type": shot.control_type,
                    "control_image_path": shot.control_image_path
                }
                for shot in scene.shots
            ],
            "additional_notes": scene.additional_notes
        }
        
        with open(scene_dir / "metadata.json", "w") as f:
            json.dump(scene_data, f, indent=2)

    def generate_scene_prompt(self, scene: Scene, shot: Shot) -> str:
        """
        Generate a complete prompt for image generation
        This will be replaced by AI Director's output in future phases
        """
        base_prompt = f"{shot.description}, {scene.location}, {scene.time_of_day}"
        atmosphere = f"{scene.mood} atmosphere, {shot.lighting} lighting"
        camera = f"{shot.shot_type}, {shot.camera_angle}"
        
        # Add character positions if specified
        character_desc = ""
        if shot.character_positions:
            character_desc = ", ".join([f"{char} {pos}" for char, pos in shot.character_positions.items()])
        
        # Combine all elements
        return f"{base_prompt}, {atmosphere}, {camera}, {character_desc}, cinematic quality, highly detailed"

    def load_scenes(self):
        """Load previously saved scenes"""
        for scene_dir in self.output_dir.glob("*"):
            if scene_dir.is_dir():
                metadata_file = scene_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        data = json.load(f)
                        shots = [Shot(**shot_data) for shot_data in data["shots"]]
                        scene = Scene(
                            scene_id=data["scene_id"],
                            location=data["location"],
                            time_of_day=data["time_of_day"],
                            mood=data["mood"],
                            characters=data["characters"],
                            shots=shots,
                            additional_notes=data.get("additional_notes")
                        )
                        self.scenes.append(scene)
