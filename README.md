# GENESIS - AI Filmmaking Pipeline

GENESIS is an advanced AI filmmaking pipeline that combines multiple AI models to generate high-quality animated videos from text descriptions. The system integrates state-of-the-art models for image generation, animation, and video processing.

## Core Components

1. **AI Cinematography Pipeline**:
   - Text-to-Image: JuggernautXL for high-quality keyframe generation
   - Image-to-Video: AnimateDiff for short video generation from keyframes
   - Video Extension: MDVC for creating longer video sequences

2. **AI Editing System**:
   - Scene composition and sequencing
   - Transition effects
   - Visual continuity management
   - Pacing and rhythm control

3. **AI Voice Generation**:
   - Character voice synthesis
   - Dialogue generation and timing
   - Voice emotion and intonation control
   - Audio-visual synchronization

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download required models:
```bash
python download_models.py
```

## Project Structure

- `frame_generator_sdxl.py`: High-quality frame generation using JuggernautXL
- `animatediff_generator.py`: Video animation using AnimateDiff
- `video_frame_generator.py`: Extended video generation with MDVC
- `voice_generator.py`: AI voice synthesis and audio processing
- `scene_manager.py`: Scene composition and management
- `story_generator.py`: AI-driven story and script generation

## Usage

1. Generate initial frames:
```bash
python test_frame_generator.py
```

2. Create animated sequences:
```bash
python test_animatediff.py
```

## Model Details

1. **JuggernautXL**:
   - State-of-the-art text-to-image model
   - High-quality image generation
   - Excellent prompt understanding

2. **AnimateDiff**:
   - Advanced animation model
   - Smooth motion generation
   - Temporal consistency preservation

3. **MDVC (Motion-Driven Video Creation)**:
   - Long-form video generation
   - Scene transition handling
   - Motion continuity management
