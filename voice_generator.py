import torchaudio
from TTS.api import TTS

# Initialize AI voice model
tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")

def generate_natural_voice(text, output_file="ai_voice.wav"):
    """
    Generate natural-sounding AI voice from text input.
    
    Args:
        text (str): The text to convert to speech
        output_file (str): Path to save the generated audio file (default: ai_voice.wav)
    """
    tts.tts_to_file(text, file_path=output_file)
    return output_file

if __name__ == "__main__":
    # Test the voice generation
    test_text = "This is an AI-generated voice for GENESIS."
    output_path = generate_natural_voice(test_text)
    print(f"High-quality AI voice generated and saved to: {output_path}")
