import torch
import torchaudio
import os
import soundfile as sf
import numpy as np
from voice_gender_classifier.model import ECAPA_gender

# Initialize model
# https://huggingface.co/JaesungHuh/voice-gender-classifier
model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

def load_audio_with_soundfile(file_path):
    """Load audio using soundfile and convert to torch tensor format expected by the model."""
    try:
        audio, sr = sf.read(file_path)
        audio = torch.from_numpy(audio).float()
        if audio.dim() > 1:
            audio = audio.mean(dim=1)
        audio = audio.unsqueeze(0)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)            
        return audio
    except Exception as e:
        print(f"Error loading audio with soundfile: {e}")
        return None

def classify_gender(file_path):
    """
    Classify the gender of a voice in an audio file.
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        dict: Classification results
    """
    try:
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        audio = load_audio_with_soundfile(file_path)
        if audio is None:
            return None
        audio = audio.to(device)
        model.eval()
        with torch.no_grad():
            output = model.forward(audio)
            _, pred = output.max(1)
            result = model.pred2gender[pred.item()]
            print(f"Classification result: {result}")
            return result
            
    except Exception as e:
        print(f"Error during gender classification: {e}")
        return None

# if __name__ == "__main__":
#     # Example usage
#     example_file = "noise_evaluation/enhanced.wav"
#     if os.path.exists(example_file):
#         classify_gender(example_file)
#     else:
#         print(f"Example file not found: {example_file}")
