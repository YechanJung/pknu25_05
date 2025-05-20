import torch
from denoiser import pretrained
from denoiser.dsp import convert_audio
import soundfile as sf
import os
import numpy as np
import sys
import traceback
# Try alternative import approach for moviepy
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback import approach
    import moviepy
    VideoFileClip = moviepy.video.io.VideoFileClip.VideoFileClip
# https://github.com/facebookresearch/denoiser
# !pip install denoiser
# import torch
# from denoiser import pretrained
# from denoiser.dsp import convert_audio
# import soundfile as sf
# def noise_reduce():
#     # load model
#     model = pretrained.dns64()
#     model.eval()

#     # audio upload
#     noisy, sr = sf.read("noisy_audio.wav")

#     # convert to model format
#     noisy = torch.from_numpy(noisy).float()
#     if noisy.dim() == 1:
#         noisy = noisy.unsqueeze(0)  # [Time] -> [Batch, Time]
#         noisy = noisy.unsqueeze(1)  # [Batch, Time] -> [Batch, Channel, Time]

#     # if needed, resampling
#     if sr != 16000:
#         noisy = convert_audio(noisy, sr, 16000)
#         sr = 16000

#     # apply noise reduction
#     with torch.no_grad():
#         enhanced = model(noisy)[0].squeeze(0).cpu().numpy()

    
#     return enhanced, sr

def noise_reduce_mp4(mp4_file_path, output_path=None, debug=True):
    """Extract audio from MP4 file and apply noise reduction.
    
    Args:
        mp4_file_path: Path to the MP4 file
        output_path: Optional path to save the enhanced audio
        debug: Whether to print detailed debug information
        
    Returns:
        Tuple of (enhanced_audio, sample_rate)
    """
    try:
        # Check if file exists
        if not os.path.exists(mp4_file_path):
            if debug: print(f"Error: File not found: {mp4_file_path}")
            return None, None
            
        # Extract audio from MP4 file
        if debug: print(f"Extracting audio from MP4 file: {mp4_file_path}")
        video = VideoFileClip(mp4_file_path)
        
        # Check if video has audio
        if video.audio is None:
            if debug: print("Error: The video file does not have an audio track")
            video.close()
            return None, None
            
        temp_audio_path = "temp_audio.wav"
        
        # Write audio to temporary WAV file
        if debug: print("Writing audio to temporary file...")
        video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', logger=None)
        video.close()
        
        # Check if temp file was created and has content
        if not os.path.exists(temp_audio_path):
            if debug: print("Error: Failed to create temporary audio file")
            return None, None
        
        if os.path.getsize(temp_audio_path) == 0:
            if debug: print("Error: Temporary audio file is empty")
            os.remove(temp_audio_path)
            return None, None
        
        # Load model
        if debug: print("Loading noise reduction model...")
        model = pretrained.dns64()
        model.eval()
        
        # Load extracted audio
        if debug: print(f"Loading audio from {temp_audio_path}...")
        try:
            noisy, sr = sf.read(temp_audio_path)
        except Exception as e:
            if debug: print(f"Error loading audio: {e}")
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            return None, None
        
        # Check if audio data was loaded
        if debug: print(f"Audio loaded: shape={noisy.shape}, sample_rate={sr}")
        if len(noisy) == 0:
            if debug: print("Error: Extracted audio is empty")
            os.remove(temp_audio_path)
            return None, None
        
        # Convert to model format
        if debug: print("Converting audio to model format...")
        try:
            noisy = torch.from_numpy(noisy).float()
            
            # Handle stereo or mono
            if noisy.dim() == 1:  # mono
                if debug: print("Processing mono audio")
                noisy = noisy.unsqueeze(0)  # [Time] -> [Batch, Time]
                noisy = noisy.unsqueeze(1)  # [Batch, Time] -> [Batch, Channel, Time]
            elif noisy.dim() == 2:  # stereo
                if debug: print(f"Processing stereo audio, shape: {noisy.shape}")
                # Take the first channel only for noise reduction
                noisy = noisy[:, 0]
                noisy = noisy.unsqueeze(0).unsqueeze(1)  # [Time] -> [Batch, Channel, Time]
        
            if debug: print(f"Audio tensor shape after formatting: {noisy.shape}")
        except Exception as e:
            if debug:
                print(f"Error during audio format conversion: {e}")
                traceback.print_exc()
            os.remove(temp_audio_path)
            return None, None
        
        # Resample if needed
        if sr != 16000:
            try:
                # Add the channels parameter (1 for mono, 2 for stereo)
                channels = 1  # We're always using mono for processing
                if debug: print(f"Resampling from {sr} Hz to 16000 Hz with {channels} channels")
                noisy = convert_audio(noisy, sr, 16000, channels)
                sr = 16000
            except Exception as e:
                if debug:
                    print(f"Error during resampling: {e}")
                    traceback.print_exc()
                os.remove(temp_audio_path)
                return None, None
        
        # Apply noise reduction
        if debug: print("Applying noise reduction...")
        try:
            with torch.no_grad():
                enhanced = model(noisy)[0].squeeze(0).cpu().numpy()
            
            if debug: print(f"Enhanced audio shape: {enhanced.shape}")
            
            # Test if enhanced audio is valid
            if len(enhanced) == 0:
                if debug: print("Error: Enhanced audio is empty")
                os.remove(temp_audio_path)
                return None, None
                
            # Clip any extreme values to avoid audio artifacts
            enhanced = np.clip(enhanced, -1.0, 1.0)
        except Exception as e:
            if debug:
                print(f"Error during noise reduction: {e}")
                traceback.print_exc()
            os.remove(temp_audio_path)
            return None, None
        
        # Save enhanced audio if output path is provided
        if output_path:
            sf.write(output_path, enhanced, sr)
            if debug: print(f"Enhanced audio saved to {output_path}")
        
        # Clean up temporary file
        os.remove(temp_audio_path)
        
        return enhanced, sr
        
    except Exception as e:
        if debug:
            print(f"Unexpected error during noise reduction: {e}")
            traceback.print_exc()
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return None, None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python mp4_noise_reducer.py <mp4_file_path> [output_wav_path]")
        sys.exit(1)
    
    mp4_file_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"enhanced_{os.path.basename(mp4_file_path).replace('.mp4', '.wav')}"
    
    enhanced, sr = noise_reduce_mp4(mp4_file_path, output_path)
    
    if enhanced is not None:
        print(f"Successfully processed audio: {len(enhanced)} samples at {sr} Hz")
    else:
        print("Audio processing failed") 