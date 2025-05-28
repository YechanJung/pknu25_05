import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
from pesq import pesq
from pystoi import stoi
import os
import shutil

def evaluate_noise_reduction(original_audio, enhanced_audio, sr, output_dir='noise_evaluation'):
    """
    Comprehensive evaluation of noise reduction quality.
    
    Args:
        original_audio: Array of original noisy audio samples
        enhanced_audio: Array of enhanced (noise-reduced) audio samples
        sr: Sample rate
        output_dir: Directory to save the evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Check if enhanced audio is empty
    if enhanced_audio is None or len(enhanced_audio) == 0:
        print("Error: Enhanced audio is empty - cannot evaluate")
        return {}
        
    # Check if original and enhanced audio have compatible shapes
    if len(original_audio.shape) != len(enhanced_audio.shape):
        # Convert mono to stereo or vice versa if needed
        if len(original_audio.shape) == 2 and len(enhanced_audio.shape) == 1:
            # Convert enhanced mono to stereo
            enhanced_audio = np.column_stack((enhanced_audio, enhanced_audio))
            print("Converted enhanced audio from mono to stereo for comparison")
        elif len(original_audio.shape) == 1 and len(enhanced_audio.shape) == 2:
            # Convert original mono to stereo
            original_audio = np.column_stack((original_audio, original_audio))
            print("Converted original audio from mono to stereo for comparison")
    
    # Ensure both signals have the same length
    min_len = min(len(original_audio), len(enhanced_audio))
    if min_len == 0:
        print("Error: One of the audio files is empty")
        return {}
        
    # Check if audio is too short for processing
    if min_len < 256:
        print(f"Warning: Audio is very short ({min_len} samples) - limited analysis possible")
    
    original_audio = original_audio[:min_len]
    enhanced_audio = enhanced_audio[:min_len]
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {}
    
    # 1. Signal-to-Noise Ratio (SNR) improvement
    # Note: This is an approximation since we don't have the clean signal
    noise_power_before = np.mean(np.power(original_audio, 2))
    signal_diff = original_audio - enhanced_audio
    noise_power_after = np.mean(np.power(signal_diff, 2))
    if noise_power_before > 0 and noise_power_after > 0:
        snr_improvement = 10 * np.log10(noise_power_before / noise_power_after)
        metrics['snr_improvement_db'] = snr_improvement
        print(f"Estimated SNR improvement: {snr_improvement:.2f} dB")
    
    # 2. Spectral comparison - create spectrograms
    # Skip if audio is too short for spectrograms
    min_samples_for_spectrogram = 512
    if min_len >= min_samples_for_spectrogram:
        try:
            # Ensure n_fft is appropriate for the audio length
            n_fft = min(2048, min_len // 2)
            
            plt.figure(figsize=(12, 8))
            
            # Original audio spectrogram
            plt.subplot(2, 1, 1)
            spec = librosa.amplitude_to_db(
                np.abs(librosa.stft(original_audio, n_fft=n_fft)), ref=np.max
            )
            librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Original Audio Spectrogram')
            
            # Enhanced audio spectrogram
            plt.subplot(2, 1, 2)
            spec = librosa.amplitude_to_db(
                np.abs(librosa.stft(enhanced_audio, n_fft=n_fft)), ref=np.max
            )
            librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Enhanced Audio Spectrogram')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'spectrogram_comparison.png'))
        except Exception as e:
            print(f"Warning: Could not create spectrograms - {e}")
    else:
        print(f"Warning: Audio too short ({min_len} samples) to create spectrograms")
    
    # 3. Save audio files for listening comparison
    sf.write(os.path.join(output_dir, 'original.wav'), original_audio, sr)
    sf.write(os.path.join(output_dir, 'enhanced.wav'), enhanced_audio, sr)
    
    # 4. Energy distribution analysis - only if audio is long enough
    if min_len >= min_samples_for_spectrogram:
        try:
            # Calculate energy in different frequency bands
            # Select n_fft based on audio length
            n_fft = min(2048, min_len // 2)
            
            # Original audio
            D_original = np.abs(librosa.stft(original_audio, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Define frequency bands (in Hz)
            bands = [
                (0, 500),      # Low frequencies (often contains noise)
                (500, 2000),   # Mid-low (important for speech)
                (2000, 5000),  # Mid-high (important for consonants/speech clarity)
                (5000, sr//2)  # High frequencies (often contains noise)
            ]
            
            band_energy_original = []
            band_energy_enhanced = []
            
            for low, high in bands:
                # Find indices for this frequency band
                band_indices = np.where((freqs >= low) & (freqs <= high))[0]
                
                # Calculate energy in this band for original audio
                original_band_energy = np.sum(np.mean(D_original[band_indices, :], axis=1))
                band_energy_original.append(original_band_energy)
                
                # Enhanced audio
                D_enhanced = np.abs(librosa.stft(enhanced_audio, n_fft=n_fft))
                enhanced_band_energy = np.sum(np.mean(D_enhanced[band_indices, :], axis=1))
                band_energy_enhanced.append(enhanced_band_energy)
            
            # Normalize for percentage
            total_energy_original = sum(band_energy_original)
            total_energy_enhanced = sum(band_energy_enhanced)
            
            if total_energy_original > 0 and total_energy_enhanced > 0:
                band_energy_pct_original = [e/total_energy_original*100 for e in band_energy_original]
                band_energy_pct_enhanced = [e/total_energy_enhanced*100 for e in band_energy_enhanced]
                
                # Display band energy distribution
                plt.figure(figsize=(10, 6))
                band_labels = ['0-500 Hz', '500-2000 Hz', '2000-5000 Hz', f'5000-{sr//2} Hz']
                
                x = np.arange(len(band_labels))
                width = 0.35
                
                plt.bar(x - width/2, band_energy_pct_original, width, label='Original')
                plt.bar(x + width/2, band_energy_pct_enhanced, width, label='Enhanced')
                
                plt.xlabel('Frequency Bands')
                plt.ylabel('Energy (%)')
                plt.title('Energy Distribution Across Frequency Bands')
                plt.xticks(x, band_labels)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'energy_distribution.png'))
                
                # Store metrics
                metrics['energy_distribution_original'] = band_energy_pct_original
                metrics['energy_distribution_enhanced'] = band_energy_pct_enhanced
                
                # Calculate improvement in speech-to-noise ratio in speech frequency bands (500-5000 Hz)
                speech_band_energy_original = band_energy_original[1] + band_energy_original[2]
                noise_band_energy_original = band_energy_original[0] + band_energy_original[3]
                
                speech_band_energy_enhanced = band_energy_enhanced[1] + band_energy_enhanced[2]
                noise_band_energy_enhanced = band_energy_enhanced[0] + band_energy_enhanced[3]
                
                if noise_band_energy_original > 0 and noise_band_energy_enhanced > 0:
                    speech_to_noise_original = speech_band_energy_original / noise_band_energy_original
                    speech_to_noise_enhanced = speech_band_energy_enhanced / noise_band_energy_enhanced
                    
                    metrics['speech_to_noise_ratio_improvement'] = speech_to_noise_enhanced - speech_to_noise_original
                    print(f"Speech-to-noise ratio improvement: {speech_to_noise_enhanced - speech_to_noise_original:.2f}")
        except Exception as e:
            print(f"Warning: Could not create energy distribution analysis - {e}")
    else:
        print(f"Warning: Audio too short ({min_len} samples) for frequency analysis")
    
    return metrics

def compare_with_reference(enhanced_audio, reference_audio, sr):
    """
    Compare enhanced audio with a clean reference (if available)
    using objective speech quality metrics.
    
    Args:
        enhanced_audio: Array of enhanced audio samples
        reference_audio: Array of clean reference audio samples
        sr: Sample rate
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Resample to 16kHz if needed (required by PESQ)
    if sr != 16000:
        from librosa import resample
        enhanced_audio = resample(y=enhanced_audio, orig_sr=sr, target_sr=16000)
        reference_audio = resample(y=reference_audio, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Ensure both signals have the same length
    min_len = min(len(enhanced_audio), len(reference_audio))
    enhanced_audio = enhanced_audio[:min_len]
    reference_audio = reference_audio[:min_len]
    
    # Normalize
    enhanced_audio = enhanced_audio / np.max(np.abs(enhanced_audio))
    reference_audio = reference_audio / np.max(np.abs(reference_audio))
    
    try:
        # PESQ (Perceptual Evaluation of Speech Quality)
        pesq_score = pesq(sr, reference_audio, enhanced_audio, 'wb')  # wb = wideband
        metrics['pesq'] = pesq_score
        print(f"PESQ score: {pesq_score:.3f} (higher is better, range: -0.5 to 4.5)")
        
        # STOI (Short-Time Objective Intelligibility)
        stoi_score = stoi(reference_audio, enhanced_audio, sr, extended=False)
        metrics['stoi'] = stoi_score
        print(f"STOI score: {stoi_score:.3f} (higher is better, range: 0 to 1)")
        
    except Exception as e:
        print(f"Error calculating speech quality metrics: {e}")
        
    return metrics

# Example usage
if __name__ == "__main__":
    # Load original noisy audio and enhanced audio
    from mp4_noise_reducer import noise_reduce_mp4
    
    # Example with MP4 file
    mp4_file = "/Users/jung-yechan/Documents/Research/pknu/MELD.Raw/train/train_splits/dia0_utt3.mp4"
    print(f"\n\nProcessing file: {mp4_file}\n")
    
    # Check if file exists before processing
    if not os.path.exists(mp4_file):
        print(f"Error: File not found: {mp4_file}")
        exit(1)
        
    # Create output directory
    output_dir = "noise_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Extract the original audio first and save it directly without modification
    import tempfile
    import moviepy
    VideoFileClip = moviepy.video.io.VideoFileClip.VideoFileClip
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        orig_temp_path = temp_file.name
    
    try:
        # Extract original audio
        print("Extracting original audio...")
        video = VideoFileClip(mp4_file)
        
        # Check if video has audio
        if video.audio is None:
            print("Error: The video file does not have an audio track")
            video.close()
            exit(1)
            
        video.audio.write_audiofile(orig_temp_path, codec='pcm_s16le', logger=None)
        video.close()
        
        # Save truly original audio directly from extraction (no modifications)
        true_orig_path = os.path.join(output_dir, "true_original.wav")
        shutil.copy(orig_temp_path, true_orig_path)
        print(f"Saved unmodified original audio to: {true_orig_path}")
        
        # Step 2: Perform noise reduction
        print("Applying noise reduction...")
        enhanced, sr = noise_reduce_mp4(mp4_file)
        
        if enhanced is not None and len(enhanced) > 0:
            print(f"Successfully reduced noise: {len(enhanced)} samples")
            
            # Load the original audio for comparison
            original, sr_orig = sf.read(orig_temp_path)
            print(f"Original audio: {original.shape}, enhanced audio: {enhanced.shape}")
            
            # Make sure we actually have audio data
            if len(original) < 10 or len(enhanced) < 10:
                print(f"Warning: Audio files are too short: original={len(original)}, enhanced={len(enhanced)} samples")
                print("This may be due to a corrupted MP4 file or one without proper audio")
                
                # Save whatever we got for debugging
                debug_dir = "debug_audio"
                os.makedirs(debug_dir, exist_ok=True)
                if len(original) > 0:
                    sf.write(os.path.join(debug_dir, "original_debug.wav"), original, sr_orig)
                if len(enhanced) > 0:
                    sf.write(os.path.join(debug_dir, "enhanced_debug.wav"), enhanced, sr)
                    
                print(f"Saved debug audio files to {debug_dir}")
            else:
                # Save enhanced audio directly
                enhanced_path = os.path.join(output_dir, "enhanced.wav")
                sf.write(enhanced_path, enhanced, sr)
                print(f"Saved enhanced audio to: {enhanced_path}")
                
                # Evaluate noise reduction
                # Note: The evaluate_noise_reduction function also saves original.wav and enhanced.wav internally
                # but we've already saved the true original separately
                evaluate_noise_reduction(original, enhanced, sr, output_dir)
            
        else:
            print("Noise reduction failed - no enhanced audio was produced")
            print("Try with a different MP4 file that contains proper audio")
            
        # Clean up temp file
        os.remove(orig_temp_path)
        
    except Exception as e:
        import traceback
        print(f"Error during evaluation: {e}")
        traceback.print_exc()
        if 'orig_temp_path' in locals() and os.path.exists(orig_temp_path):
            os.remove(orig_temp_path) 