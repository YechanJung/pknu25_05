import numpy as np
import librosa
import soundfile as sf
import os
import tempfile
import moviepy
import pandas as pd
from mp4_noise_reducer import noise_reduce_mp4

VideoFileClip = moviepy.video.io.VideoFileClip.VideoFileClip

def calculate_audio_metrics(original_audio, enhanced_audio, sr):
    """
    Calculate comprehensive audio quality metrics.
    
    Args:
        original_audio: Original noisy audio samples
        enhanced_audio: Enhanced (noise-reduced) audio samples
        sr: Sample rate
        
    Returns:
        Dictionary with calculated metrics
    """
    metrics = {}
    
    # Ensure both signals have the same length
    min_len = min(len(original_audio), len(enhanced_audio))
    if min_len == 0:
        return metrics
        
    original_audio = original_audio[:min_len]
    enhanced_audio = enhanced_audio[:min_len]
    
    # 1. RMS (Root Mean Square) values
    rms_original = np.sqrt(np.mean(original_audio**2))
    rms_enhanced = np.sqrt(np.mean(enhanced_audio**2))
    metrics['rms_original'] = rms_original
    metrics['rms_enhanced'] = rms_enhanced
    metrics['rms_reduction_db'] = 20 * np.log10(rms_original / rms_enhanced) if rms_enhanced > 0 else 0
    
    # 2. Signal-to-Noise Ratio improvement estimation
    noise_power_before = np.mean(np.power(original_audio, 2))
    signal_diff = original_audio - enhanced_audio
    noise_power_after = np.mean(np.power(signal_diff, 2))
    
    if noise_power_before > 0 and noise_power_after > 0:
        snr_improvement = 10 * np.log10(noise_power_before / noise_power_after)
        metrics['snr_improvement_db'] = snr_improvement
    else:
        metrics['snr_improvement_db'] = 0
    
    # 3. Spectral centroid (brightness measure)
    try:
        centroid_original = np.mean(librosa.feature.spectral_centroid(y=original_audio, sr=sr))
        centroid_enhanced = np.mean(librosa.feature.spectral_centroid(y=enhanced_audio, sr=sr))
        metrics['spectral_centroid_original'] = centroid_original
        metrics['spectral_centroid_enhanced'] = centroid_enhanced
        metrics['spectral_centroid_change'] = centroid_enhanced - centroid_original
    except:
        metrics['spectral_centroid_original'] = 0
        metrics['spectral_centroid_enhanced'] = 0
        metrics['spectral_centroid_change'] = 0
    
    # 4. Zero crossing rate (measure of noisiness)
    try:
        zcr_original = np.mean(librosa.feature.zero_crossing_rate(original_audio))
        zcr_enhanced = np.mean(librosa.feature.zero_crossing_rate(enhanced_audio))
        metrics['zero_crossing_rate_original'] = zcr_original
        metrics['zero_crossing_rate_enhanced'] = zcr_enhanced
        metrics['zero_crossing_rate_reduction'] = zcr_original - zcr_enhanced
    except:
        metrics['zero_crossing_rate_original'] = 0
        metrics['zero_crossing_rate_enhanced'] = 0
        metrics['zero_crossing_rate_reduction'] = 0
    
    # 5. Energy distribution in frequency bands
    if min_len >= 512:
        try:
            n_fft = min(2048, min_len // 2)
            
            # Calculate STFT for both signals
            D_original = np.abs(librosa.stft(original_audio, n_fft=n_fft))
            D_enhanced = np.abs(librosa.stft(enhanced_audio, n_fft=n_fft))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Define frequency bands
            bands = {
                'low_freq_0_500': (0, 500),
                'speech_low_500_2000': (500, 2000),
                'speech_high_2000_5000': (2000, 5000),
                'high_freq_5000_plus': (5000, sr//2)
            }
            
            for band_name, (low, high) in bands.items():
                band_indices = np.where((freqs >= low) & (freqs <= high))[0]
                
                if len(band_indices) > 0:
                    original_energy = np.sum(np.mean(D_original[band_indices, :], axis=1))
                    enhanced_energy = np.sum(np.mean(D_enhanced[band_indices, :], axis=1))
                    
                    metrics[f'{band_name}_energy_original'] = original_energy
                    metrics[f'{band_name}_energy_enhanced'] = enhanced_energy
                    
                    if original_energy > 0:
                        energy_reduction = (original_energy - enhanced_energy) / original_energy * 100
                        metrics[f'{band_name}_energy_reduction_pct'] = energy_reduction
                    else:
                        metrics[f'{band_name}_energy_reduction_pct'] = 0
        except Exception as e:
            print(f"Warning: Could not calculate frequency band metrics - {e}")
    
    return metrics

def convert_mp4_to_enhanced_wav(mp4_file_path, output_dir="enhanced_audio"):
    """
    Convert a single MP4 file to noise-reduced WAV with metrics.
    
    Args:
        mp4_file_path: Path to the MP4 file
        output_dir: Directory to save the enhanced WAV file
        
    Returns:
        Dictionary with metrics and file information
    """
    filename = os.path.basename(mp4_file_path)
    
    if not os.path.exists(mp4_file_path):
        print(f"Error: File not found: {mp4_file_path}")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Apply noise reduction (this also extracts audio internally)
        print(f"Applying noise reduction to {filename}...")
        enhanced_audio, sr = noise_reduce_mp4(mp4_file_path, debug=False)
        
        if enhanced_audio is None or len(enhanced_audio) == 0:
            print("Error: Noise reduction failed")
            return None
        
        # Step 2: Extract original audio for metrics comparison
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_audio_path = temp_file.name
        
        print(f"Extracting original audio for metrics...")
        try:
            # Try moviepy first
            video = VideoFileClip(mp4_file_path)
            
            if video.audio is None:
                print("Error: The video file does not have an audio track")
                video.close()
                return None
            
            # Use verbose=False and logger=None to suppress moviepy output
            video.audio.write_audiofile(temp_audio_path, codec='pcm_s16le', verbose=False, logger=None)
            video.close()
            
        except Exception as moviepy_error:
            print(f"Moviepy extraction failed, trying ffmpeg directly...")
            
            # Fallback to direct ffmpeg
            import subprocess
            try:
                cmd = ['ffmpeg', '-i', mp4_file_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '48000', '-ac', '1', temp_audio_path, '-y']
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    print(f"FFmpeg extraction also failed: {stderr.decode()}")
                    os.remove(temp_audio_path)
                    return None
                    
            except Exception as ffmpeg_error:
                print(f"Both moviepy and ffmpeg failed: {ffmpeg_error}")
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return None
        
        # Step 3: Save enhanced audio
        enhanced_wav_path = os.path.join(output_dir, f"{filename.replace('.mp4', '')}_enhanced.wav")
        sf.write(enhanced_wav_path, enhanced_audio, sr)
        print(f"Saved enhanced audio to: {enhanced_wav_path}")
        
        # Step 4: Load original audio for metrics comparison
        try:
            original_audio, sr_orig = sf.read(temp_audio_path)
        except Exception as e:
            print(f"Error reading original audio for metrics: {e}")
            # Clean up and return partial result
            os.remove(temp_audio_path)
            return {
                'filename': filename,
                'original_duration_seconds': 0,
                'enhanced_duration_seconds': len(enhanced_audio) / sr,
                'sample_rate': sr,
                'enhanced_wav_path': enhanced_wav_path,
                'snr_improvement_db': 0,
                'rms_reduction_db': 0
            }
        
        # Step 5: Calculate metrics
        print(f"Calculating metrics for {filename}...")
        metrics = calculate_audio_metrics(original_audio, enhanced_audio, sr)
        
        # Add file information
        result = {
            'filename': filename,
            'original_duration_seconds': len(original_audio) / sr_orig,
            'enhanced_duration_seconds': len(enhanced_audio) / sr,
            'sample_rate': sr,
            'enhanced_wav_path': enhanced_wav_path,
            **metrics
        }
        
        # Clean up temporary file
        os.remove(temp_audio_path)
        
        print(f"Successfully processed {filename}")
        print(f"SNR improvement: {metrics.get('snr_improvement_db', 0):.2f} dB")
        print(f"RMS reduction: {metrics.get('rms_reduction_db', 0):.2f} dB")
        
        return result
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        # Clean up temp file if it exists
        if 'temp_audio_path' in locals() and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return None

if __name__ == "__main__":
    # Configuration - modify this path to your MP4 directory
    mp4_directory = "/Users/jung-yechan/Documents/Research/pknu/MELD.Raw/train/train_splits/"
    output_directory = "enhanced_audio"
    csv_filename = "noise_reduction_metrics.csv"
    
    print("=== MP4 to Enhanced WAV Converter ===")
    print(f"Processing all MP4 files in: {mp4_directory}")
    print(f"Output directory: {output_directory}")
    print(f"CSV metrics file: {csv_filename}")
    print()
    
    # Check if directory exists
    if not os.path.exists(mp4_directory):
        print(f"Error: Directory not found: {mp4_directory}")
        exit(1)
    
    # Get all MP4 files in the directory
    mp4_files = [f for f in os.listdir(mp4_directory) if f.lower().endswith('.mp4')]
    
    if not mp4_files:
        print(f"No MP4 files found in {mp4_directory}")
        exit(1)
    
    print(f"Found {len(mp4_files)} MP4 files to process")
    print()
    
    # Process each MP4 file and collect metrics
    successful = 0
    failed = 0
    all_metrics = []
    
    for i, mp4_file in enumerate(mp4_files, 1):
        print(f"[{i}/{len(mp4_files)}] Processing: {mp4_file}")
        
        mp4_path = os.path.join(mp4_directory, mp4_file)
        result = convert_mp4_to_enhanced_wav(mp4_path, output_directory)
        
        if result:
            successful += 1
            print(f"✓ Success: {result['enhanced_wav_path']}")
            
            # Collect metrics for CSV
            metrics_row = {
                'filename': result['filename'],
                'snr_improvement_db': result.get('snr_improvement_db', 0),
                'rms_reduction_db': result.get('rms_reduction_db', 0),
                'original_duration_seconds': result.get('original_duration_seconds', 0),
                'enhanced_duration_seconds': result.get('enhanced_duration_seconds', 0),
                'sample_rate': result.get('sample_rate', 0),
                'enhanced_wav_path': result['enhanced_wav_path']
            }
            all_metrics.append(metrics_row)
        else:
            failed += 1
            print(f"✗ Failed: {mp4_file}")
            
            # Add failed entry to metrics
            metrics_row = {
                'filename': mp4_file,
                'snr_improvement_db': 0,
                'rms_reduction_db': 0,
                'original_duration_seconds': 0,
                'enhanced_duration_seconds': 0,
                'sample_rate': 0,
                'enhanced_wav_path': 'FAILED'
            }
            all_metrics.append(metrics_row)
        
        print("-" * 50)
    
    # Save metrics to CSV
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        csv_path = os.path.join(output_directory, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to: {csv_path}")
    
    # Summary
    print(f"\n=== PROCESSING COMPLETE ===")
    print(f"Total files: {len(mp4_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Enhanced WAV files saved to: {output_directory}")
    print(f"Metrics CSV saved to: {csv_path}")
    
    # Display summary statistics
    if successful > 0:
        successful_df = df[df['enhanced_wav_path'] != 'FAILED']
        avg_snr = successful_df['snr_improvement_db'].mean()
        avg_rms = successful_df['rms_reduction_db'].mean()
        print(f"\nAverage SNR improvement: {avg_snr:.2f} dB")
        print(f"Average RMS reduction: {avg_rms:.2f} dB")