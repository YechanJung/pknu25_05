import librosa
import numpy as np
import scipy.stats
import scipy.signal

def extract_joy_signature(utterance, sample_rate=16000):
    """
    Extract joy signature from a speech utterance.
    
    Args:
        utterance: Audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Joy signature score
    """
    utterance = np.array(utterance)
    if len(utterance) == 0:
        return 0.0
        
    try:
        # Extract energy envelope
        energy = librosa.feature.rms(
            y=utterance,
            frame_length=int(0.025 * sample_rate),
            hop_length=int(0.010 * sample_rate)
        )[0]
        
        if len(energy) == 0:
            return 0.0
            
        # Compute linear trend
        n_frames = len(energy)
        if n_frames < 2 or np.all(energy == energy[0]):
            trend = 0.0
        else:
            time_indices = np.linspace(0, 1, n_frames)
            slope, _, _, _, _ = scipy.stats.linregress(time_indices, energy)
            feature_range = np.max(energy) - np.min(energy)
            trend = slope / feature_range if feature_range > 0 else 0.0
            
        # Compute dynamism
        if len(energy) < 3:
            dynamism = 0.0
        else:
            first_derivative = np.diff(energy)
            second_derivative = np.diff(first_derivative)
            feature_range = np.max(energy) - np.min(energy)
            if feature_range > 0:
                dynamism = (0.5 * np.std(first_derivative) + 
                          0.3 * np.mean(np.abs(first_derivative)) + 
                          0.2 * np.mean(np.abs(second_derivative))) / feature_range
            else:
                dynamism = 0.0
                
        # Compute burst rate
        normalized = (energy - np.mean(energy)) / np.std(energy) if np.std(energy) > 0 else np.zeros_like(energy)
        above_threshold = normalized > 1.5
        burst_count = 0
        in_burst = False
        
        for is_above in above_threshold:
            if is_above and not in_burst:
                in_burst = True
            elif not is_above and in_burst:
                burst_count += 1
                in_burst = False
        if in_burst:
            burst_count += 1
            
        burst_rate = burst_count / n_frames if n_frames > 0 else 0
        
        # Compute attack rate
        normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy)) if np.max(energy) > np.min(energy) else np.zeros_like(energy)
        attack_rates = []
        in_attack = False
        attack_start = 0
        
        for i in range(1, len(normalized)):
            if normalized[i] > normalized[i-1] + 0.05:
                if not in_attack:
                    in_attack = True
                    attack_start = i - 1
            elif in_attack:
                attack_duration = i - attack_start
                attack_magnitude = normalized[i-1] - normalized[attack_start]
                if attack_duration > 0:
                    attack_rates.append(attack_magnitude / attack_duration)
                in_attack = False
                
        mean_attack_rate = np.mean(attack_rates) if attack_rates else 0
        
        # Compute peak rate
        peaks, _ = scipy.signal.find_peaks(normalized, distance=5, prominence=0.1)
        peak_rate = len(peaks) / n_frames if n_frames > 0 else 0
        
        # Compute joy signature
        joy_signature = (
            0.3 * trend +
            0.2 * burst_rate +
            0.2 * mean_attack_rate +
            0.15 * peak_rate +
            0.15 * dynamism
        )
        
        return joy_signature
        
    except Exception as e:
        print(f"Error computing joy signature: {e}")
        return 0.0