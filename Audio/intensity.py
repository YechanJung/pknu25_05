import librosa
import numpy as np
import scipy.stats
import scipy.signal  

def compute_linear_trend(feature_sequence):
    """
    Calculate the slope of the linear trend in a feature sequence.
    Positive values indicate rising trend, negative values indicate falling trend.
    
    Args:
        feature_sequence: Array of feature values over time
        
    Returns:
        Slope of the best-fit line

    Reference:

    Schuller, B., Steidl, S., Batliner, A., Burkhardt, F., Devillers, L., Müller, C., & Narayanan, S. (2013). 
        Paralinguistics in speech and language—State-of-the-art and the challenge. Computer Speech & Language, 27(1), 4-39.
        Shows linear regression coefficients of energy contours as key predictors of emotion intensity.

    Batliner, A., Steidl, S., Schuller, B., Seppi, D., Vogt, T., Wagner, J., ... & Amir, N. (2011).
        Whodunnit-searching for the most important feature types signalling emotion-related user states in speech.
        Computer Speech & Language, 25(1), 4-28. Validates linear trend features for emotion intensity prediction.

    """
    # Create time indices (normalized to [0,1])
    feature_sequence = np.array(feature_sequence)  
    n_frames = len(feature_sequence)
    
    # Handle edge cases
    if n_frames < 2:
        return 0.0
    
    # Check if array contains only identical values
    if np.all(feature_sequence == feature_sequence[0]):
        return 0.0
    
    time_indices = np.linspace(0, 1, n_frames)
    
    # Compute linear regression
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(
        time_indices, feature_sequence
    )
    
    # Normalize by feature range to handle different scales
    feature_range = np.max(feature_sequence) - np.min(feature_sequence)
    if feature_range > 0:
        normalized_slope = slope / feature_range
    else:
        normalized_slope = 0.0
    
    return normalized_slope

def compute_feature_dynamism(feature_sequence):
    """
    Calculate the dynamism (variability in rate of change) of a feature.
    Higher values indicate more dynamic expression.
    
    Args:
        feature_sequence: Array of feature values over time
        
    Returns:
        Dynamism score

    References:
    
    Schuller, B., Batliner, A., Steidl, S., & Seppi, D. (2011). 
        Recognising realistic emotions and affect in speech: State of the art and lessons learnt from the first challenge.Speech Communication, 53(9-10), 1062-1087.
        Demonstrates that first and second derivatives of prosodic features significantly improve emotion intensity recognition.

    Eyben, F., Scherer, K. R., Schuller, B. W., Sundberg, J., André, E., Busso, C., ... & Truong, K. P. (2016). 
        The Geneva minimalistic acoustic parameter set (GeMAPS) for voice research and affective computing. IEEE Transactions on Affective Computing, 7(2), 190-202. 
        Defines dynamism features as key performance indicators for emotion intensity detection.
    """
    feature_sequence = np.array(feature_sequence)  
    
    # Handle edge cases
    if len(feature_sequence) < 3:  # Need at least 3 points for second derivative
        print ("not enough len")
        return 0.0
    
    # Compute first derivative (rate of change)
    first_derivative = np.diff(feature_sequence)
    
    # Compute second derivative (acceleration)
    second_derivative = np.diff(first_derivative)
    
    # Calculate metrics
    first_derivative_mean = np.mean(np.abs(first_derivative))
    first_derivative_std = np.std(first_derivative)
    second_derivative_mean = np.mean(np.abs(second_derivative))
    
    # Combine into dynamism measure (normalized to feature range)
    feature_range = np.max(feature_sequence) - np.min(feature_sequence)
    if feature_range > 0:
        dynamism = (0.5 * first_derivative_std + 
                   0.3 * first_derivative_mean + 
                   0.2 * second_derivative_mean) / feature_range
    else:
        dynamism = 0.0
    
    return dynamism

def detect_bursts(feature_sequence, threshold=1.5, min_duration=3):
    """
    Detect burst patterns in a feature sequence.
    
    Args:
        feature_sequence: Array of feature values over time
        threshold: Threshold for burst detection (multiplier over local mean)
        min_duration: Minimum frames to be considered a burst
        
    Returns:
        List of (start_index, end_index, magnitude) for each burst
    """
    feature_sequence = np.array(feature_sequence)  
    
    # Handle empty array
    if len(feature_sequence) == 0:
        return []
    
    # Normalize feature sequence
    feature_mean = np.mean(feature_sequence)
    feature_std = np.std(feature_sequence)
    if feature_std > 0:
        normalized_sequence = (feature_sequence - feature_mean) / feature_std
    else:
        return []
    
    # Detect regions above threshold
    above_threshold = normalized_sequence > threshold
    
    # Find continuous regions (bursts)
    bursts = []
    in_burst = False
    burst_start = 0
    
    for i, is_above in enumerate(above_threshold):
        if is_above and not in_burst:
            # Start of new burst
            in_burst = True
            burst_start = i
        elif not is_above and in_burst:
            # End of burst
            burst_end = i
            burst_duration = burst_end - burst_start
            
            if burst_duration >= min_duration:
                # Calculate burst magnitude
                burst_magnitude = np.mean(normalized_sequence[burst_start:burst_end])
                bursts.append((burst_start, burst_end, burst_magnitude))
            
            in_burst = False
    
    # Handle case where sequence ends during a burst
    if in_burst:
        burst_end = len(above_threshold)
        burst_duration = burst_end - burst_start
        
        if burst_duration >= min_duration:
            burst_magnitude = np.mean(normalized_sequence[burst_start:burst_end])
            bursts.append((burst_start, burst_end, burst_magnitude))
    
    return bursts

def compute_burst_features(feature_sequence, threshold=1.5, min_duration=3):
    """
    Compute features based on burst patterns.
    
    Args:
        feature_sequence: Array of feature values over time
        threshold: Threshold for burst detection
        min_duration: Minimum frames for a burst
        
    Returns:
        Dictionary of burst-related features
    """
    feature_sequence = np.array(feature_sequence)  
    
    bursts = detect_bursts(feature_sequence, threshold, min_duration)
    
    n_frames = len(feature_sequence)
    if n_frames == 0:
        return {
            'burst_count': 0,
            'burst_rate': 0,
            'mean_burst_magnitude': 0,
            'max_burst_magnitude': 0,
            'burst_coverage': 0,
            'first_burst_position': 0
        }
    
    # Compute burst features
    burst_count = len(bursts)
    burst_rate = burst_count / n_frames
    
    if burst_count > 0:
        burst_magnitudes = [b[2] for b in bursts]
        burst_durations = [b[1] - b[0] for b in bursts]
        total_burst_frames = sum(burst_durations)
        
        mean_burst_magnitude = np.mean(burst_magnitudes)
        max_burst_magnitude = np.max(burst_magnitudes)
        burst_coverage = total_burst_frames / n_frames
        first_burst_position = bursts[0][0] / n_frames if bursts else 0
    else:
        mean_burst_magnitude = 0
        max_burst_magnitude = 0
        burst_coverage = 0
        first_burst_position = 0
    
    return {
        'burst_count': burst_count,
        'burst_rate': burst_rate,
        'mean_burst_magnitude': mean_burst_magnitude,
        'max_burst_magnitude': max_burst_magnitude,
        'burst_coverage': burst_coverage,
        'first_burst_position': first_burst_position
    }

def analyze_peaks(feature_sequence, distance=5, prominence=0.1, width=None):
    """
    Perform comprehensive peak analysis on a feature sequence.
    
    Args:
        feature_sequence: Array of feature values over time
        distance: Minimum samples between peaks
        prominence: Minimum peak prominence
        width: Minimum peak width
        
    Returns:
        Dictionary of peak-related features
    """
    feature_sequence = np.array(feature_sequence)  
    
    # Handle empty array
    n_frames = len(feature_sequence)
    if n_frames == 0:
        return {
            'peak_count': 0,
            'peak_rate': 0,
            'mean_peak_prominence': 0,
            'max_peak_prominence': 0,
            'peak_position_mean': 0.5,
            'peak_position_std': 0,
            'peak_width_mean': 0
        }
    
    # Normalize feature sequence to [0,1]
    if np.max(feature_sequence) > np.min(feature_sequence):
        normalized = (feature_sequence - np.min(feature_sequence)) / (np.max(feature_sequence) - np.min(feature_sequence))
    else:
        normalized = np.zeros_like(feature_sequence)
    
    # Find peaks with try/except to handle potential errors
    try:
        peaks, properties = scipy.signal.find_peaks(
            normalized,
            distance=distance,
            prominence=prominence,
            width=width
        )
    except Exception as e:
        print(f"Error finding peaks: {e}")
        return {
            'peak_count': 0,
            'peak_rate': 0,
            'mean_peak_prominence': 0,
            'max_peak_prominence': 0,
            'peak_position_mean': 0.5,
            'peak_position_std': 0,
            'peak_width_mean': 0
        }
    
    # Compute peak features
    peak_count = len(peaks)
    peak_rate = peak_count / n_frames
    
    if peak_count > 0:
        # Make sure properties contains all expected keys
        peak_prominences = properties.get('prominences', np.zeros(peak_count))
        peak_widths = properties.get('widths', np.zeros(peak_count))
        peak_positions = peaks / n_frames  # Normalize positions to [0,1]
        
        mean_peak_prominence = np.mean(peak_prominences)
        max_peak_prominence = np.max(peak_prominences) if len(peak_prominences) > 0 else 0
        peak_position_mean = np.mean(peak_positions)
        peak_position_std = np.std(peak_positions)
        peak_width_mean = np.mean(peak_widths) / n_frames if len(peak_widths) > 0 else 0  # Normalize to utterance length
    else:
        mean_peak_prominence = 0
        max_peak_prominence = 0
        peak_position_mean = 0.5  # Default to middle
        peak_position_std = 0
        peak_width_mean = 0
    
    return {
        'peak_count': peak_count,
        'peak_rate': peak_rate,
        'mean_peak_prominence': mean_peak_prominence,
        'max_peak_prominence': max_peak_prominence,
        'peak_position_mean': peak_position_mean,
        'peak_position_std': peak_position_std,
        'peak_width_mean': peak_width_mean
    }

def extract_energy_envelope(audio_signal, sample_rate=16000, window_size=0.025, window_step=0.010):
    """
    Extract energy envelope from audio signal.
    
    Args:
        audio_signal: Raw audio samples
        sample_rate: Sample rate in Hz
        window_size: Window size in seconds
        window_step: Window step in seconds
        
    Returns:
        Energy envelope (one value per frame)
    """
    audio_signal = np.array(audio_signal) 
    
    # Handle empty array
    if len(audio_signal) == 0:
        return np.array([])
    
    # Ensure audio is 1D
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal.flatten()
    
    # Convert window sizes from seconds to samples
    window_length = int(window_size * sample_rate)
    hop_length = int(window_step * sample_rate)
    
    # Ensure valid window and hop lengths
    window_length = max(window_length, 1)
    hop_length = max(hop_length, 1)
    
    try:
        # Compute RMS energy in each frame
        energy = librosa.feature.rms(
            y=audio_signal,
            frame_length=window_length,
            hop_length=hop_length
        )[0]
    except Exception as e:
        print(f"Error extracting energy envelope: {e}")
        return np.array([])
    
    return energy

def compute_envelope_features(envelope):
    """
    Compute features from energy envelope.
    
    Args:
        envelope: Energy envelope (one value per frame)
        
    Returns:
        Dictionary of envelope features
    """
    envelope = np.array(envelope)  
    
    # Handle empty array
    if len(envelope) < 2:
        return {
            'mean_attack_rate': 0,
            'mean_decay_rate': 0,
            'attack_decay_ratio': 0,
            'envelope_modulation': 0,
            'attack_count': 0,
            'decay_count': 0
        }
    
    # Normalize envelope
    if np.max(envelope) > np.min(envelope):
        normalized = (envelope - np.min(envelope)) / (np.max(envelope) - np.min(envelope))
    else:
        normalized = np.zeros_like(envelope)
    
    # Compute attack and decay
    attack_rates = []
    decay_rates = []
    
    # Find regions of increasing energy (attacks) and decreasing energy (decays)
    in_attack = False
    in_decay = False
    attack_start = 0
    decay_start = 0
    
    for i in range(1, len(normalized)):
        # Check for attack (significant increase)
        if normalized[i] > normalized[i-1] + 0.05:
            if not in_attack:
                in_attack = True
                attack_start = i - 1
            in_decay = False
        # Check for decay (significant decrease)
        elif normalized[i] < normalized[i-1] - 0.05:
            if in_attack:
                # End of attack, calculate rate
                attack_duration = i - attack_start
                attack_magnitude = normalized[i-1] - normalized[attack_start]
                if attack_duration > 0:
                    attack_rates.append(attack_magnitude / attack_duration)
                in_attack = False
            
            if not in_decay:
                in_decay = True
                decay_start = i - 1
        # Check for end of decay
        elif in_decay and (normalized[i] >= normalized[i-1] or i == len(normalized) - 1):
            # End of decay, calculate rate
            decay_duration = i - decay_start
            decay_magnitude = normalized[decay_start] - normalized[i-1]
            if decay_duration > 0:
                decay_rates.append(decay_magnitude / decay_duration)
            in_decay = False
    
    # Compute summary features
    mean_attack_rate = np.mean(attack_rates) if attack_rates else 0
    mean_decay_rate = np.mean(decay_rates) if decay_rates else 0
    attack_decay_ratio = mean_attack_rate / mean_decay_rate if mean_decay_rate > 0 else 0
    
    # Compute envelope modulation
    modulation = np.std(np.diff(normalized)) if len(normalized) > 1 else 0
    
    return {
        'mean_attack_rate': mean_attack_rate,
        'mean_decay_rate': mean_decay_rate,
        'attack_decay_ratio': attack_decay_ratio,
        'envelope_modulation': modulation,
        'attack_count': len(attack_rates),
        'decay_count': len(decay_rates)
    }

def extract_complete_trajectory_features(utterance, sample_rate=16000):
    """
    Extract comprehensive trajectory-based features from a speech utterance.
    
    Args:
        utterance: Audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary of trajectory features
    """
    utterance = np.array(utterance)  
    
    # Handle empty utterance
    if len(utterance) == 0:
        return {'error': 'Empty utterance'}
    
    features = {}
    
    try:
        # Extract energy envelope
        energy_envelope = extract_energy_envelope(utterance, sample_rate)
        
        # Skip if energy envelope extraction failed
        if len(energy_envelope) == 0:
            return {'error': 'Failed to extract energy envelope'}
        
        # Extract F0 contour (pitch) with error handling
        try:
            f0, voiced_flag, _ = librosa.pyin(
                y=utterance,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                frame_length=2048,
                hop_length=512
            )
            # Replace NaN values with zeros for unvoiced frames
            f0 = np.nan_to_num(f0)
        except Exception as e:
            print(f"Error extracting pitch: {e}")
            f0 = np.zeros(len(energy_envelope))
        
        # 1. Linear trends
        features['energy_trend'] = compute_linear_trend(energy_envelope)
        features['f0_trend'] = compute_linear_trend(f0[f0 > 0]) if np.any(f0 > 0) else 0
        
        # 2. Dynamism features
        features['energy_dynamism'] = compute_feature_dynamism(energy_envelope)
        features['f0_dynamism'] = compute_feature_dynamism(f0[f0 > 0]) if np.any(f0 > 0) else 0
        
        # 3. Burst features
        energy_burst_features = compute_burst_features(energy_envelope)
        features.update({f'energy_{k}': v for k, v in energy_burst_features.items()})
        
        # 4. Peak analysis
        energy_peak_features = analyze_peaks(energy_envelope)
        features.update({f'energy_{k}': v for k, v in energy_peak_features.items()})
        
        # 5. Envelope features
        envelope_features = compute_envelope_features(energy_envelope)
        features.update(envelope_features)
        
        # 6. Combined joy-specific features
        # Joy often has quick attacks, multiple peaks, and rising contours
        features['joy_signature'] = (
            0.3 * features['energy_trend'] +
            0.2 * features.get('energy_burst_rate', 0) +
            0.2 * features['mean_attack_rate'] +
            0.15 * features.get('energy_peak_rate', 0) +
            0.15 * features['energy_dynamism']
        )
        
    except Exception as e:
        print(f"Error extracting trajectory features: {e}")
        return {'error': str(e)}
    
    return features
# import librosa
# import numpy as np
# import scipy.stats
# import scipy.signal

# def extract_joy_signature(utterance, sample_rate=16000):
#     """
#     Extract joy signature from a speech utterance.
    
#     Args:
#         utterance: Audio samples
#         sample_rate: Sample rate in Hz
        
#     Returns:
#         Joy signature score
#     """
#     utterance = np.array(utterance)
#     if len(utterance) == 0:
#         return 0.0
        
#     try:
#         # Extract energy envelope
#         energy = librosa.feature.rms(
#             y=utterance,
#             frame_length=int(0.025 * sample_rate),
#             hop_length=int(0.010 * sample_rate)
#         )[0]
        
#         if len(energy) == 0:
#             return 0.0
            
#         # Compute linear trend
#         n_frames = len(energy)
#         if n_frames < 2 or np.all(energy == energy[0]):
#             trend = 0.0
#         else:
#             time_indices = np.linspace(0, 1, n_frames)
#             slope, _, _, _, _ = scipy.stats.linregress(time_indices, energy)
#             feature_range = np.max(energy) - np.min(energy)
#             trend = slope / feature_range if feature_range > 0 else 0.0
            
#         # Compute dynamism
#         if len(energy) < 3:
#             dynamism = 0.0
#         else:
#             first_derivative = np.diff(energy)
#             second_derivative = np.diff(first_derivative)
#             feature_range = np.max(energy) - np.min(energy)
#             if feature_range > 0:
#                 dynamism = (0.5 * np.std(first_derivative) + 
#                           0.3 * np.mean(np.abs(first_derivative)) + 
#                           0.2 * np.mean(np.abs(second_derivative))) / feature_range
#             else:
#                 dynamism = 0.0
                
#         # Compute burst rate
#         normalized = (energy - np.mean(energy)) / np.std(energy) if np.std(energy) > 0 else np.zeros_like(energy)
#         above_threshold = normalized > 1.5
#         burst_count = 0
#         in_burst = False
        
#         for is_above in above_threshold:
#             if is_above and not in_burst:
#                 in_burst = True
#             elif not is_above and in_burst:
#                 burst_count += 1
#                 in_burst = False
#         if in_burst:
#             burst_count += 1
            
#         burst_rate = burst_count / n_frames if n_frames > 0 else 0
        
#         # Compute attack rate
#         normalized = (energy - np.min(energy)) / (np.max(energy) - np.min(energy)) if np.max(energy) > np.min(energy) else np.zeros_like(energy)
#         attack_rates = []
#         in_attack = False
#         attack_start = 0
        
#         for i in range(1, len(normalized)):
#             if normalized[i] > normalized[i-1] + 0.05:
#                 if not in_attack:
#                     in_attack = True
#                     attack_start = i - 1
#             elif in_attack:
#                 attack_duration = i - attack_start
#                 attack_magnitude = normalized[i-1] - normalized[attack_start]
#                 if attack_duration > 0:
#                     attack_rates.append(attack_magnitude / attack_duration)
#                 in_attack = False
                
#         mean_attack_rate = np.mean(attack_rates) if attack_rates else 0
        
#         # Compute peak rate
#         peaks, _ = scipy.signal.find_peaks(normalized, distance=5, prominence=0.1)
#         peak_rate = len(peaks) / n_frames if n_frames > 0 else 0
        
#         # Compute joy signature
#         joy_signature = (
#             0.3 * trend +
#             0.2 * burst_rate +
#             0.2 * mean_attack_rate +
#             0.15 * peak_rate +
#             0.15 * dynamism
#         )
        
#         return joy_signature
        
#     except Exception as e:
#         print(f"Error computing joy signature: {e}")
#         return 0.0