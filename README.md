Speech Emotion Intensity Detection
This repository implements a state-of-the-art approach for detecting emotion intensity in speech using advanced projection layers instead of traditional pooling methods. The implementation is based on deep theoretical foundations of speech emotion expression and leverages the wav2vec2 model as a feature extractor.

Features
Multi-head Projection Architecture: Captures temporal-spectral complexity of emotions in speech
RankNet Training: Uses pairwise learning for more robust emotion intensity estimation
Emotion-Specific Projections: Specialized pathways for different emotion categories
Cross-Modal Integration: Optional integration of facial features for improved accuracy
Comprehensive Evaluation: Multiple metrics for assessing model performance
Installation
bash

# Clone the repository

```bash
git clone https://github.com/yechanjung/pknu25_05.git
```

# Create a virtual environment (optional but recommended)

```bash
python -m venv env
source env/bin/activate # On Windows, use: env\Scripts\activate
```

# Install dependencies

```bash
pip install -r requirements.txt
```

## Directory Structure

```
speech-emotion-intensity/
├── emotion_intensity_model.py # Core model implementation
├── emotion_intensity_utils.py # Utility functions and evaluation scripts
├── main.py # Main script for running the system
├── requirements.txt # Dependencies
└── README.md
```

Usage
The system provides a command-line interface for different operations:

1. Preprocessing a Dataset
   ```bash
   python main.py preprocess --data_path /path/to/audio_data --annotation_file /path/to/annotations.csv --output_dir preprocessed_data
   ```

The annotation file should be a CSV with at least the following columns:

file_path: Path to audio file (relative to data_path)
emotion: Emotion category (anger, happiness, sadness, fear, disgust, surprise, neutral)
intensity: Emotion intensity (0.0-1.0) 2. Training a Model

```bash
python main.py train --data_path /path/to/audio_data --train_annotation_file preprocessed_data/train_annotations.csv --val_annotation_file preprocessed_data/test_annotations.csv --output_dir model
```

<!-- --batch_size BATCH_SIZE Batch size (default: 16)
--num_epochs NUM_EPOCHS Number of epochs (default: 30)
--learning_rate LEARNING_RATE
Learning rate (default: 0.0001)
--no_ranknet Disable RankNet loss
--no_mse Disable MSE loss
--mse_weight MSE_WEIGHT Weight for MSE loss (default: 0.5)
--rank_weight RANK_WEIGHT Weight for RankNet loss (default: 0.5)
--num_projection_heads NUM_PROJECTION_HEADS
Number of projection heads (default: 4)
--no_emotion_specific Disable emotion-specific projections
--no_freeze_extractor Do not freeze feature extractor
--enable_cross_modal Enable cross-modal integration
--facial_feature_path FACIAL_FEATURE_PATH
Path to facial features 3. Evaluating a Model
bash
python main.py evaluate --model_path model/best_model.pt --data_path /path/to/audio_data --annotation_file preprocessed_data/test_annotations.csv --output_dir evaluation 4. Predicting Emotion Intensity for a Single Audio File
bash
python main.py predict --model_path model/best_model.pt --audio_file /path/to/audio.wav --output_dir prediction
Optionally, you can specify a particular emotion:

bash
python main.py predict --model_path model/best_model.pt --audio_file /path/to/audio.wav --output_dir prediction --emotion anger 5. Comparing Pooling vs Projection Methods
bash
python main.py compare --data_path /path/to/audio_data --annotation_file /path/to/annotations.csv --output_dir comparison 6. Cross-Validation
bash
python main.py crossval --data_path /path/to/audio_data --annotation_file /path/to/annotations.csv --output_dir crossval --num_folds 5
Model Architecture
The emotion intensity detection model consists of the following components: -->

Feature Extraction: Wav2vec2 model is used to extract contextualized speech representations.
Projection Layer: Instead of simple pooling operations, we use specialized projection layers that better preserve the temporal-spectral complexity of emotion manifestation in speech.
Multi-Head Architecture: Multiple projection heads focus on different aspects of emotional expression.
Emotion-Specific Pathways: Separate projection pathways for different emotion categories acknowledge that emotions manifest intensity differently.
RankNet Training: Pairwise learning approach that improves robustness to annotation inconsistencies.
Cross-Modal Integration (optional): Integration of facial features for multimodal emotion intensity detection.
Theoretical Foundations
This implementation is based on several theoretical foundations:

Component Process Model (CPM): Emotions trigger physiological changes that directly affect speech production parameters.
Temporal-Spectral Complexity: Emotion intensity information is encoded across multiple time scales and frequency bands in speech.
Information Bottleneck Theory: Projection layers create compressed representations that preserve task-relevant information.
Ordinal Learning Theory: Emotion intensity is inherently ordinal rather than categorical or continuous.
Ensemble Theory: Multiple projection heads can specialize in different aspects of emotional speech.
Datasets
The system is compatible with common emotion speech datasets such as:

IEMOCAP (Interactive Emotional Dyadic Motion Capture Database)
MSP-IMPROV (Multimodal Spontaneous-Influence Emotional Database)
RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Cross-Modal Integration
For multimodal emotion intensity detection, the system supports integration of facial features. To use this feature:

Extract facial features from videos corresponding to audio files
Save the features as a numpy dictionary: {file_path: feature_vector}
Enable cross-modal integration during training/evaluation

```bash
python main.py train --enable_cross_modal --facial_feature_path facial_features.npy ...
```
