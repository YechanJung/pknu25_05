"""
Main script for Speech Emotion Intensity Detection using Projection Layers
This script provides a complete pipeline for training and evaluating the model

Usage:
    python main.py --train --data_path /path/to/audio_data --annotation_file /path/to/annotations.csv
    python main.py --evaluate --model_path /path/to/model.pt --data_path /path/to/test_data --annotation_file /path/to/test_annotations.csv
    python main.py --predict --model_path /path/to/model.pt --audio_file /path/to/audio.wav
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import Wav2Vec2Processor
import torchaudio
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import kendalltau, spearmanr

# Import our modules
from emotion_intensity_model import EmotionIntensityDetector, EmotionSpeechDataset, RankNetLoss
from emotion_intensity_utils import (
    create_fold_splits, 
    extract_wav2vec_features,
    visualize_emotion_intensity_space,
    analyze_projection_heads,
    evaluate_model_cross_validation,
    compare_pooling_vs_projection
)

# Define emotion categories
EMOTION_CATEGORIES = ["anger", "happiness", "sadness", "fear", "disgust", "surprise", "neutral"]


def preprocess_dataset(data_path, annotation_file, output_dir, test_size=0.2, seed=42):
    """
    Preprocess the dataset and split into train/validation sets.
    
    Args:
        data_path: Path to audio data
        annotation_file: Path to annotation file
        output_dir: Directory to save processed data
        test_size: Fraction of data to use for testing
        seed: Random seed for reproducibility
        
    Returns:
        Paths to train and test annotation files
    """
    print("Preprocessing dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load annotations
    annotations = pd.read_csv(annotation_file)
    
    # Check annotation format
    required_columns = ['file_path', 'emotion', 'intensity']
    if not all(col in annotations.columns for col in required_columns):
        raise ValueError(f"Annotation file must contain columns: {required_columns}")
    
    # Normalize emotion labels if needed
    if 'emotion' in annotations.columns:
        # Map emotion labels to standard categories
        emotion_mapping = {}
        for emotion in annotations['emotion'].unique():
            # Find closest match in EMOTION_CATEGORIES
            if emotion.lower() in [e.lower() for e in EMOTION_CATEGORIES]:
                emotion_mapping[emotion] = emotion
            else:
                # Map to closest category based on first letter
                for category in EMOTION_CATEGORIES:
                    if emotion.lower()[0] == category.lower()[0]:
                        emotion_mapping[emotion] = category
                        break
                else:
                    # If no match found, map to neutral
                    emotion_mapping[emotion] = 'neutral'
        
        # Apply emotion mapping
        annotations['emotion'] = annotations['emotion'].map(emotion_mapping)
    
    # Normalize intensity values if needed
    if 'intensity' in annotations.columns:
        # Check if intensity values are normalized
        if annotations['intensity'].max() > 1.0 or annotations['intensity'].min() < 0.0:
            # Normalize to [0, 1]
            min_intensity = annotations['intensity'].min()
            max_intensity = annotations['intensity'].max()
            annotations['intensity'] = (annotations['intensity'] - min_intensity) / (max_intensity - min_intensity)
    
    # Verify all audio files exist
    valid_files = []
    for i, row in annotations.iterrows():
        file_path = os.path.join(data_path, row['file_path'])
        if os.path.exists(file_path):
            valid_files.append(i)
        else:
            print(f"Warning: File not found: {file_path}")
    
    # Filter annotations to only include existing files
    annotations = annotations.iloc[valid_files].reset_index(drop=True)
    
    # Split into train and test sets
    train_annotations, test_annotations = train_test_split(
        annotations, 
        test_size=test_size, 
        stratify=annotations['emotion'],
        random_state=seed
    )
    
    # Reset indices
    train_annotations = train_annotations.reset_index(drop=True)
    test_annotations = test_annotations.reset_index(drop=True)
    
    # Save train and test annotations
    train_file = os.path.join(output_dir, "train_annotations.csv")
    test_file = os.path.join(output_dir, "test_annotations.csv")
    train_annotations.to_csv(train_file, index=False)
    test_annotations.to_csv(test_file, index=False)
    
    print(f"Saved {len(train_annotations)} training samples and {len(test_annotations)} test samples")
    
    return train_file, test_file


def train_model(
    data_path,
    train_annotation_file,
    val_annotation_file,
    output_dir,
    batch_size=16,
    num_epochs=30,
    learning_rate=1e-4,
    use_ranknet=True,
    use_mse=True,
    mse_weight=0.5,
    rank_weight=0.5,
    num_projection_heads=4,
    emotion_specific_projections=True,
    freeze_feature_extractor=True,
    enable_cross_modal=False,
    facial_feature_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the emotion intensity detection model.
    
    Args:
        data_path: Path to audio data
        train_annotation_file: Path to training annotation file
        val_annotation_file: Path to validation annotation file
        output_dir: Directory to save model and results
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        use_ranknet: Whether to use RankNet loss
        use_mse: Whether to use MSE loss
        mse_weight: Weight for MSE loss
        rank_weight: Weight for RankNet loss
        num_projection_heads: Number of projection heads
        emotion_specific_projections: Whether to use emotion-specific projections
        freeze_feature_extractor: Whether to freeze the feature extractor
        enable_cross_modal: Whether to enable cross-modal integration
        facial_feature_path: Path to facial features (optional)
        device: Device to run on
    
    Returns:
        Trained model
    """
    print("Training model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Create datasets
    train_dataset = EmotionSpeechDataset(
        data_path=data_path,
        annotation_file=train_annotation_file,
        processor=processor,
        create_pairs=use_ranknet,
        facial_feature_path=facial_feature_path
    )
    
    val_dataset = EmotionSpeechDataset(
        data_path=data_path,
        annotation_file=val_annotation_file,
        processor=processor,
        create_pairs=False,  # No pairs for validation
        facial_feature_path=facial_feature_path
    )
    
    # Initialize model
    facial_feature_dim = None
    if enable_cross_modal and facial_feature_path is not None:
        # Load facial features to get dimensions
        facial_features = np.load(facial_feature_path, allow_pickle=True).item()
        # Get a sample key
        sample_key = list(facial_features.keys())[0]
        # Get feature dimension
        facial_feature_dim = facial_features[sample_key].shape[0]
    
    model = EmotionIntensityDetector(
        pretrained_model_name="facebook/wav2vec2-base-960h",
        num_projection_heads=num_projection_heads,
        freeze_feature_extractor=freeze_feature_extractor,
        emotion_specific_projections=emotion_specific_projections,
        enable_cross_modal=enable_cross_modal,
        facial_feature_dim=facial_feature_dim
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'mae': [],
        'r2': [],
        'kendall_tau': [],
        'spearman_rho': []
    }
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            if train_dataset.create_pairs:
                # Process pairs for RankNet
                first_batch = {k: v.to(device) for k, v in batch['first_sample'].items() if k != 'facial_features'}
                second_batch = {k: v.to(device) for k, v in batch['second_sample'].items() if k != 'facial_features'}
                
                # Add facial features if available
                if 'facial_features' in batch['first_sample'] and batch['first_sample']['facial_features'] is not None:
                    first_batch['facial_features'] = batch['first_sample']['facial_features'].to(device)
                if 'facial_features' in batch['second_sample'] and batch['second_sample']['facial_features'] is not None:
                    second_batch['facial_features'] = batch['second_sample']['facial_features'].to(device)
                
                # Forward pass for both samples
                first_outputs = model(
                    input_values=first_batch['input_values'],
                    attention_mask=first_batch['attention_mask'],
                    emotion_labels=first_batch['emotion'],
                    facial_features=first_batch.get('facial_features')
                )
                
                second_outputs = model(
                    input_values=second_batch['input_values'],
                    attention_mask=second_batch['attention_mask'],
                    emotion_labels=second_batch['emotion'],
                    facial_features=second_batch.get('facial_features')
                )
                
                # Compute loss
                loss = model.compute_loss(
                    outputs=first_outputs,
                    target_intensities=first_batch['intensity'],
                    paired_outputs=second_outputs,
                    paired_intensities=second_batch['intensity'],
                    use_mse=use_mse,
                    use_ranknet=use_ranknet,
                    mse_weight=mse_weight,
                    rank_weight=rank_weight
                )
            else:
                # Process single samples
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'facial_features'}
                
                # Add facial features if available
                if 'facial_features' in batch and batch['facial_features'] is not None:
                    inputs['facial_features'] = batch['facial_features'].to(device)
                
                # Forward pass
                outputs = model(
                    input_values=inputs['input_values'],
                    attention_mask=inputs['attention_mask'],
                    emotion_labels=inputs['emotion'],
                    facial_features=inputs.get('facial_features')
                )
                
                # Compute loss (MSE only for single samples)
                loss = torch.nn.functional.mse_loss(outputs['intensity'], inputs['intensity'])
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Process single samples
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'facial_features'}
                
                # Add facial features if available
                if 'facial_features' in batch and batch['facial_features'] is not None:
                    inputs['facial_features'] = batch['facial_features'].to(device)
                
                # Forward pass
                outputs = model(
                    input_values=inputs['input_values'],
                    attention_mask=inputs['attention_mask'],
                    emotion_labels=inputs['emotion'],
                    facial_features=inputs.get('facial_features')
                )
                
                # Compute loss (MSE only for validation)
                loss = torch.nn.functional.mse_loss(outputs['intensity'], inputs['intensity'])
                
                # Accumulate loss
                val_loss += loss.item()
                
                # Collect predictions and ground truth
                all_predictions.extend(outputs['intensity'].cpu().numpy().flatten())
                all_targets.extend(inputs['intensity'].cpu().numpy().flatten())
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        r2 = r2_score(all_targets, all_predictions)
        kendall_tau, _ = kendalltau(all_targets, all_predictions)
        spearman_rho, _ = spearmanr(all_targets, all_predictions)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mae'].append(mae)
        history['r2'].append(r2)
        history['kendall_tau'].append(kendall_tau)
        history['spearman_rho'].append(spearman_rho)
        
        # Print training progress
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | Kendall's Tau: {kendall_tau:.4f} | Spearman's Rho: {spearman_rho:.4f}")
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Plot MAE
    plt.subplot(2, 2, 2)
    plt.plot(history['mae'])
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    plt.grid(alpha=0.3)
    
    # Plot R²
    plt.subplot(2, 2, 3)
    plt.plot(history['r2'])
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Score')
    plt.grid(alpha=0.3)
    
    # Plot Kendall's Tau and Spearman's Rho
    plt.subplot(2, 2, 4)
    plt.plot(history['kendall_tau'], label='Kendall\'s Tau')
    plt.plot(history['spearman_rho'], label='Spearman\'s Rho')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.title('Rank Correlation')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "training_history.png"), dpi=300)
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt")))
    
    # Generate visualizations
    try:
        # Visualize emotion intensity space
        visualize_emotion_intensity_space(
            model=model,
            val_dataset=val_dataset,
            device=device,
            output_path=os.path.join(output_dir, "emotion_intensity_space.png")
        )
        
        # Analyze projection heads
        analyze_projection_heads(
            model=model,
            val_dataset=val_dataset,
            device=device,
            num_heads=num_projection_heads,
            output_path=os.path.join(output_dir, "projection_heads_analysis.png")
        )
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Save training history
    pd.DataFrame(history).to_csv(os.path.join(output_dir, "training_history.csv"), index=False)
    
    # Save model configuration
    model_config = {
        'num_projection_heads': num_projection_heads,
        'emotion_specific_projections': emotion_specific_projections,
        'freeze_feature_extractor': freeze_feature_extractor,
        'enable_cross_modal': enable_cross_modal,
        'use_ranknet': use_ranknet,
        'use_mse': use_mse,
        'mse_weight': mse_weight,
        'rank_weight': rank_weight,
        'best_val_loss': best_val_loss,
        'final_metrics': {
            'mae': history['mae'][-1],
            'r2': history['r2'][-1],
            'kendall_tau': history['kendall_tau'][-1],
            'spearman_rho': history['spearman_rho'][-1]
        }
    }
    
    with open(os.path.join(output_dir, "model_config.json"), 'w') as f:
        import json
        json.dump(model_config, f, indent=4)
    
    return model


def evaluate_model(
    model_path,
    data_path,
    annotation_file,
    output_dir,
    batch_size=16,
    facial_feature_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Evaluate a trained emotion intensity detection model.
    
    Args:
        model_path: Path to trained model
        data_path: Path to audio data
        annotation_file: Path to annotation file
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        facial_feature_path: Path to facial features (optional)
        device: Device to run on
    
    Returns:
        Dictionary of evaluation metrics
    """
    print("Evaluating model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configuration
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "model_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            import json
            model_config = json.load(f)
    else:
        print("Model configuration not found. Using default configuration.")
        model_config = {
            'num_projection_heads': 4,
            'emotion_specific_projections': True,
            'freeze_feature_extractor': True,
            'enable_cross_modal': False
        }
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Create dataset
    test_dataset = EmotionSpeechDataset(
        data_path=data_path,
        annotation_file=annotation_file,
        processor=processor,
        create_pairs=False,
        facial_feature_path=facial_feature_path
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Initialize model
    facial_feature_dim = None
    if model_config['enable_cross_modal'] and facial_feature_path is not None:
        # Load facial features to get dimensions
        facial_features = np.load(facial_feature_path, allow_pickle=True).item()
        # Get a sample key
        sample_key = list(facial_features.keys())[0]
        # Get feature dimension
        facial_feature_dim = facial_features[sample_key].shape[0]
    
    model = EmotionIntensityDetector(
        pretrained_model_name="facebook/wav2vec2-base-960h",
        num_projection_heads=model_config['num_projection_heads'],
        freeze_feature_extractor=model_config['freeze_feature_extractor'],
        emotion_specific_projections=model_config['emotion_specific_projections'],
        enable_cross_modal=model_config['enable_cross_modal'],
        facial_feature_dim=facial_feature_dim
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate model
    all_predictions = []
    all_targets = []
    all_emotions = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Process samples
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'facial_features'}
            
            # Add facial features if available
            if 'facial_features' in batch and batch['facial_features'] is not None:
                inputs['facial_features'] = batch['facial_features'].to(device)
            
            # Forward pass
            outputs = model(
                input_values=inputs['input_values'],
                attention_mask=inputs['attention_mask'],
                emotion_labels=inputs['emotion'],
                facial_features=inputs.get('facial_features')
            )
            
            # Collect predictions and ground truth
            all_predictions.extend(outputs['intensity'].cpu().numpy().flatten())
            all_targets.extend(inputs['intensity'].cpu().numpy().flatten())
            all_emotions.extend(inputs['emotion'].cpu().numpy().flatten())
            all_embeddings.append(outputs['embeddings'].cpu().numpy())
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    kendall_tau, _ = kendalltau(all_targets, all_predictions)
    spearman_rho, _ = spearmanr(all_targets, all_predictions)
    
    # Calculate per-emotion metrics
    emotion_metrics = {}
    for emotion_idx in range(len(EMOTION_CATEGORIES)):
        # Filter by emotion
        emotion_mask = np.array(all_emotions) == emotion_idx
        
        # Skip if no samples for this emotion
        if not np.any(emotion_mask):
            continue
        
        # Get predictions and targets for this emotion
        emotion_predictions = np.array(all_predictions)[emotion_mask]
        emotion_targets = np.array(all_targets)[emotion_mask]
        
        # Calculate metrics
        emotion_mae = mean_absolute_error(emotion_targets, emotion_predictions)
        emotion_r2 = r2_score(emotion_targets, emotion_predictions)
        emotion_kendall, _ = kendalltau(emotion_targets, emotion_predictions)
        emotion_spearman, _ = spearmanr(emotion_targets, emotion_predictions)
        
        # Store metrics
        emotion_metrics[EMOTION_CATEGORIES[emotion_idx]] = {
            'mae': emotion_mae,
            'r2': emotion_r2,
            'kendall_tau': emotion_kendall,
            'spearman_rho': emotion_spearman,
            'samples': np.sum(emotion_mask)
        }
    
    # Print overall metrics
    print("\n=== Overall Metrics ===")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Kendall's Tau: {kendall_tau:.4f}")
    print(f"Spearman's Rho: {spearman_rho:.4f}")
    
    # Print per-emotion metrics
    print("\n=== Per-Emotion Metrics ===")
    for emotion, metrics in emotion_metrics.items():
        print(f"{emotion} ({metrics['samples']} samples):")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  Kendall's Tau: {metrics['kendall_tau']:.4f}")
        print(f"  Spearman's Rho: {metrics['spearman_rho']:.4f}")
    
    # Create scatter plot of predictions vs ground truth
    plt.figure(figsize=(10, 8))
    
    # Create color map for emotions
    emotion_colors = plt.cm.tab10(np.linspace(0, 1, len(EMOTION_CATEGORIES)))
    
    # Plot each emotion separately
    for emotion_idx in range(len(EMOTION_CATEGORIES)):
        # Filter by emotion
        emotion_mask = np.array(all_emotions) == emotion_idx
        
        # Skip if no samples for this emotion
        if not np.any(emotion_mask):
            continue
        
        # Get predictions and targets for this emotion
        emotion_predictions = np.array(all_predictions)[emotion_mask]
        emotion_targets = np.array(all_targets)[emotion_mask]
        
        # Create scatter plot
        plt.scatter(
            emotion_targets, 
            emotion_predictions, 
            alpha=0.7, 
            label=EMOTION_CATEGORIES[emotion_idx],
            color=emotion_colors[emotion_idx]
        )
    
    # Add diagonal line (perfect predictions)
    min_val = min(min(all_targets), min(all_predictions))
    max_val = max(max(all_targets), max(all_predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Add labels and title
    plt.xlabel('Ground Truth Intensity', fontsize=14)
    plt.ylabel('Predicted Intensity', fontsize=14)
    plt.title('Predicted vs Ground Truth Emotion Intensity', fontsize=16)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "prediction_scatter.png"), dpi=300)
    
    # Generate visualizations
    try:
        # Visualize emotion intensity space
        visualize_emotion_intensity_space(
            model=model,
            val_dataset=test_dataset,
            device=device,
            output_path=os.path.join(output_dir, "emotion_intensity_space.png")
        )
        
        # Analyze projection heads
        analyze_projection_heads(
            model=model,
            val_dataset=test_dataset,
            device=device,
            num_heads=model_config['num_projection_heads'],
            output_path=os.path.join(output_dir, "projection_heads_analysis.png")
        )
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    # Save metrics
    all_metrics = {
        'overall': {
            'mae': mae,
            'r2': r2,
            'kendall_tau': kendall_tau,
            'spearman_rho': spearman_rho
        },
        'per_emotion': emotion_metrics
    }
    
    with open(os.path.join(output_dir, "evaluation_metrics.json"), 'w') as f:
        import json
        json.dump(all_metrics, f, indent=4)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'ground_truth': all_targets,
        'prediction': all_predictions,
        'emotion': [EMOTION_CATEGORIES[e] for e in all_emotions]
    })
    predictions_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    
    return all_metrics


def predict_emotion_intensity(
    model_path,
    audio_file,
    output_dir=None,
    emotion=None,
    facial_feature_path=None,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Predict emotion intensity for a single audio file.
    
    Args:
        model_path: Path to trained model
        audio_file: Path to audio file
        output_dir: Directory to save results (optional)
        emotion: Emotion category (optional, if not provided will predict for all emotions)
        facial_feature_path: Path to facial features (optional)
        device: Device to run on
        
    Returns:
        Dictionary of predicted intensities for each emotion
    """
    print(f"Predicting emotion intensity for {audio_file}...")
    
    # Load model configuration
    model_dir = os.path.dirname(model_path)
    config_path = os.path.join(model_dir, "model_config.json")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            import json
            model_config = json.load(f)
    else:
        print("Model configuration not found. Using default configuration.")
        model_config = {
            'num_projection_heads': 4,
            'emotion_specific_projections': True,
            'freeze_feature_extractor': True,
            'enable_cross_modal': False
        }
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Load audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Convert to mono if necessary
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Process waveform
    inputs = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding="max_length",
        max_length=16000 * 5  # 5 seconds at 16kHz
    )
    
    # Initialize model
    facial_feature_dim = None
    if model_config['enable_cross_modal'] and facial_feature_path is not None:
        # Load facial features
        facial_features = np.load(facial_feature_path, allow_pickle=True).item()
        # Get feature dimension
        facial_feature_dim = list(facial_features.values())[0].shape[0]
    
    model = EmotionIntensityDetector(
        pretrained_model_name="facebook/wav2vec2-base-960h",
        num_projection_heads=model_config['num_projection_heads'],
        freeze_feature_extractor=model_config['freeze_feature_extractor'],
        emotion_specific_projections=model_config['emotion_specific_projections'],
        enable_cross_modal=model_config['enable_cross_modal'],
        facial_feature_dim=facial_feature_dim
    )
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Get facial features if available
    facial_features_tensor = None
    if model_config['enable_cross_modal'] and facial_feature_path is not None:
        # Check if audio file name is in facial features
        audio_filename = os.path.basename(audio_file)
        if audio_filename in facial_features:
            facial_features_tensor = torch.tensor(
                facial_features[audio_filename], 
                dtype=torch.float
            ).unsqueeze(0).to(device)
    
    # Predict for all emotions if none specified
    emotions_to_predict = [EMOTION_CATEGORIES.index(emotion)] if emotion else range(len(EMOTION_CATEGORIES))
    
    # Make predictions
    predictions = {}
    
    with torch.no_grad():
        for emotion_idx in emotions_to_predict:
            # Create emotion tensor
            emotion_tensor = torch.tensor([emotion_idx], dtype=torch.long).to(device)
            
            # Forward pass
            outputs = model(
                input_values=inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device),
                emotion_labels=emotion_tensor,
                facial_features=facial_features_tensor
            )
            
            # Store prediction
            emotion_name = EMOTION_CATEGORIES[emotion_idx]
            predictions[emotion_name] = outputs['intensity'].item()
    
    # Print predictions
    print("\nPredicted Emotion Intensities:")
    for emotion_name, intensity in predictions.items():
        print(f"{emotion_name}: {intensity:.4f}")
    
    # Save predictions if output directory provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        with open(os.path.join(output_dir, "prediction.json"), 'w') as f:
            import json
            json.dump(predictions, f, indent=4)
        
        # Create bar chart
        plt.figure(figsize=(10, 6))
        
        # Sort emotions by intensity
        sorted_emotions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        emotion_names = [e[0] for e in sorted_emotions]
        intensities = [e[1] for e in sorted_emotions]
        
        # Create bar chart
        bars = plt.bar(emotion_names, intensities, color=plt.cm.viridis(np.linspace(0, 1, len(intensities))))
        
        # Add intensity values as text
        for i, v in enumerate(intensities):
            plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
        
        # Add labels and title
        plt.xlabel('Emotion', fontsize=14)
        plt.ylabel('Intensity', fontsize=14)
        plt.title(f'Predicted Emotion Intensities for {os.path.basename(audio_file)}', fontsize=16)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add grid
        plt.grid(alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, "emotion_intensities.png"), dpi=300)
    
    return predictions


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Speech Emotion Intensity Detection")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser("preprocess", help="Preprocess dataset")
    preprocess_parser.add_argument("--data_path", type=str, required=True, help="Path to audio data")
    preprocess_parser.add_argument("--annotation_file", type=str, required=True, help="Path to annotation file")
    preprocess_parser.add_argument("--output_dir", type=str, default="preprocessed_data", help="Output directory")
    preprocess_parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data to use for testing")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data_path", type=str, required=True, help="Path to audio data")
    train_parser.add_argument("--train_annotation_file", type=str, required=True, help="Path to training annotation file")
    train_parser.add_argument("--val_annotation_file", type=str, required=True, help="Path to validation annotation file")
    train_parser.add_argument("--output_dir", type=str, default="model", help="Output directory")
    train_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--no_ranknet", action="store_true", help="Disable RankNet loss")
    train_parser.add_argument("--no_mse", action="store_true", help="Disable MSE loss")
    train_parser.add_argument("--mse_weight", type=float, default=0.5, help="Weight for MSE loss")
    train_parser.add_argument("--rank_weight", type=float, default=0.5, help="Weight for RankNet loss")
    train_parser.add_argument("--num_projection_heads", type=int, default=4, help="Number of projection heads")
    train_parser.add_argument("--no_emotion_specific", action="store_true", help="Disable emotion-specific projections")
    train_parser.add_argument("--no_freeze_extractor", action="store_true", help="Do not freeze feature extractor")
    train_parser.add_argument("--enable_cross_modal", action="store_true", help="Enable cross-modal integration")
    train_parser.add_argument("--facial_feature_path", type=str, help="Path to facial features")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate model")
    eval_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    eval_parser.add_argument("--data_path", type=str, required=True, help="Path to audio data")
    eval_parser.add_argument("--annotation_file", type=str, required=True, help="Path to annotation file")
    eval_parser.add_argument("--output_dir", type=str, default="evaluation", help="Output directory")
    eval_parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    eval_parser.add_argument("--facial_feature_path", type=str, help="Path to facial features")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict emotion intensity")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    predict_parser.add_argument("--audio_file", type=str, required=True, help="Path to audio file")
    predict_parser.add_argument("--output_dir", type=str, help="Output directory")
    predict_parser.add_argument("--emotion", type=str, choices=EMOTION_CATEGORIES, help="Emotion to predict")
    predict_parser.add_argument("--facial_feature_path", type=str, help="Path to facial features")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare pooling vs projection")
    compare_parser.add_argument("--data_path", type=str, required=True, help="Path to audio data")
    compare_parser.add_argument("--annotation_file", type=str, required=True, help="Path to annotation file")
    compare_parser.add_argument("--output_dir", type=str, default="comparison", help="Output directory")
    
    # Cross-validation command
    crossval_parser = subparsers.add_parser("crossval", help="Cross-validation")
    crossval_parser.add_argument("--data_path", type=str, required=True, help="Path to audio data")
    crossval_parser.add_argument("--annotation_file", type=str, required=True, help="Path to annotation file")
    crossval_parser.add_argument("--output_dir", type=str, default="crossval", help="Output directory")
    crossval_parser.add_argument("--num_folds", type=int, default=5, help="Number of folds")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "preprocess":
        preprocess_dataset(
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            output_dir=args.output_dir,
            test_size=args.test_size
        )
    
    elif args.command == "train":
        train_model(
            data_path=args.data_path,
            train_annotation_file=args.train_annotation_file,
            val_annotation_file=args.val_annotation_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            use_ranknet=not args.no_ranknet,
            use_mse=not args.no_mse,
            mse_weight=args.mse_weight,
            rank_weight=args.rank_weight,
            num_projection_heads=args.num_projection_heads,
            emotion_specific_projections=not args.no_emotion_specific,
            freeze_feature_extractor=not args.no_freeze_extractor,
            enable_cross_modal=args.enable_cross_modal,
            facial_feature_path=args.facial_feature_path
        )
    
    elif args.command == "evaluate":
        evaluate_model(
            model_path=args.model_path,
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            facial_feature_path=args.facial_feature_path
        )
    
    elif args.command == "predict":
        predict_emotion_intensity(
            model_path=args.model_path,
            audio_file=args.audio_file,
            output_dir=args.output_dir,
            emotion=args.emotion,
            facial_feature_path=args.facial_feature_path
        )
    
    elif args.command == "compare":
        compare_pooling_vs_projection(
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            output_dir=args.output_dir
        )
    
    elif args.command == "crossval":
        from emotion_intensity_model import EmotionIntensityDetector, EmotionSpeechDataset
        
        evaluate_model_cross_validation(
            model_class=EmotionIntensityDetector,
            dataset_class=EmotionSpeechDataset,
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            num_folds=args.num_folds,
            output_dir=args.output_dir
        )