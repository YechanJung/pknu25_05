"""
Utility functions and evaluation scripts for speech emotion intensity detection
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import kendalltau, spearmanr
from transformers import Wav2Vec2Processor
import torchaudio
from tqdm import tqdm
import argparse


def create_fold_splits(annotation_file, num_folds=5, seed=42):
    """
    Create stratified cross-validation folds for model evaluation.
    
    Args:
        annotation_file: Path to annotation CSV file
        num_folds: Number of folds to create
        seed: Random seed for reproducibility
        
    Returns:
        List of dictionaries containing train and test indices for each fold
    """
    # Load annotations
    annotations = pd.read_csv(annotation_file)
    
    # Set random seed
    np.random.seed(seed)
    
    # Stratify by emotion and intensity
    # Create intensity bins for stratification
    annotations['intensity_bin'] = pd.qcut(annotations['intensity'], q=5, labels=False)
    
    # Create stratification groups
    annotations['strat_group'] = annotations['emotion'] + '_' + annotations['intensity_bin'].astype(str)
    
    # Get unique groups
    groups = annotations['strat_group'].unique()
    
    # Create folds
    folds = []
    for fold in range(num_folds):
        test_indices = []
        
        # For each group, select a portion for the test set
        for group in groups:
            group_indices = annotations[annotations['strat_group'] == group].index.tolist()
            
            # Shuffle indices
            np.random.shuffle(group_indices)
            
            # Calculate split point
            split_point = len(group_indices) // num_folds
            
            # Select test indices for this fold
            start_idx = fold * split_point
            end_idx = (fold + 1) * split_point if fold < num_folds - 1 else len(group_indices)
            fold_test_indices = group_indices[start_idx:end_idx]
            
            # Add to test indices
            test_indices.extend(fold_test_indices)
        
        # Create train indices (all indices not in test)
        all_indices = set(annotations.index.tolist())
        train_indices = list(all_indices - set(test_indices))
        
        # Add fold split
        folds.append({
            'train': train_indices,
            'test': test_indices
        })
    
    return folds


def extract_wav2vec_features(
    data_path,
    annotation_file,
    output_path,
    processor_name="facebook/wav2vec2-base-960h",
    max_duration=5.0  # in seconds
):
    """
    Pre-extract wav2vec2 features for the dataset to speed up training.
    
    Args:
        data_path: Path to audio files
        annotation_file: Path to annotation CSV file
        output_path: Path to save extracted features
        processor_name: Name of wav2vec2 processor
        max_duration: Maximum audio duration in seconds
    """
    # Load annotations
    annotations = pd.read_csv(annotation_file)
    
    # Load wav2vec2 processor
    processor = Wav2Vec2Processor.from_pretrained(processor_name)
    
    # Load wav2vec2 model
    model = Wav2Vec2Model.from_pretrained(processor_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Process each audio file
    features = {}
    for i, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Extracting features"):
        try:
            # Load audio file
            audio_path = os.path.join(data_path, row['file_path'])
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000
            
            # Trim or pad to max_duration
            max_samples = int(max_duration * sample_rate)
            if waveform.shape[1] > max_samples:
                # Trim
                waveform = waveform[:, :max_samples]
            else:
                # Pad
                padding = max_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            # Process audio
            inputs = processor(
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).to(device)
            
            # Extract features
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Store features
            features[row['file_path']] = {
                'hidden_states': outputs.last_hidden_state.cpu().numpy(),
                'attention_mask': inputs.attention_mask.cpu().numpy()
            }
            
        except Exception as e:
            print(f"Error processing {row['file_path']}: {e}")
    
    # Save features
    np.save(os.path.join(output_path, "wav2vec_features.npy"), features)
    
    return features


def visualize_emotion_intensity_space(model, val_dataset, device='cuda', output_path=None):
    """
    Visualize the learned emotion intensity space using t-SNE.
    
    Args:
        model: Trained emotion intensity model
        val_dataset: Validation dataset
        device: Device to run the model on
        output_path: Path to save the visualization
    """
    from sklearn.manifold import TSNE
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect embeddings and metadata
    all_embeddings = []
    all_emotions = []
    all_intensities = []
    
    with torch.no_grad():
        for i in tqdm(range(len(val_dataset)), desc="Collecting embeddings"):
            # Get sample
            sample = val_dataset[i]
            
            # Skip if this is a pair dataset
            if isinstance(sample, dict) and 'first_sample' in sample:
                sample = sample['first_sample']
                
            # Move tensors to device
            input_values = sample['input_values'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            emotion = sample['emotion'].unsqueeze(0).to(device)
            
            # Get facial features if available
            facial_features = None
            if 'facial_features' in sample and sample['facial_features'] is not None:
                facial_features = sample['facial_features'].unsqueeze(0).to(device)
            
            # Forward pass
            outputs = model(
                input_values=input_values,
                attention_mask=attention_mask,
                emotion_labels=emotion,
                facial_features=facial_features
            )
            
            # Collect embedding
            all_embeddings.append(outputs['embeddings'].cpu().numpy())
            all_emotions.append(sample['emotion'].item())
            all_intensities.append(sample['intensity'].item())
    
    # Convert to numpy arrays
    embeddings = np.vstack(all_embeddings)
    emotions = np.array(all_emotions)
    intensities = np.array(all_intensities)
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Create color map for emotions
    emotion_categories = val_dataset.emotion_map
    emotion_names = [k for k, v in sorted(emotion_categories.items(), key=lambda x: x[1])]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(emotion_names)))
    
    # Plot each emotion as a separate scatter plot
    for i, emotion_name in enumerate(emotion_names):
        emotion_id = emotion_categories[emotion_name]
        mask = emotions == emotion_id
        
        # Skip if no samples for this emotion
        if not np.any(mask):
            continue
        
        # Get data for this emotion
        x = embeddings_2d[mask, 0]
        y = embeddings_2d[mask, 1]
        intensity = intensities[mask]
        
        # Create scatter plot
        scatter = plt.scatter(x, y, c=intensity, cmap='viridis', 
                             label=emotion_name, alpha=0.7, edgecolors='w', 
                             linewidth=0.5, s=100)
    
    # Add colorbar for intensity
    cbar = plt.colorbar()
    cbar.set_label('Emotion Intensity', fontsize=12)
    
    # Add legend for emotions
    plt.legend(fontsize=12)
    
    # Add labels and title
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)
    plt.title('Emotion Intensity Embedding Space', fontsize=16)
    
    # Add grid
    plt.grid(alpha=0.3)
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show figure
    plt.show()


def analyze_projection_heads(model, val_dataset, device='cuda', num_heads=4, output_path=None):
    """
    Analyze what each projection head is learning.
    
    Args:
        model: Trained emotion intensity model
        val_dataset: Validation dataset
        device: Device to run the model on
        num_heads: Number of projection heads to analyze
        output_path: Path to save the analysis
    """
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of samples
    samples = [val_dataset[i] for i in range(min(32, len(val_dataset)))]
    
    # Skip if this is a pair dataset
    if isinstance(samples[0], dict) and 'first_sample' in samples[0]:
        samples = [s['first_sample'] for s in samples]
    
    # Process samples
    results = []
    
    with torch.no_grad():
        for sample in samples:
            # Move tensors to device
            input_values = sample['input_values'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            emotion = sample['emotion'].unsqueeze(0).to(device)
            
            # Get facial features if available
            facial_features = None
            if 'facial_features' in sample and sample['facial_features'] is not None:
                facial_features = sample['facial_features'].unsqueeze(0).to(device)
            
            # Forward pass through wav2vec2
            wav2vec_outputs = model.wav2vec2(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get hidden states
            hidden_states = wav2vec_outputs.last_hidden_state
            
            # Apply each projection head
            head_outputs = []
            emotion_idx = emotion.item()
            
            if model.projection.emotion_specific:
                # Use emotion-specific projection heads
                for head_idx in range(num_heads):
                    head = model.projection.projection_heads[emotion_idx][head_idx]
                    head_output = head(hidden_states, attention_mask)
                    head_outputs.append(head_output.cpu().numpy())
            else:
                # Use shared projection heads
                for head_idx in range(num_heads):
                    head = model.projection.projection_heads[head_idx]
                    head_output = head(hidden_states, attention_mask)
                    head_outputs.append(head_output.cpu().numpy())
            
            # Collect results
            results.append({
                'emotion': emotion.item(),
                'intensity': sample['intensity'].item(),
                'head_outputs': head_outputs
            })
    
    # Create correlation matrix between head outputs and intensity
    correlations = np.zeros((num_heads,))
    
    for head_idx in range(num_heads):
        head_values = np.array([r['head_outputs'][head_idx].mean() for r in results])
        intensities = np.array([r['intensity'] for r in results])
        
        # Calculate correlation
        correlation, _ = spearmanr(head_values, intensities)
        correlations[head_idx] = correlation
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_heads), correlations)
    plt.xlabel('Projection Head Index', fontsize=12)
    plt.ylabel('Correlation with Intensity', fontsize=12)
    plt.title('Projection Head Correlation with Emotion Intensity', fontsize=14)
    plt.grid(alpha=0.3)
    plt.xticks(range(num_heads))
    
    # Add correlation values as text
    for i, corr in enumerate(correlations):
        plt.text(i, corr + 0.02, f'{corr:.2f}', ha='center', fontsize=10)
    
    # Save figure if output path provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show figure
    plt.show()
    
    return correlations


def evaluate_model_cross_validation(
    model_class,
    dataset_class,
    data_path,
    annotation_file,
    num_folds=5,
    batch_size=16,
    num_epochs=30,
    learning_rate=1e-4,
    device='cuda',
    output_dir='results',
    seed=42
):
    """
    Evaluate the model using cross-validation.
    
    Args:
        model_class: Model class to evaluate
        dataset_class: Dataset class to use
        data_path: Path to audio data
        annotation_file: Path to annotation file
        num_folds: Number of cross-validation folds
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to run on
        output_dir: Directory to save results
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    # Create fold splits
    folds = create_fold_splits(annotation_file, num_folds=num_folds, seed=seed)
    
    # Load annotations
    annotations = pd.read_csv(annotation_file)
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Initialize results
    all_metrics = []
    
    # Train and evaluate on each fold
    for fold_idx, fold in enumerate(folds):
        print(f"\n=== Fold {fold_idx+1}/{num_folds} ===")
        
        # Create train and test datasets
        train_annotations = annotations.iloc[fold['train']].reset_index(drop=True)
        test_annotations = annotations.iloc[fold['test']].reset_index(drop=True)
        
        # Save fold annotations
        train_annotations.to_csv(os.path.join(output_dir, f"fold_{fold_idx+1}_train.csv"), index=False)
        test_annotations.to_csv(os.path.join(output_dir, f"fold_{fold_idx+1}_test.csv"), index=False)
        
        # Create temporary annotation files for the dataset
        train_annotation_file = os.path.join(output_dir, f"fold_{fold_idx+1}_train.csv")
        test_annotation_file = os.path.join(output_dir, f"fold_{fold_idx+1}_test.csv")
        
        # Create datasets
        train_dataset = dataset_class(
            data_path=data_path,
            annotation_file=train_annotation_file,
            processor=processor,
            create_pairs=True
        )
        
        test_dataset = dataset_class(
            data_path=data_path,
            annotation_file=test_annotation_file,
            processor=processor,
            create_pairs=False  # No pairs for evaluation
        )
        
        # Initialize model
        model = model_class(
            pretrained_model_name="facebook/wav2vec2-base-960h",
            freeze_feature_extractor=True,
            emotion_specific_projections=True
        )
        
        # Train model
        from train_emotion_intensity_model import train_emotion_intensity_model
        
        trained_model = train_emotion_intensity_model(
            train_dataset=train_dataset,
            val_dataset=test_dataset,
            model=model,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            use_ranknet=True,
            use_mse=True
        )
        
        # Evaluate model
        trained_model.eval()
        predictions = []
        ground_truth = []
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        with torch.no_grad():
            for batch in test_loader:
                # Move tensors to device
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'facial_features'}
                
                # Add facial features if available
                if 'facial_features' in batch and batch['facial_features'] is not None:
                    inputs['facial_features'] = batch['facial_features'].to(device)
                
                # Forward pass
                outputs = trained_model(
                    input_values=inputs['input_values'],
                    attention_mask=inputs['attention_mask'],
                    emotion_labels=inputs['emotion'],
                    facial_features=inputs.get('facial_features')
                )
                
                # Collect predictions and ground truth
                predictions.extend(outputs['intensity'].cpu().numpy().flatten())
                ground_truth.extend(inputs['intensity'].cpu().numpy().flatten())
        
        # Calculate metrics
        mae = mean_absolute_error(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)
        kendall, _ = kendalltau(ground_truth, predictions)
        spearman, _ = spearmanr(ground_truth, predictions)
        
        metrics = {
            'fold': fold_idx + 1,
            'mae': mae,
            'r2': r2,
            'kendall_tau': kendall,
            'spearman_rho': spearman
        }
        
        # Save metrics
        all_metrics.append(metrics)
        
        # Print metrics
        print(f"Fold {fold_idx+1} Metrics:")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Kendall's Tau: {kendall:.4f}")
        print(f"  Spearman's Rho: {spearman:.4f}")
        
        # Save model
        torch.save(trained_model.state_dict(), os.path.join(output_dir, f"model_fold_{fold_idx+1}.pt"))
        
        # Generate visualizations
        try:
            # Visualize emotion intensity space
            visualize_emotion_intensity_space(
                model=trained_model,
                val_dataset=test_dataset,
                device=device,
                output_path=os.path.join(output_dir, f"emotion_space_fold_{fold_idx+1}.png")
            )
            
            # Analyze projection heads
            analyze_projection_heads(
                model=trained_model,
                val_dataset=test_dataset,
                device=device,
                output_path=os.path.join(output_dir, f"projection_heads_fold_{fold_idx+1}.png")
            )
        except Exception as e:
            print(f"Error generating visualizations: {e}")
    
    # Calculate average metrics
    avg_metrics = {
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'r2': np.mean([m['r2'] for m in all_metrics]),
        'kendall_tau': np.mean([m['kendall_tau'] for m in all_metrics]),
        'spearman_rho': np.mean([m['spearman_rho'] for m in all_metrics])
    }
    
    # Print average metrics
    print("\n=== Average Metrics ===")
    print(f"  MAE: {avg_metrics['mae']:.4f}")
    print(f"  R²: {avg_metrics['r2']:.4f}")
    print(f"  Kendall's Tau: {avg_metrics['kendall_tau']:.4f}")
    print(f"  Spearman's Rho: {avg_metrics['spearman_rho']:.4f}")
    
    # Save all metrics
    all_metrics.append({'fold': 'average', **avg_metrics})
    pd.DataFrame(all_metrics).to_csv(os.path.join(output_dir, "evaluation_metrics.csv"), index=False)
    
    return avg_metrics


def compare_pooling_vs_projection(
    data_path,
    annotation_file,
    output_dir='comparison_results',
    batch_size=16,
    num_epochs=15,
    device='cuda'
):
    """
    Compare pooling vs projection methods for emotion intensity detection.
    
    Args:
        data_path: Path to audio data
        annotation_file: Path to annotation file
        output_dir: Directory to save results
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        device: Device to run on
    """
    import torch.nn as nn
    from transformers import Wav2Vec2Model, Wav2Vec2Processor
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load annotations
    annotations = pd.read_csv(annotation_file)
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    # Define baseline pooling model
    class PoolingEmotionIntensityModel(nn.Module):
        def __init__(
            self,
            pretrained_model_name="facebook/wav2vec2-base-960h",
            pooling_type='mean',
            hidden_dim=256
        ):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
            self.pooling_type = pooling_type
            
            # Get dimension of wav2vec2 outputs
            wav2vec2_dim = self.wav2vec2.config.hidden_size
            
            # Projection layers
            self.projection = nn.Sequential(
                nn.Linear(wav2vec2_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(
            self,
            input_values,
            attention_mask=None,
            emotion_labels=None,
            facial_features=None
        ):
            # Extract features using wav2vec2
            outputs = self.wav2vec2(
                input_values=input_values,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get hidden states
            hidden_states = outputs.last_hidden_state  # [batch_size, sequence_length, wav2vec2_dim]
            
            # Apply pooling
            if self.pooling_type == 'mean':
                # Mean pooling
                if attention_mask is not None:
                    # Create a mask for padding
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    # Apply mask and compute mean
                    pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
                else:
                    pooled = hidden_states.mean(dim=1)
            
            elif self.pooling_type == 'max':
                # Max pooling
                if attention_mask is not None:
                    # Create a mask for padding (set padding to large negative value)
                    mask_expanded = attention_mask.unsqueeze(-1).float()
                    masked_hidden = hidden_states * mask_expanded + (1 - mask_expanded) * -1e10
                    pooled = masked_hidden.max(dim=1)[0]
                else:
                    pooled = hidden_states.max(dim=1)[0]
            
            elif self.pooling_type == 'attention':
                # Attention pooling
                # Create attention scores
                attention_scores = torch.matmul(
                    hidden_states, 
                    hidden_states.transpose(-1, -2)
                ) / (hidden_states.size(-1) ** 0.5)
                
                # Apply mask if provided
                if attention_mask is not None:
                    # Create 2D mask
                    mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
                    # Apply mask (set padding to large negative value)
                    attention_scores = attention_scores * mask_2d + (1 - mask_2d) * -1e10
                
                # Apply softmax to get attention weights
                attention_weights = torch.softmax(attention_scores, dim=-1)
                
                # Apply attention to hidden states
                context = torch.matmul(attention_weights, hidden_states)
                
                # Sum over sequence dimension
                pooled = context.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) if attention_mask is not None else 1)
            
            else:
                raise ValueError(f"Unknown pooling type: {self.pooling_type}")
            
            # Project to intensity score
            intensity = self.projection(pooled)
            
            return {
                'intensity': intensity,
                'embeddings': pooled
            }
    
    # Create datasets
    from sklearn.model_selection import train_test_split
    
    train_annotations, test_annotations = train_test_split(
        annotations, 
        test_size=0.2, 
        stratify=annotations['emotion'],
        random_state=42
    )
    
    # Reset indices
    train_annotations = train_annotations.reset_index(drop=True)
    test_annotations = test_annotations.reset_index(drop=True)
    
    # Save annotations
    train_annotation_file = os.path.join(output_dir, "train_annotations.csv")
    test_annotation_file = os.path.join(output_dir, "test_annotations.csv")
    train_annotations.to_csv(train_annotation_file, index=False)
    test_annotations.to_csv(test_annotation_file, index=False)
    
    # Create dataset class
    from your_module_name import EmotionSpeechDataset
    
    # Create datasets
    train_dataset = EmotionSpeechDataset(
        data_path=data_path,
        annotation_file=train_annotation_file,
        processor=processor,
        create_pairs=True
    )
    
    test_dataset = EmotionSpeechDataset(
        data_path=data_path,
        annotation_file=test_annotation_file,
        processor=processor,
        create_pairs=False
    )
    
    # Define models to compare
    models = {
        'mean_pooling': PoolingEmotionIntensityModel(pooling_type='mean'),
        'max_pooling': PoolingEmotionIntensityModel(pooling_type='max'),
        'attention_pooling': PoolingEmotionIntensityModel(pooling_type='attention'),
        'projection': EmotionIntensityDetector(
            emotion_specific_projections=True,
            num_projection_heads=4
        )
    }
    
    # Evaluate each model
    results = {}
    
    for model_name, model in models.items():
        print(f"\n=== Evaluating {model_name} ===")
        
        # Move model to device
        model = model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                
                if train_dataset.create_pairs:
                    # Process pairs
                    first_batch = {k: v.to(device) for k, v in batch['first_sample'].items() if k != 'facial_features'}
                    second_batch = {k: v.to(device) for k, v in batch['second_sample'].items() if k != 'facial_features'}
                    
                    # Forward pass for both samples
                    first_outputs = model(
                        input_values=first_batch['input_values'],
                        attention_mask=first_batch['attention_mask'],
                        emotion_labels=first_batch['emotion'] if 'emotion' in first_batch else None
                    )
                    
                    second_outputs = model(
                        input_values=second_batch['input_values'],
                        attention_mask=second_batch['attention_mask'],
                        emotion_labels=second_batch['emotion'] if 'emotion' in second_batch else None
                    )
                    
                    # Calculate intensity differences
                    intensity_diff = batch['intensity_diff'].to(device)
                    
                    # Convert intensity differences to target probabilities
                    target_prob = (intensity_diff > 0).float().to(device)
                    
                    # Calculate the probability that the first sample is more intense
                    score_diff = first_outputs['intensity'].squeeze() - second_outputs['intensity'].squeeze()
                    pred_prob = torch.sigmoid(score_diff)
                    
                    # Calculate loss
                    loss = F.binary_cross_entropy(pred_prob, target_prob)
                else:
                    # Process single samples
                    inputs = {k: v.to(device) for k, v in batch.items() if k != 'facial_features'}
                    
                    # Forward pass
                    outputs = model(
                        input_values=inputs['input_values'],
                        attention_mask=inputs['attention_mask'],
                        emotion_labels=inputs['emotion'] if 'emotion' in inputs else None
                    )
                    
                    # Compute loss
                    loss = F.mse_loss(outputs['intensity'], inputs['intensity'])
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
                
                # Accumulate loss
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}")
        
        # Evaluation
        model.eval()
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Process single samples
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'facial_features'}
                
                # Forward pass
                outputs = model(
                    input_values=inputs['input_values'],
                    attention_mask=inputs['attention_mask'],
                    emotion_labels=inputs['emotion'] if 'emotion' in inputs else None
                )
                
                # Collect predictions and ground truth
                predictions.extend(outputs['intensity'].cpu().numpy().flatten())
                ground_truth.extend(inputs['intensity'].cpu().numpy().flatten())
        
        # Calculate metrics
        mae = mean_absolute_error(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)
        kendall, _ = kendalltau(ground_truth, predictions)
        spearman, _ = spearmanr(ground_truth, predictions)
        
        # Store results
        results[model_name] = {
            'mae': mae,
            'r2': r2,
            'kendall_tau': kendall,
            'spearman_rho': spearman
        }
        
        # Print metrics
        print(f"Model: {model_name}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Kendall's Tau: {kendall:.4f}")
        print(f"  Spearman's Rho: {spearman:.4f}")
        
        # Save model
        torch.save(model.state_dict(), os.path.join(output_dir, f"{model_name}_model.pt"))
    
    # Create comparison plot
    plt.figure(figsize=(14, 10))
    
    # Plot MAE (lower is better)
    plt.subplot(2, 2, 1)
    mae_values = [results[model]['mae'] for model in models.keys()]
    plt.bar(models.keys(), mae_values)
    plt.title('Mean Absolute Error (lower is better)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # Plot R² (higher is better)
    plt.subplot(2, 2, 2)
    r2_values = [results[model]['r2'] for model in models.keys()]
    plt.bar(models.keys(), r2_values)
    plt.title('R² Score (higher is better)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # Plot Kendall's Tau (higher is better)
    plt.subplot(2, 2, 3)
    kendall_values = [results[model]['kendall_tau'] for model in models.keys()]
    plt.bar(models.keys(), kendall_values)
    plt.title('Kendall\'s Tau (higher is better)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # Plot Spearman's Rho (higher is better)
    plt.subplot(2, 2, 4)
    spearman_values = [results[model]['spearman_rho'] for model in models.keys()]
    plt.bar(models.keys(), spearman_values)
    plt.title('Spearman\'s Rho (higher is better)', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, "model_comparison.png"), dpi=300)
    
    # Show figure
    plt.show()
    
    # Save results to CSV
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv(os.path.join(output_dir, "comparison_results.csv"))
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Emotion Intensity Utilities")
    parser.add_argument("--extract_features", action="store_true", help="Extract wav2vec features")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model with cross-validation")
    parser.add_argument("--compare", action="store_true", help="Compare pooling vs projection")
    parser.add_argument("--data_path", type=str, help="Path to audio data")
    parser.add_argument("--annotation_file", type=str, help="Path to annotation file")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()
    
    if args.extract_features:
        print("Extracting wav2vec features...")
        extract_wav2vec_features(
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            output_path=args.output_dir
        )
    
    if args.evaluate:
        print("Evaluating model with cross-validation...")
        from your_module_name import EmotionIntensityDetector, EmotionSpeechDataset
        
        evaluate_model_cross_validation(
            model_class=EmotionIntensityDetector,
            dataset_class=EmotionSpeechDataset,
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            output_dir=args.output_dir
        )
    
    if args.compare:
        print("Comparing pooling vs projection...")
        compare_pooling_vs_projection(
            data_path=args.data_path,
            annotation_file=args.annotation_file,
            output_dir=args.output_dir
        )