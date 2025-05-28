"""
Speech Emotion Intensity Detection using Projection Layers
Based on theoretical foundations of emotion manifestation in speech

This implementation includes:
1. Custom projection layers that outperform pooling
2. Multi-head projection architecture for temporal-spectral analysis
3. RankNet implementation for pairwise learning of emotion intensity
4. Cross-modal integration capabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import torchaudio
import os
from sklearn.metrics import mean_absolute_error, accuracy_score
import pandas as pd
from scipy.stats import kendalltau, spearmanr

# Define emotion categories
EMOTION_CATEGORIES = ["anger", "happiness", "sadness", "fear", "disgust", "surprise", "neutral"]


class EmotionProjectionHead(nn.Module):
    """
    Projection head for a specific emotion category that captures the temporal-spectral complexity
    of emotion intensity manifestation in speech.
    
    Theoretical basis:
    - Micro-prosodic features through temporal convolutions
    - Segmental features through self-attention
    - Supra-segmental features through long-range context
    """
    def __init__(
        self, 
        input_dim: int = 768, 
        hidden_dim: int = 256, 
        output_dim: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        use_attention: bool = True
    ):
        super().__init__()
        
        # Multi-scale temporal convolutions for capturing features at different time scales
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim // 4, kernel_size=k, padding=k//2)
            for k in [1, 3, 5, 7]  # Different kernel sizes capture different temporal contexts
        ])
        
        # Multi-layer projection with residual connections
        self.projection_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim if i > 0 else hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            for i in range(num_layers)
        ])
        
        # Self-attention mechanism for focusing on emotionally salient regions
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Final projection to output dimension
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the projection head.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            attention_mask: Mask tensor of shape [batch_size, sequence_length]
            
        Returns:
            Projected tensor of shape [batch_size, output_dim]
        """
        # Convert to channel-first for convolutions [batch_size, input_dim, sequence_length]
        x_conv = x.transpose(1, 2)
        
        # Apply multi-scale temporal convolutions
        conv_outputs = []
        for conv in self.temporal_convs:
            conv_outputs.append(F.gelu(conv(x_conv)))
        
        # Concatenate outputs along feature dimension and convert back to sequence-first
        x = torch.cat(conv_outputs, dim=1).transpose(1, 2)
        
        # Apply multi-layer projection with residual connections
        for layer in self.projection_layers:
            residual = x
            x = layer(x) + residual
        
        if self.use_attention and attention_mask is not None:
            # Apply self-attention with provided mask
            # Convert attention_mask to key_padding_mask format (True for positions to mask)
            key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None
            residual = x
            x, _ = self.attention(
                query=x,
                key=x,
                value=x,
                key_padding_mask=key_padding_mask
            )
            x = self.attention_norm(x + residual)
        
        # Calculate weighted average using attention scores or simple mean if no attention
        if attention_mask is not None:
            # Create a normalized mask for weighted pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # Apply mask and compute weighted average
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            # Simple mean pooling as fallback
            x = x.mean(dim=1)
        
        # Final projection
        x = self.output_projection(x)
        
        return x


class MultiHeadEmotionProjection(nn.Module):
    """
    Multi-head projection network for emotion intensity detection.
    
    Theoretical basis:
    - Ensemble theory: Different heads can specialize in different aspects of emotional speech
    - Information bottleneck: Each head creates a compressed representation preserving different aspects
    """
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        output_dim: int = 64,
        num_heads: int = 4,
        emotion_specific: bool = True,
        num_emotions: int = 7  # Based on EMOTION_CATEGORIES
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.emotion_specific = emotion_specific
        self.num_emotions = num_emotions
        
        if emotion_specific:
            # Create separate projection heads for each emotion category
            self.projection_heads = nn.ModuleList([
                nn.ModuleList([
                    EmotionProjectionHead(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim
                    ) for _ in range(num_heads)
                ]) for _ in range(num_emotions)
            ])
        else:
            # Create shared projection heads for all emotions
            self.projection_heads = nn.ModuleList([
                EmotionProjectionHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim
                ) for _ in range(num_heads)
            ])
        
        # Integration layer to combine outputs from multiple heads
        integration_input_dim = output_dim * num_heads
        self.integration_layer = nn.Sequential(
            nn.Linear(integration_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        emotion_idx: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the multi-head projection.
        
        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_dim]
            emotion_idx: Tensor of emotion indices of shape [batch_size]
            attention_mask: Mask tensor of shape [batch_size, sequence_length]
            
        Returns:
            Projected tensor of shape [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        if self.emotion_specific and emotion_idx is not None:
            # Use emotion-specific projection heads
            head_outputs = []
            for i in range(batch_size):
                # Get the emotion index for this sample
                emo_idx = emotion_idx[i].item()
                # Apply each projection head for this emotion
                sample_outputs = []
                for head in self.projection_heads[emo_idx]:
                    # Extract this sample and its mask
                    sample_x = x[i:i+1]
                    sample_mask = attention_mask[i:i+1] if attention_mask is not None else None
                    # Apply projection
                    out = head(sample_x, sample_mask)
                    sample_outputs.append(out)
                # Concatenate outputs from all heads for this sample
                sample_concat = torch.cat(sample_outputs, dim=1)
                head_outputs.append(sample_concat)
            # Stack outputs from all samples
            outputs = torch.cat(head_outputs, dim=0)
        else:
            # Use shared projection heads
            head_outputs = []
            for head in self.projection_heads:
                out = head(x, attention_mask)
                head_outputs.append(out)
            # Concatenate outputs from all heads
            outputs = torch.cat(head_outputs, dim=1)
        
        # Integrate outputs from multiple heads
        integrated = self.integration_layer(outputs)
        
        return integrated


class RankNetLoss(nn.Module):
    """
    Implementation of RankNet loss for learning pairwise preferences of emotion intensity.
    
    Theoretical basis:
    - Ordinal Learning Theory: Emotion intensity is inherently ordinal
    - Preference Learning: Framing intensity estimation as learning preferences
    """
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def forward(
        self, 
        predicted_scores: torch.Tensor,  # Shape: [batch_size]
        paired_scores: torch.Tensor,     # Shape: [batch_size]
        intensity_diff: torch.Tensor     # Shape: [batch_size]
    ) -> torch.Tensor:
        """
        Compute RankNet loss for pairs of samples.
        
        Args:
            predicted_scores: Predicted intensity scores for the first samples in pairs
            paired_scores: Predicted intensity scores for the second samples in pairs
            intensity_diff: Target intensity differences (positive if first sample is more intense)
            
        Returns:
            RankNet loss value
        """
        # Calculate the probability that the first sample is more intense
        score_diff = predicted_scores - paired_scores
        pred_prob = torch.sigmoid(self.sigma * score_diff)
        
        # Convert intensity differences to target probabilities (0 or 1)
        target_prob = (intensity_diff > 0).float()
        
        # Calculate cross-entropy loss
        loss = F.binary_cross_entropy(pred_prob, target_prob)
        
        return loss


class EmotionIntensityDetector(nn.Module):
    """
    Complete model for speech emotion intensity detection combining wav2vec2 
    with multi-head projection and RankNet training.
    """
    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-base-960h",
        hidden_dim: int = 256,
        output_dim: int = 64,
        final_dim: int = 1,
        num_projection_heads: int = 4,
        freeze_feature_extractor: bool = True,
        emotion_specific_projections: bool = True,
        enable_cross_modal: bool = False,
        facial_feature_dim: Optional[int] = None
    ):
        super().__init__()
        
        # Load wav2vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        
        # Freeze feature extractor if specified
        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False
        
        # Get the dimension of wav2vec2 outputs
        wav2vec2_dim = self.wav2vec2.config.hidden_size
        
        # Multi-head projection layer
        self.projection = MultiHeadEmotionProjection(
            input_dim=wav2vec2_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_projection_heads,
            emotion_specific=emotion_specific_projections
        )
        
        # Cross-modal integration if enabled
        self.enable_cross_modal = enable_cross_modal
        if enable_cross_modal and facial_feature_dim is not None:
            # Cross-modal attention for integrating facial features
            self.cross_modal_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                dropout=0.3,
                batch_first=True
            )
            # Projection for facial features
            self.facial_projection = nn.Linear(facial_feature_dim, output_dim)
            
        # Final regression layer for intensity prediction
        self.intensity_regressor = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, final_dim)
        )
        
        # RankNet loss
        self.rank_loss = RankNetLoss()
        
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        facial_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the emotion intensity detector.
        
        Args:
            input_values: Audio input values of shape [batch_size, sequence_length]
            attention_mask: Attention mask of shape [batch_size, sequence_length]
            emotion_labels: Emotion category indices of shape [batch_size]
            facial_features: Optional facial features of shape [batch_size, facial_feature_dim]
            
        Returns:
            Dictionary containing:
                - 'intensity': Predicted intensity values of shape [batch_size, 1]
                - 'embeddings': Emotion embeddings of shape [batch_size, output_dim]
        """
        # Extract features using wav2vec2
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get the hidden states
        hidden_states = outputs.last_hidden_state  # [batch_size, sequence_length, wav2vec2_dim]
        
        # Apply multi-head projection
        projected = self.projection(
            hidden_states, 
            emotion_idx=emotion_labels,
            attention_mask=attention_mask
        )  # [batch_size, output_dim]
        
        # Cross-modal integration if enabled and facial features provided
        if self.enable_cross_modal and facial_features is not None:
            # Project facial features
            facial_projected = self.facial_projection(facial_features)  # [batch_size, output_dim]
            
            # Expand facial features for cross-attention
            facial_projected = facial_projected.unsqueeze(1)  # [batch_size, 1, output_dim]
            projected_expanded = projected.unsqueeze(1)  # [batch_size, 1, output_dim]
            
            # Apply cross-modal attention
            fused, _ = self.cross_modal_attention(
                query=projected_expanded,
                key=facial_projected,
                value=facial_projected
            )
            
            # Combine speech and facial information
            projected = projected + fused.squeeze(1)
        
        # Predict intensity
        intensity = self.intensity_regressor(projected)  # [batch_size, 1]
        
        return {
            'intensity': intensity,
            'embeddings': projected
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        target_intensities: torch.Tensor,
        paired_outputs: Optional[Dict[str, torch.Tensor]] = None,
        paired_intensities: Optional[torch.Tensor] = None,
        use_mse: bool = True,
        use_ranknet: bool = True,
        mse_weight: float = 0.5,
        rank_weight: float = 0.5
    ) -> torch.Tensor:
        """
        Compute combined loss for emotion intensity prediction.
        
        Args:
            outputs: Output dictionary from forward pass
            target_intensities: Target intensity values of shape [batch_size, 1]
            paired_outputs: Optional output dictionary for paired samples
            paired_intensities: Optional target intensities for paired samples
            use_mse: Whether to use MSE loss
            use_ranknet: Whether to use RankNet loss
            mse_weight: Weight for MSE loss
            rank_weight: Weight for RankNet loss
            
        Returns:
            Combined loss value
        """
        losses = []
        loss_weights = []
        
        # MSE loss for direct intensity regression
        if use_mse:
            mse_loss = F.mse_loss(outputs['intensity'], target_intensities)
            losses.append(mse_loss)
            loss_weights.append(mse_weight)
        
        # RankNet loss for pairwise learning
        if use_ranknet and paired_outputs is not None and paired_intensities is not None:
            # Calculate intensity differences
            intensity_diff = target_intensities - paired_intensities
            
            # Compute RankNet loss
            rank_loss = self.rank_loss(
                predicted_scores=outputs['intensity'].squeeze(),
                paired_scores=paired_outputs['intensity'].squeeze(),
                intensity_diff=intensity_diff.squeeze()
            )
            losses.append(rank_loss)
            loss_weights.append(rank_weight)
        
        # Combine losses
        total_loss = sum(w * l for w, l in zip(loss_weights, losses))
        
        return total_loss


class EmotionSpeechDataset(Dataset):
    """
    Dataset for emotion speech samples with intensity labels.
    Compatible with common datasets like IEMOCAP, MSP-IMPROV, and RAVDESS.
    """
    def __init__(
        self,
        data_path: str,
        annotation_file: str,
        processor: Wav2Vec2Processor,
        max_length: int = 16000 * 5,  # 5 seconds at 16kHz
        create_pairs: bool = True,
        emotion_map: Optional[Dict[str, int]] = None,
        facial_feature_path: Optional[str] = None
    ):
        self.data_path = data_path
        self.processor = processor
        self.max_length = max_length
        self.create_pairs = create_pairs
        
        # Load annotations
        self.annotations = pd.read_csv(annotation_file)
        
        # Set up emotion mapping
        if emotion_map is None:
            self.emotion_map = {emotion: i for i, emotion in enumerate(EMOTION_CATEGORIES)}
        else:
            self.emotion_map = emotion_map
        
        # Create sample pairs for RankNet training if enabled
        if create_pairs:
            self.create_sample_pairs()
        
        # Load facial features if provided
        self.facial_features = None
        if facial_feature_path is not None:
            self.facial_features = np.load(facial_feature_path, allow_pickle=True).item()
    
    def create_sample_pairs(self):
        """Create pairs of samples for RankNet training based on emotion intensity."""
        pairs = []
        
        # Group by emotion
        for emotion in self.annotations['emotion'].unique():
            # Get samples for this emotion
            emotion_samples = self.annotations[self.annotations['emotion'] == emotion]
            
            # Sort by intensity
            sorted_samples = emotion_samples.sort_values('intensity')
            
            # Create pairs with significant intensity differences
            for i in range(len(sorted_samples)):
                for j in range(i + 1, len(sorted_samples)):
                    # Only create pairs with significant intensity difference
                    intensity_diff = sorted_samples.iloc[j]['intensity'] - sorted_samples.iloc[i]['intensity']
                    if abs(intensity_diff) >= 0.2:  # Threshold for significant difference
                        pairs.append((
                            sorted_samples.iloc[i].name,  # Index of first sample
                            sorted_samples.iloc[j].name,  # Index of second sample
                            intensity_diff                # Intensity difference
                        ))
        
        self.pairs = pairs
    
    def __len__(self):
        return len(self.annotations) if not self.create_pairs else len(self.pairs)
    
    def __getitem__(self, idx):
        if self.create_pairs:
            # Get pair of samples
            first_idx, second_idx, intensity_diff = self.pairs[idx]
            
            # Get first sample
            first_sample = self._get_sample(first_idx)
            
            # Get second sample
            second_sample = self._get_sample(second_idx)
            
            return {
                'first_sample': first_sample,
                'second_sample': second_sample,
                'intensity_diff': torch.tensor(intensity_diff, dtype=torch.float)
            }
        else:
            # Get single sample
            return self._get_sample(idx)
    
    def _get_sample(self, idx):
        """Get a single sample from the dataset."""
        # Get sample metadata
        sample = self.annotations.iloc[idx]
        
        # Load audio file
        audio_path = os.path.join(self.data_path, sample['file_path'])
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Process waveform
        input_values = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length
        )
        
        # Get facial features if available
        facial_features = None
        if self.facial_features is not None and sample['file_path'] in self.facial_features:
            facial_features = torch.tensor(
                self.facial_features[sample['file_path']], 
                dtype=torch.float
            )
        
        return {
            'input_values': input_values.input_values.squeeze(),
            'attention_mask': input_values.attention_mask.squeeze(),
            'emotion': torch.tensor(self.emotion_map[sample['emotion']], dtype=torch.long),
            'intensity': torch.tensor(sample['intensity'], dtype=torch.float).unsqueeze(-1),
            'facial_features': facial_features
        }


def train_emotion_intensity_model(
    train_dataset,
    val_dataset,
    model,
    batch_size=16,
    num_epochs=30,
    learning_rate=1e-4,
    device='cuda',
    use_ranknet=True,
    use_mse=True,
    mse_weight=0.5,
    rank_weight=0.5
):
    """Train the emotion intensity detector model."""
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Create learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
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
                loss = F.mse_loss(outputs['intensity'], inputs['intensity'])
            
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
            for batch in val_loader:
                if val_dataset.create_pairs:
                    # Process pairs
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
                    
                    # Collect predictions for metrics
                    all_predictions.extend(first_outputs['intensity'].cpu().numpy().flatten())
                    all_predictions.extend(second_outputs['intensity'].cpu().numpy().flatten())
                    all_targets.extend(first_batch['intensity'].cpu().numpy().flatten())
                    all_targets.extend(second_batch['intensity'].cpu().numpy().flatten())
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
                    loss = F.mse_loss(outputs['intensity'], inputs['intensity'])
                    
                    # Collect predictions for metrics
                    all_predictions.extend(outputs['intensity'].cpu().numpy().flatten())
                    all_targets.extend(inputs['intensity'].cpu().numpy().flatten())
                
                # Accumulate loss
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Calculate evaluation metrics
        mae = mean_absolute_error(all_targets, all_predictions)
        
        # Calculate ranking correlation
        kendall_tau, _ = kendalltau(all_targets, all_predictions)
        spearman_rho, _ = spearmanr(all_targets, all_predictions)
        
        # Print training progress
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | MAE: {mae:.4f} | Kendall's Tau: {kendall_tau:.4f} | Spearman's Rho: {spearman_rho:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_emotion_intensity_model.pt')
    
    # Load the best model for evaluation
    model.load_state_dict(torch.load('best_emotion_intensity_model.pt'))
    
    return model