"""
Contrastive Learning Model for Multi-Label Patent Classification
Implements the methodology from Section 4.1 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaConfig
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import math


class ContrastiveEncoder(nn.Module):
    """
    RoBERTa-based encoder with projection layers for contrastive learning
    
    Architecture:
    RoBERTa-large -> Projection layers -> d-dimensional embeddings
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # RoBERTa backbone
        roberta_config = RobertaConfig.from_pretrained(config.backbone_model)
        self.roberta = RobertaModel.from_pretrained(
            config.backbone_model, 
            config=roberta_config
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, config.embedding_dim),
            nn.ReLU(),
            nn.Linear(config.embedding_dim, config.projection_dim),
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.projection_dim)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            embeddings: L2-normalized embeddings [batch_size, projection_dim]
        """
        # RoBERTa forward pass
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # Project to contrastive space
        projected = self.projection_head(cls_embedding)  # [batch_size, projection_dim]
        projected = self.layer_norm(projected)
        
        # L2 normalization for cosine similarity
        embeddings = F.normalize(projected, p=2, dim=1)
        
        return embeddings


class MultiLabelContrastiveLoss(nn.Module):
    """
    Multi-label contrastive loss combining instance-level and label-aware objectives
    Implements Equations 1-4 from the methodology
    """
    
    def __init__(self, config, num_categories: int):
        super().__init__()
        self.config = config
        self.num_categories = num_categories
        self.tau = config.temperature
        self.lambda_label = config.lambda_label
        self.alpha_penalty = config.alpha_penalty
        
    def compute_label_similarity(self, labels_i: torch.Tensor, labels_j: torch.Tensor) -> torch.Tensor:
        """
        Compute label similarity using Equation 3: Jaccard + penalty term
        
        Args:
            labels_i: Multi-hot labels for sample i [batch_size, num_categories]
            labels_j: Multi-hot labels for sample j [batch_size, num_categories]
            
        Returns:
            similarity: Label similarity scores [batch_size, batch_size]
        """
        # Expand for pairwise computation
        labels_i_expanded = labels_i.unsqueeze(1)  # [batch_size, 1, num_categories]
        labels_j_expanded = labels_j.unsqueeze(0)  # [1, batch_size, num_categories]
        
        # Intersection and union
        intersection = torch.sum(labels_i_expanded * labels_j_expanded, dim=2)
        union = torch.sum(torch.clamp(labels_i_expanded + labels_j_expanded, max=1.0), dim=2)
        
        # Jaccard similarity (avoid division by zero)
        jaccard = intersection / (union + 1e-8)
        
        # Symmetric difference penalty
        symmetric_diff = torch.sum(torch.abs(labels_i_expanded - labels_j_expanded), dim=2)
        penalty = torch.exp(-self.alpha_penalty * symmetric_diff)
        
        # Combined label similarity (Equation 3)
        label_similarity = jaccard * penalty
        
        return label_similarity
    
    def compute_instance_loss(self, embeddings: torch.Tensor, labels: torch.Tensor,
                            temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute instance-level contrastive loss (Equation 2)
        
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Multi-hot labels [batch_size, num_categories]
            temperature: Optional adaptive temperature [batch_size] or scalar
            
        Returns:
            loss: Instance-level contrastive loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        if temperature is None:
            temperature = self.tau
            
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # [batch_size, batch_size]
        
        # Apply temperature scaling
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            # Adaptive temperature per sample
            temperature_matrix = temperature.unsqueeze(1)  # [batch_size, 1]
            similarity_matrix = similarity_matrix / temperature_matrix
        else:
            similarity_matrix = similarity_matrix / temperature
        
        # Find positive pairs: patents sharing at least one label
        labels_expanded_i = labels.unsqueeze(1)  # [batch_size, 1, num_categories]
        labels_expanded_j = labels.unsqueeze(0)  # [1, batch_size, num_categories]
        
        # Positive mask: intersection is non-empty
        positive_mask = torch.sum(labels_expanded_i * labels_expanded_j, dim=2) > 0
        positive_mask = positive_mask.float()
        
        # Remove self-similarities
        identity_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        positive_mask = positive_mask.masked_fill(identity_mask, 0.0)
        
        # Numerator: sum over positive pairs
        exp_sim = torch.exp(similarity_matrix)
        numerator = torch.sum(exp_sim * positive_mask, dim=1)  # [batch_size]
        
        # Denominator: sum over all pairs except self
        denominator_mask = ~identity_mask
        denominator = torch.sum(exp_sim * denominator_mask.float(), dim=1)  # [batch_size]
        
        # Instance loss (Equation 2, corrected without |P_i| normalization)
        loss = -torch.log(numerator / (denominator + 1e-8))
        
        # Only compute loss for samples that have positive pairs
        valid_samples = (torch.sum(positive_mask, dim=1) > 0)
        
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss
    
    def compute_label_loss(self, embeddings: torch.Tensor, labels: torch.Tensor,
                          temperature: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute label-aware contrastive loss (Equation 4, corrected)
        
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Multi-hot labels [batch_size, num_categories]
            temperature: Optional adaptive temperature
            
        Returns:
            loss: Label-aware contrastive loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        if temperature is None:
            temperature = self.tau
            
        # Compute similarity matrices
        similarity_matrix = torch.matmul(embeddings, embeddings.T)  # [batch_size, batch_size]
        label_similarity = self.compute_label_similarity(labels, labels)  # [batch_size, batch_size]
        
        # Apply temperature
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature_matrix = temperature.unsqueeze(1)
            similarity_matrix = similarity_matrix / temperature_matrix
        else:
            similarity_matrix = similarity_matrix / temperature
        
        # Remove self-similarities
        identity_mask = torch.eye(batch_size, device=device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(identity_mask, float('-inf'))
        label_similarity = label_similarity.masked_fill(identity_mask, 0.0)
        
        # Compute weighted contrastive loss (Equation 4, corrected with normalization)
        exp_sim = torch.exp(similarity_matrix)
        
        # Numerator: label-similarity weighted positive terms
        weighted_numerator = torch.sum(label_similarity * exp_sim, dim=1)  # [batch_size]
        
        # Denominator: all pairs
        denominator = torch.sum(exp_sim, dim=1)  # [batch_size]
        
        # Normalization factor to make it a proper probability distribution
        label_sum = torch.sum(label_similarity, dim=1)  # [batch_size]
        
        # Label loss with proper normalization (Equation 4, corrected)
        loss = -torch.log(weighted_numerator / (denominator + 1e-8)) / (label_sum + 1e-8)
        
        # Only compute loss for samples with valid label similarities
        valid_samples = (label_sum > 1e-8)
        
        if valid_samples.sum() > 0:
            loss = loss[valid_samples].mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        return loss
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor,
                complexity_scores: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined multi-label contrastive loss (Equation 1)
        
        Args:
            embeddings: L2-normalized embeddings [batch_size, embedding_dim]
            labels: Multi-hot labels [batch_size, num_categories]
            complexity_scores: Optional complexity scores for adaptive temperature [batch_size]
            
        Returns:
            total_loss: Combined contrastive loss
            loss_dict: Dictionary with individual loss components
        """
        # Adaptive temperature scaling (Equation 5)
        temperature = self.tau
        if complexity_scores is not None:
            temperature = self.config.tau_0 * (1 + self.config.beta_complexity * complexity_scores)
        
        # Compute individual losses
        instance_loss = self.compute_instance_loss(embeddings, labels, temperature)
        label_loss = self.compute_label_loss(embeddings, labels, temperature)
        
        # Combined loss (Equation 1)
        total_loss = instance_loss + self.lambda_label * label_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'instance_loss': instance_loss,
            'label_loss': label_loss,
            'lambda_label': torch.tensor(self.lambda_label)
        }
        
        return total_loss, loss_dict


class MomentumEncoder:
    """
    Momentum encoder for stable positive pair generation
    Implements Equation 6 from the methodology
    """
    
    def __init__(self, encoder: ContrastiveEncoder, momentum: float = 0.99):
        self.momentum = momentum
        self.encoder = copy.deepcopy(encoder)
        
        # Initialize with same weights as main encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def update(self, main_encoder: ContrastiveEncoder):
        """
        Update momentum encoder weights using exponential moving average
        θ' = m*θ' + (1-m)*θ (Equation 6)
        """
        for main_param, momentum_param in zip(main_encoder.parameters(), self.encoder.parameters()):
            momentum_param.data = (
                self.momentum * momentum_param.data + 
                (1.0 - self.momentum) * main_param.data
            )
    
    @torch.no_grad()
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode using momentum encoder (no gradients)"""
        return self.encoder(input_ids, attention_mask)


class ContrastivePatentModel(nn.Module):
    """
    Complete contrastive learning model for patent classification
    Combines encoder, momentum encoder, and multi-label contrastive loss
    """
    
    def __init__(self, config, num_categories: int):
        super().__init__()
        self.config = config
        self.num_categories = num_categories
        
        # Main encoder
        self.encoder = ContrastiveEncoder(config)
        
        # Momentum encoder
        self.momentum_encoder = MomentumEncoder(self.encoder, config.momentum)
        
        # Contrastive loss
        self.contrastive_loss = MultiLabelContrastiveLoss(config, num_categories)
        
        # Complexity scorer (for adaptive temperature)
        self.complexity_scorer = nn.Linear(config.projection_dim, 1)
        
    def compute_complexity_scores(self, abstracts: List[str]) -> torch.Tensor:
        """Compute technical complexity scores for adaptive temperature scaling"""
        from data_utils import PatentProcessor
        processor = PatentProcessor()
        
        scores = []
        for abstract in abstracts:
            complexity = processor.compute_complexity(abstract)
            scores.append(complexity)
        
        return torch.tensor(scores, dtype=torch.float32, device=next(self.parameters()).device)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: torch.Tensor, abstracts: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for contrastive learning
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Multi-hot labels [batch_size, num_categories]
            abstracts: List of patent abstracts for complexity scoring
            
        Returns:
            output_dict: Dictionary containing embeddings, losses, and metadata
        """
        # Encode using main encoder
        embeddings = self.encoder(input_ids, attention_mask)
        
        # Compute complexity scores for adaptive temperature
        complexity_scores = None
        if abstracts is not None:
            complexity_scores = self.compute_complexity_scores(abstracts)
        
        # Compute contrastive loss
        total_loss, loss_dict = self.contrastive_loss(embeddings, labels, complexity_scores)
        
        # Update momentum encoder
        self.momentum_encoder.update(self.encoder)
        
        output = {
            'embeddings': embeddings,
            'loss': total_loss,
            **loss_dict
        }
        
        if complexity_scores is not None:
            output['complexity_scores'] = complexity_scores
        
        return output
    
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode patents without computing loss"""
        return self.encoder(input_ids, attention_mask)
    
    @torch.no_grad()
    def encode_momentum(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode using momentum encoder"""
        return self.momentum_encoder.encode(input_ids, attention_mask)