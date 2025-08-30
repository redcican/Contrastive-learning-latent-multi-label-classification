"""
Configuration file for Retrieval-Augmented Contrastive Learning 
for Multi-Label Patent Classification
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # RoBERTa backbone
    backbone_model: str = "roberta-large"
    embedding_dim: int = 1024
    projection_dim: int = 256
    
    # Contrastive learning parameters
    temperature: float = 0.1
    lambda_label: float = 0.5  # Balance between instance and label loss
    alpha_penalty: float = 1.0  # Label disparity penalty
    momentum: float = 0.99  # Momentum encoder coefficient
    
    # Temperature scaling parameters
    beta_complexity: float = 0.1
    tau_0: float = 0.1
    
    # Retrieval parameters
    num_demonstrations: int = 5
    alpha_semantic: float = 0.4
    alpha_technical: float = 0.3
    alpha_diversity: float = 0.3
    gamma_edit: float = 0.1
    delta_adaptive: float = 0.2
    eta_label_match: float = 1.0
    
    # Few-shot prediction parameters
    tau_attention: float = 0.1
    alpha_pos: float = 1.0
    beta_pos: float = 0.1
    alpha_label_bias: float = 0.5
    lambda1_scoring: float = 0.3
    lambda2_scoring: float = 0.2
    
    # Adaptive thresholding
    tau_threshold_base: float = 0.5
    delta_tau: float = 0.1
    gamma_uncertainty: float = 0.2
    tau_proto: float = 0.1
    epsilon_sparse: float = 2.0
    
    # Chain-of-thought parameters
    beta_cot: float = 0.7  # CoT vs base prediction weight


@dataclass
class GPTConfig:
    """GPT-4o API configuration"""
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 2048
    top_p: float = 0.9
    frequency_penalty: float = 0.2
    response_format: Dict[str, str] = None
    
    def __post_init__(self):
        if self.response_format is None:
            self.response_format = {"type": "json_object"}


@dataclass
class DataConfig:
    """Dataset configuration"""
    # Dataset paths
    data_dir: str = "./data"
    patent_corpus_file: str = "uav_patents.json"
    annotations_file: str = "patent_annotations.json"
    
    # Dataset splitting
    train_years: tuple = (2000, 2020)
    test_years: tuple = (2021, 2023)
    train_size: int = 12000
    test_size: int = 3000
    
    # Patent categories
    categories: List[str] = None
    num_categories: int = 10
    avg_labels_per_patent: float = 2.3
    
    # Few-shot setup
    n_way: int = 5  # Number of categories per episode
    k_shots: List[int] = None  # [1, 3, 5, 10]
    query_size: int = 100
    num_episodes: int = 50
    
    # Category frequency tiers
    frequent_categories: List[str] = None
    moderate_categories: List[str] = None
    sparse_categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = [
                "VTOL & Hybrid Flight",
                "Surveillance & Mapping", 
                "Flight Control & Stability",
                "Modular & Deployable",
                "Endurance & Power Systems",
                "Structural & Materials",
                "Logistics & Cargo",
                "Bionic & Flapping Wing", 
                "Specialized Applications",
                "Multi-Environment"
            ]
        
        if self.k_shots is None:
            self.k_shots = [1, 3, 5, 10]
            
        if self.frequent_categories is None:
            self.frequent_categories = [
                "VTOL & Hybrid Flight",
                "Surveillance & Mapping", 
                "Flight Control & Stability"
            ]
            
        if self.moderate_categories is None:
            self.moderate_categories = [
                "Modular & Deployable",
                "Endurance & Power Systems",
                "Structural & Materials"
            ]
            
        if self.sparse_categories is None:
            self.sparse_categories = [
                "Logistics & Cargo",
                "Bionic & Flapping Wing", 
                "Specialized Applications",
                "Multi-Environment"
            ]


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Contrastive pre-training
    pretrain_epochs: int = 50
    pretrain_batch_size: int = 32
    pretrain_lr: float = 5e-5
    warmup_steps: int = 1000
    
    # Fine-tuning
    finetune_epochs: int = 10
    finetune_lr: float = 2e-5
    
    # Optimization
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    
    # Training phases
    freeze_backbone_pretrain: bool = True
    joint_finetune: bool = True
    
    # Hardware
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Logging and checkpointing
    log_interval: int = 100
    save_interval: int = 1000
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics
    metrics: List[str] = None
    
    # Statistical testing
    significance_alpha: float = 0.0083  # Bonferroni corrected
    
    # Hardware specification (for reporting)
    hardware: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["macro_f1", "micro_f1", "lrap", "coverage_error"]
            
        if self.hardware is None:
            self.hardware = {
                "gpu": "NVIDIA GeForce RTX 4090",
                "memory": "32GB"
            }


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = None
    gpt: GPTConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    # Experiment settings
    experiment_name: str = "patent_classification"
    seed: int = 42
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.gpt is None:
            self.gpt = GPTConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()


def get_config() -> Config:
    """Get default configuration"""
    return Config()


def load_config(config_path: str) -> Config:
    """Load configuration from file"""
    # Implementation for loading from YAML/JSON would go here
    return Config()