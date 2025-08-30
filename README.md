# Contrastive Learning Enhanced Retrieval-Augmented Few-Shot Framework for Multi-Label Patent Classification

This repository contains the complete implementation of the framework described in our paper "Contrastive learning enhanced retrieval-augmented few-shot framework for multi-label patent classification".

## Overview

Our framework integrates four main components:

1. **Contrastive Pre-training** (`contrastive_model.py`): Domain-adapted embeddings that capture multi-label co-occurrence patterns
2. **Retrieval-Augmented Demonstration Selection** (`retrieval_module.py`): Multi-faceted similarity scoring for informative example selection
3. **Few-Shot Multi-Label Prediction** (`few_shot_predictor.py`): Structured prompt construction with embedding-guided attention
4. **Chain-of-Thought Reasoning** (`cot_reasoning.py`): GPT-4o integration for interpretable multi-label decisions

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd patent-classification-framework

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run complete experiment with sample data
python run_experiment.py --create_sample_data --output_dir ./results

# Quick evaluation (reduced episodes for testing)
python run_experiment.py --create_sample_data --quick_eval --output_dir ./results

# Skip training phase (for inference only)
python run_experiment.py --skip_training --output_dir ./results
```

### Custom Configuration

```python
from config import Config, get_config
from main_framework import PatentClassificationFramework

# Load and modify configuration
config = get_config()
config.model.num_demonstrations = 10  # Increase demonstrations
config.data.k_shots = [1, 5, 10]      # Custom shot settings

# Initialize framework
framework = PatentClassificationFramework(config)
```

## Architecture

### Core Components

```
patent-classification-framework/
├── config.py                 # Configuration management
├── data_utils.py             # Patent data handling and episode generation
├── contrastive_model.py      # Multi-label contrastive learning
├── retrieval_module.py       # Demonstration selection with similarity scoring
├── few_shot_predictor.py     # Structured prediction with attention mechanisms
├── cot_reasoning.py          # Chain-of-thought reasoning with GPT-4o
├── main_framework.py         # Complete pipeline integration
├── evaluation.py             # Metrics and statistical testing
└── run_experiment.py         # Experimental runner
```

### Key Features

- **Algorithm 1 Implementation**: Complete pipeline following paper methodology
- **Mathematical Accuracy**: All equations (1-30) correctly implemented
- **Modular Design**: Each component can be used independently
- **Comprehensive Evaluation**: Macro-F1, Micro-F1, LRAP, Coverage Error with statistical testing
- **GPT-4o Integration**: Structured reasoning for interpretable predictions

## Methodology Implementation

### Phase 1: Contrastive Pre-training

```python
# Multi-label contrastive loss (Equations 1-4)
contrastive_model = ContrastivePatentModel(config, num_categories)
loss_total = loss_instance + λ * loss_label

# Adaptive temperature scaling (Equation 5)
τ_i = τ_0 * (1 + β * complexity(x_i))
```

### Phase 2: Retrieval-Augmented Selection

```python
# Multi-faceted similarity scoring (Equation 6)
score = α1 * sim_semantic + α2 * sim_technical + α3 * diversity

# Adaptive weighting (Equation 10)
α2_adaptive = α2 * (1 + δ * tech_density_ratio)
```

### Phase 3: Few-Shot Prediction

```python
# Embedding-guided attention (Equations 15-16)
w_i = softmax(sim(z_q, z_i)/τ_attn + b_i)

# Adaptive thresholding (Equation 18)
τ_j^thresh = τ_0 + Δτ*(freq - μ_freq) + γ*uncertainty
```

### Phase 4: Chain-of-Thought Reasoning

```python
# CoT-enhanced prediction (Equation 26)
ŷ_CoT = β * p_CoT + (1-β) * p_base

# GPT-4o configuration
gpt_config = {
    "temperature": 0.3,
    "max_tokens": 2048,
    "top_p": 0.9,
    "frequency_penalty": 0.2
}
```

## Experimental Results

Our framework achieves the following results on UAV patent classification:

| Method | Macro-F1 | Micro-F1 | LRAP | Coverage Error |
|--------|----------|-----------|------|----------------|
| **Our Framework** | **0.847±0.021** | **0.892±0.018** | **0.878±0.019** | **1.23±0.087** |
| RoBERTa-Large | 0.729±0.034 | 0.801±0.029 | 0.756±0.032 | 1.87±0.142 |
| XLNet-Large | 0.741±0.031 | 0.815±0.027 | 0.768±0.028 | 1.74±0.128 |

### Component Contributions (Ablation Study)

- **Contrastive Pre-training**: 6.8% drop when removed
- **Semantic Retrieval**: 9.7% drop when removed  
- **Chain-of-Thought**: 5.4% drop when removed
- **Inter-label Dependencies**: 2.8% drop when removed

## Configuration

The framework uses a hierarchical configuration system:

```python
@dataclass
class Config:
    model: ModelConfig        # Architecture parameters
    gpt: GPTConfig           # GPT-4o API settings
    data: DataConfig         # Dataset configuration
    training: TrainingConfig # Training parameters
    evaluation: EvaluationConfig # Evaluation setup
```

### Key Parameters

```python
# Model configuration
config.model.temperature = 0.1           # Contrastive temperature
config.model.lambda_label = 0.5          # Label loss weight
config.model.num_demonstrations = 5      # Retrieved examples
config.model.beta_cot = 0.7             # CoT vs base prediction weight

# Data configuration  
config.data.k_shots = [1, 3, 5, 10]     # Shot settings
config.data.num_episodes = 50           # Evaluation episodes
config.data.categories = [...]          # Target categories
```

## Hardware Requirements

- **GPU**: NVIDIA GeForce RTX 4090 (32GB) or equivalent
- **CPU**: Multi-core processor for parallel processing
- **RAM**: 32GB+ recommended for large patent datasets
- **Storage**: 50GB+ for datasets and model checkpoints

## Data Format

Patents should be formatted as JSON with the following structure:

```json
{
  "id": "patent_000001",
  "abstract": "This invention relates to a vertical takeoff...",
  "labels": ["VTOL & Hybrid Flight", "Flight Control & Stability"],
  "year": 2021,
  "ipc_codes": ["B64C29/00", "B64C27/82"],
  "citations": ["patent_000123", "patent_000456"],
  "technical_terms": ["vtol", "autopilot", "flight control"]
}
```

## Evaluation

The framework includes comprehensive evaluation with:

- **Multi-label Metrics**: Macro-F1, Micro-F1, LRAP, Coverage Error
- **Statistical Testing**: Paired t-tests with Bonferroni correction
- **Ablation Analysis**: Component contribution measurement
- **Category-wise Analysis**: Per-category performance breakdown

```python
# Run evaluation
evaluator = FewShotEvaluationSuite(categories)
results = evaluator.evaluate_episodes(method_results)
comparison = evaluator.compare_methods(results, baseline="RoBERTa-Large")
```

## API Integration

### OpenAI GPT-4o Setup

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

```python
# GPT-4o configuration in code
config.gpt.model = "gpt-4o"
config.gpt.temperature = 0.3
config.gpt.max_tokens = 2048
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{zheng2025contrastive,
  title={Contrastive learning enhanced retrieval-augmented few-shot framework for multi-label patent classification},
  author={Zheng, Wenlong and Li, Xin and Cui, Guoqing},
  journal={PLOS ONE},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

- **Wenlong Zheng**: Ningbo University of Finance and Economics
- **Xin Li**: The First Topographic Surveying Brigade of Ministry of Natural Resource of P.R.C
- **Guoqing Cui**: Northwest Land and Resources Research Center, Shaanxi Normal University

For questions about the implementation, please open an issue in this repository.

## Acknowledgments

- Patent data sourced from National Intellectual Property Administration's IP Search and Consultation Center China
- Built on top of Hugging Face Transformers and PyTorch
- GPT-4o integration via OpenAI API
- Evaluation metrics implementation based on scikit-learn