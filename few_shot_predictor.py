"""
Few-Shot Multi-Label Prediction Module
Implements the methodology from Section 4.3 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math
from dataclasses import dataclass
from collections import defaultdict

from data_utils import Patent, PatentDataset


@dataclass
class PromptComponents:
    """Structure for prompt components"""
    instruction: str
    demonstrations: List[str]
    query: str
    full_prompt: str


class PromptConstructor:
    """
    Constructs structured prompts for few-shot multi-label classification
    Implements Equations 13-14 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.categories = config.categories
    
    def format_demonstration(self, patent: Patent) -> str:
        """
        Format a single demonstration as input-output pair (Equation 14)
        Demo(x_i, y_i) = "Patent": x_i → "Categories": {l_j : y_i^j = 1}
        """
        # Clean and truncate abstract if needed
        abstract = patent.abstract[:500] + "..." if len(patent.abstract) > 500 else patent.abstract
        
        # Format labels as comma-separated list
        label_list = ", ".join(patent.labels) if patent.labels else "None"
        
        demonstration = f"""Patent: {abstract}
Categories: {label_list}
---"""
        
        return demonstration
    
    def create_instruction(self) -> str:
        """Create task instruction emphasizing multi-label nature"""
        categories_list = ", ".join(self.categories)
        
        instruction = f"""You are an expert patent classifier. Your task is to classify patent abstracts into one or more of the following technological categories:

{categories_list}

Instructions:
1. A patent can belong to multiple categories simultaneously
2. Analyze the technical content and innovations described
3. Consider the examples provided as guidance for classification patterns
4. Respond with all applicable categories, or "None" if no categories apply
5. Be precise and consider the patent's core technological contributions

Examples:
"""
        return instruction
    
    def create_query_prompt(self, query_patent: Patent) -> str:
        """Create query prompt for classification"""
        abstract = query_patent.abstract[:500] + "..." if len(query_patent.abstract) > 500 else query_patent.abstract
        
        query_prompt = f"""Patent: {abstract}
Categories: """
        
        return query_prompt
    
    def construct_prompt(self, query_patent: Patent, demonstration_patents: List[Patent]) -> PromptComponents:
        """
        Construct complete prompt for few-shot classification (Equation 13)
        P(x_q, D_k) = Instruct ⊕ ⊕Demo(x_i, y_i) ⊕ Query(x_q)
        """
        # Create instruction
        instruction = self.create_instruction()
        
        # Format demonstrations
        demonstrations = []
        for demo_patent in demonstration_patents:
            demo_str = self.format_demonstration(demo_patent)
            demonstrations.append(demo_str)
        
        # Create query
        query = self.create_query_prompt(query_patent)
        
        # Combine all components
        full_prompt = instruction + "\n\n" + "\n\n".join(demonstrations) + "\n\n" + query
        
        return PromptComponents(
            instruction=instruction,
            demonstrations=demonstrations,
            query=query,
            full_prompt=full_prompt
        )


class AttentionMechanism:
    """
    Embedding-guided attention for demonstration weighting
    Implements Equations 15-16 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.tau_attn = config.tau_attention
        self.alpha_pos = config.alpha_pos
        self.beta_pos = config.beta_pos
        self.alpha_label = config.alpha_label_bias
    
    def compute_attention_bias(self, position: int, label_overlap: int) -> float:
        """
        Compute attention bias term (Equation 16)
        b_i = α_pos * exp(-β_pos * i) + α_label * |y_i ∩ y_q^pred|
        """
        # Position-dependent decay (earlier demonstrations get higher weight)
        position_term = self.alpha_pos * math.exp(-self.beta_pos * position)
        
        # Label overlap term
        label_term = self.alpha_label * label_overlap
        
        return position_term + label_term
    
    def compute_attention_weights(self, query_embedding: torch.Tensor,
                                demo_embeddings: List[torch.Tensor],
                                predicted_labels: List[str],
                                demo_patents: List[Patent]) -> torch.Tensor:
        """
        Compute embedding-guided attention weights (Equation 15)
        w_i = softmax(sim(z_q, z_i)/τ_attn + b_i)
        """
        if len(demo_embeddings) != len(demo_patents):
            raise ValueError("Mismatch between embeddings and patents")
        
        attention_scores = []
        
        for i, (demo_embedding, demo_patent) in enumerate(zip(demo_embeddings, demo_patents)):
            # Cosine similarity
            similarity = F.cosine_similarity(query_embedding, demo_embedding, dim=0).item()
            
            # Label overlap count
            demo_labels = set(demo_patent.labels)
            pred_labels = set(predicted_labels)
            label_overlap = len(demo_labels & pred_labels)
            
            # Attention bias
            bias = self.compute_attention_bias(i, label_overlap)
            
            # Combined attention score
            attention_score = similarity / self.tau_attn + bias
            attention_scores.append(attention_score)
        
        # Convert to tensor and apply softmax
        attention_tensor = torch.tensor(attention_scores, dtype=torch.float32)
        attention_weights = F.softmax(attention_tensor, dim=0)
        
        return attention_weights


class InterLabelDependencyModel:
    """
    Models dependencies between labels for structured prediction
    Implements the ω_jm terms in Equation 17
    """
    
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.category_to_id = {cat: idx for idx, cat in enumerate(categories)}
        self.dependencies = {}
        
        # Initialize dependency matrix
        self._initialize_dependencies()
    
    def _initialize_dependencies(self):
        """Initialize label dependencies based on domain knowledge"""
        # In practice, these could be learned from data or defined based on expert knowledge
        # For UAV patents, some example dependencies:
        dependencies = {
            ("VTOL & Hybrid Flight", "Flight Control & Stability"): 0.8,
            ("Surveillance & Mapping", "Flight Control & Stability"): 0.6,  
            ("Endurance & Power Systems", "VTOL & Hybrid Flight"): 0.5,
            ("Modular & Deployable", "Logistics & Cargo"): 0.7,
            ("Bionic & Flapping Wing", "Flight Control & Stability"): 0.4,
            # Add more dependencies as needed
        }
        
        # Convert to matrix form
        num_cats = len(self.categories)
        self.dependency_matrix = torch.zeros(num_cats, num_cats)
        
        for (cat1, cat2), weight in dependencies.items():
            if cat1 in self.category_to_id and cat2 in self.category_to_id:
                idx1 = self.category_to_id[cat1]
                idx2 = self.category_to_id[cat2]
                self.dependency_matrix[idx1, idx2] = weight
                self.dependency_matrix[idx2, idx1] = weight  # Symmetric
    
    def get_dependency_score(self, category_j: str, assigned_labels: List[str]) -> float:
        """Get dependency score for category j given already assigned labels"""
        if category_j not in self.category_to_id:
            return 0.0
            
        j_idx = self.category_to_id[category_j]
        dependency_sum = 0.0
        
        for assigned_label in assigned_labels:
            if assigned_label in self.category_to_id:
                m_idx = self.category_to_id[assigned_label]
                dependency_sum += self.dependency_matrix[j_idx, m_idx].item()
        
        return dependency_sum


class AdaptiveThresholdingModule:
    """
    Adaptive thresholding based on uncertainty and category frequency
    Implements Equation 18-19 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.tau_base = config.tau_threshold_base
        self.delta_tau = config.delta_tau
        self.gamma_uncertainty = config.gamma_uncertainty
    
    def compute_uncertainty(self, demonstration_labels: List[List[str]], 
                           attention_weights: torch.Tensor, 
                           category: str) -> float:
        """
        Compute prediction uncertainty for adaptive thresholding (Equation 19, corrected)
        uncertainty = -p_j log(p_j) - (1-p_j) log(1-p_j) where p_j = Σw_i·y_i^j
        """
        # Compute weighted probability that category is present
        prob_present = 0.0
        
        for demo_labels, weight in zip(demonstration_labels, attention_weights):
            if category in demo_labels:
                prob_present += weight.item()
        
        # Clamp probability to avoid log(0)
        prob_present = max(1e-8, min(1 - 1e-8, prob_present))
        
        # Binary entropy as uncertainty measure
        uncertainty = -prob_present * math.log(prob_present) - (1 - prob_present) * math.log(1 - prob_present)
        
        return uncertainty
    
    def compute_category_frequency(self, category: str, corpus_patents: List[Patent]) -> float:
        """Compute category frequency in corpus"""
        if not corpus_patents:
            return 0.0
            
        count = sum(1 for patent in corpus_patents if category in patent.labels)
        frequency = count / len(corpus_patents)
        
        return frequency
    
    def compute_adaptive_threshold(self, category: str, 
                                 demonstration_labels: List[List[str]],
                                 attention_weights: torch.Tensor,
                                 corpus_patents: List[Patent]) -> float:
        """
        Compute adaptive threshold for category (Equation 18)
        τ_j^thresh = τ_0 + Δτ·(freq(l_j) - μ_freq) + γ·uncertainty(x_q, l_j)
        """
        # Category frequency
        category_freq = self.compute_category_frequency(category, corpus_patents)
        
        # Mean frequency across all categories
        all_frequencies = [
            self.compute_category_frequency(cat, corpus_patents) 
            for cat in self.config.categories
        ]
        mean_freq = sum(all_frequencies) / len(all_frequencies) if all_frequencies else 0.0
        
        # Uncertainty
        uncertainty = self.compute_uncertainty(demonstration_labels, attention_weights, category)
        
        # Adaptive threshold
        adaptive_threshold = (
            self.tau_base + 
            self.delta_tau * (category_freq - mean_freq) +
            self.gamma_uncertainty * uncertainty
        )
        
        return adaptive_threshold


class PrototypeFallbackModule:
    """
    Prototype-based fallback for sparse categories
    Implements Equations 20-22 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.epsilon_sparse = config.epsilon_sparse
        self.tau_proto = config.tau_proto
        self.category_prototypes = {}
    
    def build_prototypes(self, dataset: PatentDataset, encoder):
        """
        Build category prototypes (Equation 20)
        proto_j = (1/|S_j|) * Σ f_θ(x) for x in S_j
        """
        encoder.eval()
        device = next(encoder.parameters()).device
        
        with torch.no_grad():
            for category in dataset.categories:
                category_patents = dataset.get_patents_by_category(category)
                
                if not category_patents:
                    continue
                
                embeddings = []
                for patent in category_patents:
                    tokens = dataset.tokenize_text(patent.abstract)
                    input_ids = tokens['input_ids'].unsqueeze(0).to(device)
                    attention_mask = tokens['attention_mask'].unsqueeze(0).to(device)
                    
                    embedding = encoder.encode(input_ids, attention_mask).squeeze(0)
                    embeddings.append(embedding)
                
                if embeddings:
                    prototype = torch.stack(embeddings).mean(dim=0)
                    self.category_prototypes[category] = F.normalize(prototype, p=2, dim=0)
    
    def compute_fallback_probability(self, query_embedding: torch.Tensor, category: str) -> float:
        """
        Compute prototype-based fallback probability (Equation 21, corrected)
        p_fallback = σ(z_q^T * proto_j / (||z_q|| ||proto_j|| τ_proto))
        """
        if category not in self.category_prototypes:
            return 0.0
        
        prototype = self.category_prototypes[category]
        
        # Cosine similarity scaled by temperature
        cosine_sim = F.cosine_similarity(query_embedding, prototype, dim=0).item()
        scaled_sim = cosine_sim / self.tau_proto
        
        # Sigmoid activation
        fallback_prob = torch.sigmoid(torch.tensor(scaled_sim)).item()
        
        return fallback_prob
    
    def is_sparse_category(self, category: str, demonstration_labels: List[List[str]]) -> bool:
        """Check if category has insufficient demonstration coverage"""
        category_count = sum(1 for demo_labels in demonstration_labels if category in demo_labels)
        return category_count < self.epsilon_sparse
    
    def get_demo_coverage_weight(self, category: str, demonstration_labels: List[List[str]]) -> float:
        """Compute demonstration coverage weight (α_demo in Equation 22)"""
        category_count = sum(1 for demo_labels in demonstration_labels if category in demo_labels)
        alpha_demo = min(1.0, category_count / self.epsilon_sparse)
        return alpha_demo


class FewShotPredictor:
    """
    Main few-shot prediction module
    Integrates all components for multi-label classification
    """
    
    def __init__(self, config, dataset: PatentDataset):
        self.config = config
        self.dataset = dataset
        
        # Initialize components
        self.prompt_constructor = PromptConstructor(config)
        self.attention_mechanism = AttentionMechanism(config)
        self.dependency_model = InterLabelDependencyModel(config.categories)
        self.threshold_module = AdaptiveThresholdingModule(config)
        self.prototype_module = PrototypeFallbackModule(config)
        
        # Language model scoring weights
        self.lambda1_scoring = config.lambda1_scoring
        self.lambda2_scoring = config.lambda2_scoring
    
    def initialize_prototypes(self, encoder):
        """Initialize category prototypes for fallback mechanism"""
        self.prototype_module.build_prototypes(self.dataset, encoder)
    
    def compute_base_probability(self, category: str, 
                               language_model_score: float,
                               category_similarity: float,
                               embedding_prior: float,
                               dependency_score: float) -> float:
        """
        Compute base probability combining multiple evidence sources (Equation 17-18)
        """
        # Combine evidence sources (Equation 18)
        combined_score = (
            language_model_score + 
            self.lambda1_scoring * category_similarity +
            self.lambda2_scoring * embedding_prior
        )
        
        # Add inter-label dependencies (Equation 17)
        final_score = combined_score + dependency_score
        
        # Sigmoid activation
        probability = torch.sigmoid(torch.tensor(final_score)).item()
        
        return probability
    
    def predict_single_category(self, category: str,
                              query_embedding: torch.Tensor,
                              demonstration_patents: List[Patent],
                              demo_embeddings: List[torch.Tensor],
                              attention_weights: torch.Tensor,
                              assigned_labels: List[str],
                              language_model_scores: Dict[str, float]) -> Tuple[float, bool]:
        """
        Predict single category with fallback mechanism (Equations 17-22)
        """
        demonstration_labels = [patent.labels for patent in demonstration_patents]
        
        # Check if sparse category
        is_sparse = self.prototype_module.is_sparse_category(category, demonstration_labels)
        
        # Compute base probability
        lm_score = language_model_scores.get(category, 0.0)
        category_sim = 0.0  # Would compute from category prototypes
        embedding_prior = 0.0  # Would compute from embedding
        dependency_score = self.dependency_model.get_dependency_score(category, assigned_labels)
        
        base_prob = self.compute_base_probability(
            category, lm_score, category_sim, embedding_prior, dependency_score
        )
        
        # Compute final probability with fallback
        if is_sparse:
            # Fallback probability
            fallback_prob = self.prototype_module.compute_fallback_probability(query_embedding, category)
            
            # Coverage weight
            alpha_demo = self.prototype_module.get_demo_coverage_weight(category, demonstration_labels)
            
            # Combined probability (Equation 22)
            final_prob = alpha_demo * base_prob + (1 - alpha_demo) * fallback_prob
        else:
            final_prob = base_prob
        
        # Adaptive thresholding
        threshold = self.threshold_module.compute_adaptive_threshold(
            category, demonstration_labels, attention_weights, self.dataset.patents
        )
        
        # Final prediction
        predicted = final_prob > threshold
        
        return final_prob, predicted
    
    def predict_multi_label(self, query_patent: Patent,
                          query_embedding: torch.Tensor,
                          demonstration_patents: List[Patent],
                          demo_embeddings: List[torch.Tensor],
                          language_model_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Perform multi-label prediction with inter-label dependencies
        """
        # Estimate predicted labels for attention computation (simplified)
        predicted_labels_estimate = []
        for category, score in language_model_scores.items():
            if score > 0.5:  # Simple threshold for estimation
                predicted_labels_estimate.append(category)
        
        # Compute attention weights
        attention_weights = self.attention_mechanism.compute_attention_weights(
            query_embedding, demo_embeddings, predicted_labels_estimate, demonstration_patents
        )
        
        # Predict categories in order (to handle dependencies)
        category_probabilities = {}
        category_predictions = {}
        assigned_labels = []
        
        # Sort categories by language model confidence for dependency modeling
        sorted_categories = sorted(
            self.config.categories, 
            key=lambda cat: language_model_scores.get(cat, 0.0), 
            reverse=True
        )
        
        for category in sorted_categories:
            prob, predicted = self.predict_single_category(
                category, query_embedding, demonstration_patents, demo_embeddings,
                attention_weights, assigned_labels, language_model_scores
            )
            
            category_probabilities[category] = prob
            category_predictions[category] = predicted
            
            if predicted:
                assigned_labels.append(category)
        
        return {
            'probabilities': category_probabilities,
            'predictions': category_predictions,
            'predicted_labels': assigned_labels,
            'attention_weights': attention_weights.tolist()
        }
    
    def predict(self, query_patent: Patent,
               demonstration_patents: List[Patent],
               encoder,
               language_model_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Main prediction interface
        """
        # Get embeddings
        device = next(encoder.parameters()).device
        encoder.eval()
        
        with torch.no_grad():
            # Query embedding
            query_tokens = self.dataset.tokenize_text(query_patent.abstract)
            query_input_ids = query_tokens['input_ids'].unsqueeze(0).to(device)
            query_attention_mask = query_tokens['attention_mask'].unsqueeze(0).to(device)
            query_embedding = encoder.encode(query_input_ids, query_attention_mask).squeeze(0)
            
            # Demonstration embeddings
            demo_embeddings = []
            for demo_patent in demonstration_patents:
                demo_tokens = self.dataset.tokenize_text(demo_patent.abstract)
                demo_input_ids = demo_tokens['input_ids'].unsqueeze(0).to(device)
                demo_attention_mask = demo_tokens['attention_mask'].unsqueeze(0).to(device)
                demo_embedding = encoder.encode(demo_input_ids, demo_attention_mask).squeeze(0)
                demo_embeddings.append(demo_embedding)
        
        # Predict labels
        prediction_results = self.predict_multi_label(
            query_patent, query_embedding, demonstration_patents, 
            demo_embeddings, language_model_scores
        )
        
        # Add prompt for reference
        prompt_components = self.prompt_constructor.construct_prompt(
            query_patent, demonstration_patents
        )
        
        prediction_results['prompt'] = prompt_components.full_prompt
        
        return prediction_results