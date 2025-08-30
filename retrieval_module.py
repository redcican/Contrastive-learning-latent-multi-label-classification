"""
Retrieval-Augmented Demonstration Selection Module
Implements the methodology from Section 4.2 of the paper
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass
import math
from collections import defaultdict
import editdistance

from data_utils import Patent, PatentDataset


@dataclass
class RetrievedDemo:
    """Structure for retrieved demonstration"""
    patent: Patent
    score: float
    semantic_sim: float
    technical_sim: float
    diversity_score: float


class TechnicalSimilarityScorer:
    """
    Computes technical domain alignment features
    Implements Equation 8 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.ipc_weight = 0.4  # w1
        self.term_weight = 0.4  # w2  
        self.cite_weight = 0.2  # w3
    
    def compute_ipc_overlap(self, query_patent: Patent, candidate_patent: Patent) -> float:
        """Compute IPC classification code overlap"""
        query_codes = set(query_patent.ipc_codes)
        candidate_codes = set(candidate_patent.ipc_codes)
        
        if not query_codes or not candidate_codes:
            return 0.0
            
        intersection = len(query_codes & candidate_codes)
        union = len(query_codes | candidate_codes)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_term_match(self, query_patent: Patent, candidate_patent: Patent) -> float:
        """Compute technical term matching score"""
        query_terms = set(query_patent.technical_terms)
        candidate_terms = set(candidate_patent.technical_terms)
        
        if not query_terms or not candidate_terms:
            return 0.0
            
        intersection = len(query_terms & candidate_terms)
        union = len(query_terms | candidate_terms)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_citation_relation(self, query_patent: Patent, candidate_patent: Patent) -> float:
        """Compute citation relationship score"""
        query_citations = set(query_patent.citations)
        candidate_citations = set(candidate_patent.citations)
        
        # Check if patents cite each other or share citations
        mutual_citation = (
            query_patent.id in candidate_citations or 
            candidate_patent.id in query_citations
        )
        
        if mutual_citation:
            return 1.0
            
        # Shared citations
        if query_citations and candidate_citations:
            intersection = len(query_citations & candidate_citations)
            union = len(query_citations | candidate_citations)
            return intersection / union if union > 0 else 0.0
        
        return 0.0
    
    def compute_technical_similarity(self, query_patent: Patent, candidate_patent: Patent) -> float:
        """
        Compute overall technical similarity (Equation 8)
        sim_tech(x_q, x_j) = w1·IPC_overlap + w2·term_match + w3·cite_rel
        """
        ipc_sim = self.compute_ipc_overlap(query_patent, candidate_patent)
        term_sim = self.compute_term_match(query_patent, candidate_patent)
        cite_sim = self.compute_citation_relation(query_patent, candidate_patent)
        
        technical_similarity = (
            self.ipc_weight * ipc_sim +
            self.term_weight * term_sim + 
            self.cite_weight * cite_sim
        )
        
        return technical_similarity


class SemanticSimilarityScorer:
    """
    Computes semantic similarity with edit distance penalty
    Implements Equation 7 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.gamma = config.gamma_edit
    
    def compute_edit_distance_penalty(self, query_abstract: str, candidate_abstract: str) -> float:
        """Compute normalized edit distance penalty"""
        # Normalize abstracts for comparison
        query_norm = query_abstract.lower().strip()
        candidate_norm = candidate_abstract.lower().strip()
        
        # Compute edit distance
        edit_dist = editdistance.eval(query_norm, candidate_norm)
        
        # Normalize by maximum possible distance
        max_len = max(len(query_norm), len(candidate_norm))
        if max_len == 0:
            return 0.0
            
        normalized_dist = edit_dist / max_len
        return normalized_dist
    
    def compute_semantic_similarity(self, query_embedding: torch.Tensor, 
                                  candidate_embedding: torch.Tensor,
                                  query_abstract: str, candidate_abstract: str) -> float:
        """
        Compute semantic similarity with edit distance penalty (Equation 7)
        sim_sem(z_q, z_j) = cosine_sim * exp(-γ * d_edit)
        """
        # Cosine similarity
        cosine_sim = F.cosine_similarity(query_embedding, candidate_embedding, dim=0).item()
        
        # Edit distance penalty
        edit_penalty = self.compute_edit_distance_penalty(query_abstract, candidate_abstract)
        penalty_factor = math.exp(-self.gamma * edit_penalty)
        
        # Combined semantic similarity
        semantic_similarity = cosine_sim * penalty_factor
        
        return semantic_similarity


class DiversityScorer:
    """
    Promotes diversity in retrieved demonstrations
    Implements Equation 9 from the methodology (corrected)
    """
    
    def __init__(self, config):
        self.config = config
    
    def compute_jaccard_similarity(self, labels_a: List[str], labels_b: List[str]) -> float:
        """Compute Jaccard similarity between label sets"""
        set_a = set(labels_a)
        set_b = set(labels_b)
        
        if not set_a or not set_b:
            return 0.0
            
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_diversity_score(self, candidate_patent: Patent, 
                              candidate_embedding: torch.Tensor,
                              retrieved_patents: List[Patent],
                              retrieved_embeddings: List[torch.Tensor]) -> float:
        """
        Compute diversity score to avoid redundant retrievals (Equation 9, corrected)
        div(x_j | R) = 1 - max(sim_sem * jaccard_similarity)
        """
        if not retrieved_patents:
            return 1.0  # Maximum diversity if no patents retrieved yet
        
        max_redundancy = 0.0
        
        for retrieved_patent, retrieved_embedding in zip(retrieved_patents, retrieved_embeddings):
            # Semantic similarity
            sem_sim = F.cosine_similarity(candidate_embedding, retrieved_embedding, dim=0).item()
            
            # Label overlap (Jaccard similarity)
            label_overlap = self.compute_jaccard_similarity(
                candidate_patent.labels, retrieved_patent.labels
            )
            
            # Combined redundancy score
            redundancy = sem_sim * label_overlap
            max_redundancy = max(max_redundancy, redundancy)
        
        # Diversity score (higher is more diverse)
        diversity_score = 1.0 - max_redundancy
        
        return diversity_score


class LabelMatchScorer:
    """
    Computes label matching score for multi-label retrieval
    Implements Equation 11 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.eta = config.eta_label_match
    
    def estimate_query_label_count(self, query_embedding: torch.Tensor,
                                 category_prototypes: Dict[str, torch.Tensor]) -> int:
        """Estimate number of labels for query based on embedding similarity to prototypes"""
        similarities = []
        
        for category, prototype in category_prototypes.items():
            sim = F.cosine_similarity(query_embedding, prototype, dim=0).item()
            similarities.append(sim)
        
        # Use a threshold to estimate number of likely labels
        threshold = 0.3  # This could be learned
        estimated_count = sum(1 for sim in similarities if sim > threshold)
        
        return max(1, estimated_count)  # At least one label
    
    def compute_label_probabilities(self, query_embedding: torch.Tensor,
                                  category_prototypes: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute probability of each label given query embedding"""
        probs = {}
        
        for category, prototype in category_prototypes.items():
            sim = F.cosine_similarity(query_embedding, prototype, dim=0).item()
            # Convert similarity to probability using sigmoid
            prob = torch.sigmoid(torch.tensor(sim * 3.0)).item()  # Scale for sharper probabilities
            probs[category] = prob
        
        return probs
    
    def compute_label_match_score(self, query_embedding: torch.Tensor,
                                candidate_patent: Patent,
                                category_prototypes: Dict[str, torch.Tensor]) -> float:
        """
        Compute label matching score (Equation 11)
        label_match = exp(-η * |pred_count - actual_count|) * ∏P(l|z_q)
        """
        # Estimate query label count
        pred_count = self.estimate_query_label_count(query_embedding, category_prototypes)
        actual_count = len(candidate_patent.labels)
        
        # Count difference penalty
        count_diff = abs(pred_count - actual_count)
        count_penalty = math.exp(-self.eta * count_diff)
        
        # Label probability product
        label_probs = self.compute_label_probabilities(query_embedding, category_prototypes)
        prob_product = 1.0
        
        for label in candidate_patent.labels:
            if label in label_probs:
                prob_product *= label_probs[label]
        
        # Combined label match score
        label_match_score = count_penalty * prob_product
        
        return label_match_score


class AdaptiveWeightingModule:
    """
    Dynamically adapts retrieval weights based on query characteristics
    Implements Equation 10 from the methodology
    """
    
    def __init__(self, config):
        self.config = config
        self.delta = config.delta_adaptive
    
    def compute_technical_density(self, patent: Patent) -> float:
        """Compute technical term density for adaptive weighting"""
        abstract_words = patent.abstract.split()
        if not abstract_words:
            return 0.0
            
        technical_density = len(patent.technical_terms) / len(abstract_words)
        return technical_density
    
    def compute_average_density(self, patents: List[Patent]) -> float:
        """Compute average technical density across patent corpus"""
        if not patents:
            return 0.0
            
        densities = [self.compute_technical_density(p) for p in patents]
        return sum(densities) / len(densities)
    
    def adapt_weights(self, query_patent: Patent, corpus_patents: List[Patent]) -> Tuple[float, float, float]:
        """
        Adapt retrieval weights based on query complexity (Equation 10)
        α2_adaptive = α2 * (1 + δ * tech_density_ratio)
        """
        query_density = self.compute_technical_density(query_patent)
        avg_density = self.compute_average_density(corpus_patents)
        
        if avg_density == 0:
            density_ratio = 1.0
        else:
            density_ratio = query_density / avg_density
        
        # Adapt technical weight
        alpha_semantic = self.config.alpha_semantic
        alpha_technical = self.config.alpha_technical * (1 + self.delta * density_ratio)
        alpha_diversity = self.config.alpha_diversity
        
        # Normalize weights to sum to 1
        total = alpha_semantic + alpha_technical + alpha_diversity
        alpha_semantic /= total
        alpha_technical /= total  
        alpha_diversity /= total
        
        return alpha_semantic, alpha_technical, alpha_diversity


class RetrievalModule:
    """
    Main retrieval module for demonstration selection
    Implements the complete retrieval process from Section 4.2
    """
    
    def __init__(self, config, dataset: PatentDataset):
        self.config = config
        self.dataset = dataset
        
        # Initialize scoring modules
        self.semantic_scorer = SemanticSimilarityScorer(config)
        self.technical_scorer = TechnicalSimilarityScorer(config)
        self.diversity_scorer = DiversityScorer(config)
        self.label_scorer = LabelMatchScorer(config)
        self.adaptive_weighter = AdaptiveWeightingModule(config)
        
        # Category prototypes for label matching
        self.category_prototypes = {}
    
    def build_category_prototypes(self, encoder, device: str = 'cuda'):
        """Build category prototypes by averaging embeddings of patents in each category"""
        print("Building category prototypes...")
        encoder.eval()
        
        with torch.no_grad():
            for category in self.dataset.categories:
                category_patents = self.dataset.get_patents_by_category(category)
                
                if not category_patents:
                    continue
                    
                embeddings = []
                for patent in category_patents[:50]:  # Limit for efficiency
                    tokens = self.dataset.tokenize_text(patent.abstract)
                    input_ids = tokens['input_ids'].unsqueeze(0).to(device)
                    attention_mask = tokens['attention_mask'].unsqueeze(0).to(device)
                    
                    embedding = encoder.encode(input_ids, attention_mask).squeeze(0)
                    embeddings.append(embedding)
                
                if embeddings:
                    prototype = torch.stack(embeddings).mean(dim=0)
                    self.category_prototypes[category] = F.normalize(prototype, p=2, dim=0)
        
        print(f"Built prototypes for {len(self.category_prototypes)} categories")
    
    def compute_multi_faceted_score(self, query_patent: Patent, query_embedding: torch.Tensor,
                                  candidate_patent: Patent, candidate_embedding: torch.Tensor,
                                  retrieved_patents: List[Patent], 
                                  retrieved_embeddings: List[torch.Tensor],
                                  adapted_weights: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Compute multi-faceted similarity score (Equation 6)
        score = α1·sim_sem + α2·sim_tech + α3·div
        """
        alpha_semantic, alpha_technical, alpha_diversity = adapted_weights
        
        # Semantic similarity (Equation 7)
        semantic_sim = self.semantic_scorer.compute_semantic_similarity(
            query_embedding, candidate_embedding, 
            query_patent.abstract, candidate_patent.abstract
        )
        
        # Technical similarity (Equation 8)  
        technical_sim = self.technical_scorer.compute_technical_similarity(
            query_patent, candidate_patent
        )
        
        # Diversity score (Equation 9)
        diversity_score = self.diversity_scorer.compute_diversity_score(
            candidate_patent, candidate_embedding, retrieved_patents, retrieved_embeddings
        )
        
        # Label matching score (Equation 11)
        label_match = self.label_scorer.compute_label_match_score(
            query_embedding, candidate_patent, self.category_prototypes
        )
        
        # Combined score
        total_score = (
            alpha_semantic * semantic_sim +
            alpha_technical * technical_sim + 
            alpha_diversity * diversity_score
        ) * label_match  # Multiply by label match as additional factor
        
        return {
            'total_score': total_score,
            'semantic_sim': semantic_sim,
            'technical_sim': technical_sim, 
            'diversity_score': diversity_score,
            'label_match': label_match
        }
    
    def retrieve_demonstrations(self, query_patent: Patent, query_embedding: torch.Tensor,
                              candidate_patents: List[Patent], candidate_embeddings: List[torch.Tensor],
                              k: int) -> List[RetrievedDemo]:
        """
        Retrieve top-k demonstrations using greedy selection
        Implements Algorithm 1 Phase 2: Retrieval-Augmented Demonstration Selection
        """
        if len(candidate_patents) != len(candidate_embeddings):
            raise ValueError("Mismatch between patents and embeddings")
        
        # Adapt weights based on query characteristics (Equation 10)
        adapted_weights = self.adaptive_weighter.adapt_weights(query_patent, candidate_patents)
        
        retrieved_demos = []
        retrieved_patents = []
        retrieved_embeddings = []
        used_indices = set()
        
        # Greedy selection of demonstrations
        for step in range(min(k, len(candidate_patents))):
            best_score = -float('inf')
            best_idx = -1
            best_scores = None
            
            # Evaluate all remaining candidates
            for idx, (candidate_patent, candidate_embedding) in enumerate(zip(candidate_patents, candidate_embeddings)):
                if idx in used_indices:
                    continue
                
                scores = self.compute_multi_faceted_score(
                    query_patent, query_embedding,
                    candidate_patent, candidate_embedding,
                    retrieved_patents, retrieved_embeddings,
                    adapted_weights
                )
                
                if scores['total_score'] > best_score:
                    best_score = scores['total_score']
                    best_idx = idx
                    best_scores = scores
            
            if best_idx >= 0:
                # Add best candidate
                best_patent = candidate_patents[best_idx]
                best_embedding = candidate_embeddings[best_idx]
                
                demo = RetrievedDemo(
                    patent=best_patent,
                    score=best_scores['total_score'],
                    semantic_sim=best_scores['semantic_sim'],
                    technical_sim=best_scores['technical_sim'],
                    diversity_score=best_scores['diversity_score']
                )
                
                retrieved_demos.append(demo)
                retrieved_patents.append(best_patent)
                retrieved_embeddings.append(best_embedding)
                used_indices.add(best_idx)
        
        return retrieved_demos
    
    def order_demonstrations(self, retrieved_demos: List[RetrievedDemo]) -> List[RetrievedDemo]:
        """
        Order demonstrations balancing relevance and diversity (Equation 12)
        D_k^ordered = arg sort[λ1·score - λ2·∑sim_sem]
        """
        lambda1 = 1.0  # Relevance weight
        lambda2 = 0.3  # Diversity penalty weight
        
        if len(retrieved_demos) <= 1:
            return retrieved_demos
        
        ordered_demos = []
        remaining_demos = retrieved_demos.copy()
        
        # Start with highest scoring demonstration
        best_demo = max(remaining_demos, key=lambda x: x.score)
        ordered_demos.append(best_demo)
        remaining_demos.remove(best_demo)
        
        # Order remaining demonstrations
        while remaining_demos:
            best_score = -float('inf')
            best_demo = None
            
            for demo in remaining_demos:
                # Compute diversity penalty (sum of similarities to already selected)
                diversity_penalty = 0.0
                for selected_demo in ordered_demos:
                    # Use semantic similarity as proxy (could be computed from embeddings)
                    sim = demo.semantic_sim  # This is a simplification
                    diversity_penalty += sim
                
                # Combined ordering score
                ordering_score = lambda1 * demo.score - lambda2 * diversity_penalty
                
                if ordering_score > best_score:
                    best_score = ordering_score
                    best_demo = demo
            
            if best_demo:
                ordered_demos.append(best_demo)
                remaining_demos.remove(best_demo)
            else:
                break
        
        return ordered_demos
    
    def retrieve_for_query(self, query_patent: Patent, encoder, 
                          exclude_ids: Optional[Set[str]] = None) -> List[Patent]:
        """
        Main interface for retrieving demonstrations for a query patent
        """
        if exclude_ids is None:
            exclude_ids = set()
        
        # Get query embedding
        encoder.eval()
        with torch.no_grad():
            tokens = self.dataset.tokenize_text(query_patent.abstract)
            device = next(encoder.parameters()).device
            input_ids = tokens['input_ids'].unsqueeze(0).to(device)
            attention_mask = tokens['attention_mask'].unsqueeze(0).to(device)
            
            query_embedding = encoder.encode(input_ids, attention_mask).squeeze(0)
        
        # Prepare candidate patents and embeddings
        candidate_patents = []
        candidate_embeddings = []
        
        for patent in self.dataset.patents:
            if patent.id in exclude_ids or patent.id == query_patent.id:
                continue
                
            with torch.no_grad():
                tokens = self.dataset.tokenize_text(patent.abstract)
                input_ids = tokens['input_ids'].unsqueeze(0).to(device)
                attention_mask = tokens['attention_mask'].unsqueeze(0).to(device)
                
                embedding = encoder.encode(input_ids, attention_mask).squeeze(0)
                candidate_embeddings.append(embedding)
                candidate_patents.append(patent)
        
        # Retrieve demonstrations
        retrieved_demos = self.retrieve_demonstrations(
            query_patent, query_embedding, candidate_patents, 
            candidate_embeddings, self.config.num_demonstrations
        )
        
        # Order demonstrations
        ordered_demos = self.order_demonstrations(retrieved_demos)
        
        # Return ordered patents
        return [demo.patent for demo in ordered_demos]