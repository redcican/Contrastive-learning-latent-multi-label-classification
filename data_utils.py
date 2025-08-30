"""
Data utilities for patent classification
Handles patent corpus loading, preprocessing, and few-shot episode construction
"""

import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import re


@dataclass
class Patent:
    """Patent data structure"""
    id: str
    abstract: str
    labels: List[str]
    year: int
    ipc_codes: List[str] = None
    citations: List[str] = None
    technical_terms: List[str] = None
    
    def __post_init__(self):
        if self.ipc_codes is None:
            self.ipc_codes = []
        if self.citations is None:
            self.citations = []
        if self.technical_terms is None:
            self.technical_terms = []


@dataclass 
class Episode:
    """Few-shot learning episode"""
    categories: List[str]
    support_set: List[Patent]  # K demonstrations per category
    query_set: List[Patent]    # Patents to classify
    k_shot: int
    episode_id: int


class PatentProcessor:
    """Patent text preprocessing and feature extraction"""
    
    def __init__(self):
        self.technical_terms = set()
        self._load_technical_vocabulary()
    
    def _load_technical_vocabulary(self):
        """Load domain-specific technical terms for UAV patents"""
        # Common UAV technical terms (in practice, this would be loaded from a file)
        uav_terms = {
            'quadcopter', 'multirotor', 'vtol', 'autopilot', 'gyroscope',
            'accelerometer', 'magnetometer', 'barometer', 'gps', 'lidar',
            'ultrasonic', 'obstacle avoidance', 'flight controller', 'esc',
            'brushless motor', 'propeller', 'battery', 'lipo', 'gimbal',
            'camera stabilization', 'fpv', 'telemetry', 'payload', 'drone',
            'unmanned aerial vehicle', 'autonomous flight', 'waypoint',
            'geofencing', 'return to home', 'failsafe', 'radio control',
            'servo', 'flight mode', 'attitude', 'altitude', 'navigation'
        }
        self.technical_terms.update(uav_terms)
    
    def clean_abstract(self, text: str) -> str:
        """Clean patent abstract text"""
        # Remove patent-specific boilerplate
        text = re.sub(r'\b(patent|invention|discloses?|describes?)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(fig\.?\s*\d+|figure\s+\d+)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\([^)]*\)', '', text)  # Remove parenthetical expressions
        
        # Clean special characters but preserve technical notation
        text = re.sub(r'[^\w\s\.\-\+\°\%]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms from patent text"""
        text_lower = text.lower()
        found_terms = []
        
        for term in self.technical_terms:
            if term in text_lower:
                found_terms.append(term)
        
        return found_terms
    
    def compute_complexity(self, text: str) -> float:
        """Compute technical complexity score based on term density"""
        technical_terms = self.extract_technical_terms(text)
        words = text.split()
        
        if not words:
            return 0.0
            
        return len(technical_terms) / len(words)


class PatentDataset:
    """Patent dataset with multi-label support"""
    
    def __init__(self, patents: List[Patent], categories: List[str], tokenizer: RobertaTokenizer = None):
        self.patents = patents
        self.categories = categories
        self.category_to_id = {cat: idx for idx, cat in enumerate(categories)}
        self.id_to_category = {idx: cat for idx, cat in enumerate(categories)}
        self.processor = PatentProcessor()
        
        if tokenizer is None:
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        else:
            self.tokenizer = tokenizer
            
        self._build_indices()
    
    def _build_indices(self):
        """Build category and patent indices for efficient retrieval"""
        self.category_patents = defaultdict(list)
        self.patent_categories = {}
        
        for patent in self.patents:
            # Map patents to categories
            patent_category_ids = []
            for label in patent.labels:
                if label in self.category_to_id:
                    cat_id = self.category_to_id[label]
                    self.category_patents[cat_id].append(patent)
                    patent_category_ids.append(cat_id)
            
            self.patent_categories[patent.id] = patent_category_ids
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of patents across categories"""
        distribution = {}
        for cat_id, patents in self.category_patents.items():
            category_name = self.id_to_category[cat_id]
            distribution[category_name] = len(patents)
        return distribution
    
    def get_patents_by_category(self, category: str) -> List[Patent]:
        """Get all patents for a specific category"""
        if category not in self.category_to_id:
            return []
        cat_id = self.category_to_id[category]
        return self.category_patents[cat_id]
    
    def encode_labels(self, labels: List[str]) -> np.ndarray:
        """Encode string labels to multi-hot binary vector"""
        binary_labels = np.zeros(len(self.categories), dtype=np.float32)
        for label in labels:
            if label in self.category_to_id:
                binary_labels[self.category_to_id[label]] = 1.0
        return binary_labels
    
    def decode_labels(self, binary_labels: np.ndarray, threshold: float = 0.5) -> List[str]:
        """Decode binary labels to category names"""
        predicted_categories = []
        for i, prob in enumerate(binary_labels):
            if prob >= threshold:
                predicted_categories.append(self.categories[i])
        return predicted_categories
    
    def tokenize_text(self, text: str, max_length: int = 512) -> Dict[str, torch.Tensor]:
        """Tokenize patent abstract text"""
        cleaned_text = self.processor.clean_abstract(text)
        
        encoding = self.tokenizer(
            cleaned_text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class FewShotEpisodeGenerator:
    """Generate few-shot learning episodes for evaluation"""
    
    def __init__(self, dataset: PatentDataset, config):
        self.dataset = dataset
        self.config = config
        self.categories = dataset.categories
        
        # Category frequency tiers for balanced sampling
        self.frequent_cats = set(config.frequent_categories)
        self.moderate_cats = set(config.moderate_categories)
        self.sparse_cats = set(config.sparse_categories)
    
    def sample_balanced_categories(self, n_way: int, seed: Optional[int] = None) -> List[str]:
        """Sample categories ensuring representation from each frequency tier"""
        if seed is not None:
            random.seed(seed)
            
        selected_categories = []
        
        # Ensure at least one from each tier
        if n_way >= 3:
            selected_categories.append(random.choice(list(self.frequent_cats)))
            selected_categories.append(random.choice(list(self.moderate_cats)))
            selected_categories.append(random.choice(list(self.sparse_cats)))
            remaining = n_way - 3
        else:
            remaining = n_way
        
        # Sample remaining categories
        all_cats = set(self.categories) - set(selected_categories)
        selected_categories.extend(random.sample(list(all_cats), remaining))
        
        return selected_categories
    
    def sample_support_patents(self, category: str, k_shot: int, 
                              exclude_ids: Set[str] = None) -> List[Patent]:
        """Sample k patents for support set from given category"""
        if exclude_ids is None:
            exclude_ids = set()
            
        category_patents = self.dataset.get_patents_by_category(category)
        
        # Filter out excluded patents
        available_patents = [p for p in category_patents if p.id not in exclude_ids]
        
        if len(available_patents) < k_shot:
            # If insufficient patents, sample with replacement
            sampled = random.choices(available_patents, k=k_shot)
        else:
            sampled = random.sample(available_patents, k_shot)
        
        return sampled
    
    def sample_query_patents(self, categories: List[str], query_size: int,
                            exclude_ids: Set[str] = None) -> List[Patent]:
        """Sample query patents containing various combinations of selected categories"""
        if exclude_ids is None:
            exclude_ids = set()
        
        # Find patents that have at least one of the selected categories
        query_candidates = []
        category_set = set(categories)
        
        for patent in self.dataset.patents:
            if patent.id in exclude_ids:
                continue
                
            patent_categories = set(patent.labels)
            if patent_categories & category_set:  # Intersection is non-empty
                query_candidates.append(patent)
        
        if len(query_candidates) < query_size:
            # Sample with replacement if insufficient candidates
            sampled = random.choices(query_candidates, k=query_size)
        else:
            sampled = random.sample(query_candidates, query_size)
        
        return sampled
    
    def generate_episode(self, n_way: int, k_shot: int, query_size: int, 
                        episode_id: int, seed: Optional[int] = None) -> Episode:
        """Generate a single few-shot episode"""
        if seed is not None:
            random.seed(seed + episode_id)  # Ensure different seed per episode
        
        # Sample categories
        selected_categories = self.sample_balanced_categories(n_way, seed)
        
        # Sample support set
        support_patents = []
        used_patent_ids = set()
        
        for category in selected_categories:
            category_support = self.sample_support_patents(
                category, k_shot, exclude_ids=used_patent_ids
            )
            support_patents.extend(category_support)
            used_patent_ids.update(p.id for p in category_support)
        
        # Sample query set
        query_patents = self.sample_query_patents(
            selected_categories, query_size, exclude_ids=used_patent_ids
        )
        
        return Episode(
            categories=selected_categories,
            support_set=support_patents,
            query_set=query_patents,
            k_shot=k_shot,
            episode_id=episode_id
        )
    
    def generate_episodes(self, n_way: int, k_shot: int, num_episodes: int,
                         query_size: int, seed: Optional[int] = None) -> List[Episode]:
        """Generate multiple episodes for evaluation"""
        episodes = []
        
        for episode_id in range(num_episodes):
            episode = self.generate_episode(
                n_way, k_shot, query_size, episode_id, seed
            )
            episodes.append(episode)
        
        return episodes


class PatentDataLoader:
    """Data loader for patent corpus"""
    
    def __init__(self, config):
        self.config = config
        self.processor = PatentProcessor()
        
    def load_patent_corpus(self, corpus_file: str) -> List[Patent]:
        """Load patent corpus from JSON file"""
        with open(corpus_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        patents = []
        for item in data:
            patent = Patent(
                id=item['id'],
                abstract=item['abstract'],
                labels=item['labels'],
                year=item['year'],
                ipc_codes=item.get('ipc_codes', []),
                citations=item.get('citations', []),
                technical_terms=self.processor.extract_technical_terms(item['abstract'])
            )
            patents.append(patent)
        
        return patents
    
    def split_temporal(self, patents: List[Patent]) -> Tuple[List[Patent], List[Patent]]:
        """Split patents by temporal cutoff"""
        train_patents = []
        test_patents = []
        
        train_start, train_end = self.config.train_years
        test_start, test_end = self.config.test_years
        
        for patent in patents:
            if train_start <= patent.year <= train_end:
                train_patents.append(patent)
            elif test_start <= patent.year <= test_end:
                test_patents.append(patent)
        
        return train_patents, test_patents
    
    def create_datasets(self, corpus_file: str) -> Tuple[PatentDataset, PatentDataset]:
        """Create train and test datasets"""
        # Load full corpus
        patents = self.load_patent_corpus(corpus_file)
        
        # Temporal split
        train_patents, test_patents = self.split_temporal(patents)
        
        # Create datasets
        train_dataset = PatentDataset(train_patents, self.config.categories)
        test_dataset = PatentDataset(test_patents, self.config.categories)
        
        print(f"Loaded {len(patents)} total patents")
        print(f"Train set: {len(train_patents)} patents ({self.config.train_years})")
        print(f"Test set: {len(test_patents)} patents ({self.config.test_years})")
        print("Category distribution:")
        
        train_dist = train_dataset.get_category_distribution()
        for category, count in train_dist.items():
            print(f"  {category}: {count}")
        
        return train_dataset, test_dataset


def collate_patents(batch: List[Patent]) -> Dict[str, Any]:
    """Collate function for patent dataloader"""
    patent_ids = [p.id for p in batch]
    abstracts = [p.abstract for p in batch]
    labels = [p.labels for p in batch]
    
    return {
        'patent_ids': patent_ids,
        'abstracts': abstracts,
        'labels': labels,
        'patents': batch
    }