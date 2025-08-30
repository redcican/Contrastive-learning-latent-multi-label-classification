"""
Chain-of-Thought Multi-Label Reasoning Module
Implements the methodology from Section 4.4 of the paper
"""

import openai
import json
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

from data_utils import Patent
from few_shot_predictor import PromptComponents


@dataclass
class ReasoningStep:
    """Individual reasoning step for a category"""
    category: str
    technological_features: List[str]
    evidence_for: List[str]
    evidence_against: List[str]
    comparison_with_demos: str
    final_decision: bool
    confidence_score: float


@dataclass
class CoTResult:
    """Complete chain-of-thought reasoning result"""
    query_patent_id: str
    reasoning_steps: List[ReasoningStep]
    final_predictions: List[str]
    overall_confidence: float
    reasoning_quality: str


class GPTInterface:
    """
    Interface for GPT-4o API calls with patent-specific configuration
    Implements the API parameters from Section 4.4
    """
    
    def __init__(self, config):
        self.config = config
        # GPT-4o API configuration (from paper methodology)
        self.model = config.model  # "gpt-4o"
        self.temperature = config.temperature  # 0.3
        self.max_tokens = config.max_tokens  # 2048
        self.top_p = config.top_p  # 0.9
        self.frequency_penalty = config.frequency_penalty  # 0.2
        self.response_format = config.response_format  # {"type": "json_object"}
        
        # Rate limiting
        self.calls_per_minute = 50
        self.call_times = []
        
    def _rate_limit_check(self):
        """Ensure rate limiting compliance"""
        current_time = time.time()
        
        # Remove calls older than 1 minute
        self.call_times = [t for t in self.call_times if current_time - t < 60]
        
        # Check if we need to wait
        if len(self.call_times) >= self.calls_per_minute:
            sleep_time = 60 - (current_time - self.call_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.call_times = []
        
        self.call_times.append(current_time)
    
    def call_gpt4o(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Make API call to GPT-4o with specified parameters
        """
        self._rate_limit_check()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                response_format=self.response_format
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON response
            if self.response_format.get("type") == "json_object":
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    logging.error(f"Failed to parse JSON response: {content}")
                    return {"error": "JSON parsing failed", "raw_content": content}
            
            return {"content": content}
            
        except Exception as e:
            logging.error(f"GPT-4o API call failed: {str(e)}")
            return {"error": str(e)}


class ReasoningTemplateManager:
    """
    Manages reasoning templates for structured CoT prompts
    Implements the reasoning template from Equation 23-24
    """
    
    def __init__(self, categories: List[str]):
        self.categories = categories
    
    def create_system_prompt(self) -> str:
        """Create system prompt for patent classification task"""
        categories_list = ", ".join(self.categories)
        
        system_prompt = f"""You are an expert patent classifier specializing in unmanned aerial vehicle (UAV) technologies. Your task is to analyze patent abstracts and classify them into one or more technological categories.

Available categories: {categories_list}

Your analysis should:
1. Identify key technological features in the patent
2. Compare these features against demonstration examples
3. Evaluate evidence for and against each category
4. Make structured decisions with confidence scores
5. Handle inter-label dependencies (related technologies often co-occur)

Always respond in JSON format with structured reasoning for each category."""
        
        return system_prompt
    
    def create_reasoning_template(self, query_patent: Patent, 
                                demonstration_patents: List[Patent],
                                target_categories: List[str],
                                previously_assigned: List[str] = None) -> str:
        """
        Create structured reasoning template (Equation 23)
        Reason_template includes: Extract, Compare, Evaluate, Decide steps
        """
        if previously_assigned is None:
            previously_assigned = []
        
        # Format demonstrations
        demo_examples = []
        for i, demo in enumerate(demonstration_patents):
            demo_text = f"""
Example {i+1}:
Patent: {demo.abstract[:400]}...
Categories: {', '.join(demo.labels)}
"""
            demo_examples.append(demo_text)
        
        demonstrations_text = "\n".join(demo_examples)
        
        # Previously assigned context
        context_text = ""
        if previously_assigned:
            context_text = f"\nPreviously assigned categories for this patent: {', '.join(previously_assigned)}"
        
        # Create structured reasoning prompt
        reasoning_prompt = f"""
Based on the following examples, analyze the query patent and provide structured reasoning for each of the specified categories.

DEMONSTRATION EXAMPLES:
{demonstrations_text}

QUERY PATENT TO CLASSIFY:
Patent: {query_patent.abstract}

{context_text}

CATEGORIES TO EVALUATE: {', '.join(target_categories)}

For EACH category, provide reasoning following this structure:
1. EXTRACT: Identify key technological features in the query patent relevant to this category
2. COMPARE: Compare these features against the demonstration examples for this category  
3. EVALUATE: List evidence FOR and AGAINST assigning this category
4. DECIDE: Make a final decision (true/false) with confidence score (0-1)

Consider inter-label dependencies: if categories are technologically related, they may co-occur.

Respond in JSON format:
{{
  "reasoning_steps": [
    {{
      "category": "category_name",
      "technological_features": ["feature1", "feature2", ...],
      "evidence_for": ["evidence1", "evidence2", ...], 
      "evidence_against": ["evidence1", "evidence2", ...],
      "comparison_with_demos": "detailed comparison text",
      "final_decision": true/false,
      "confidence_score": 0.0-1.0
    }}
  ],
  "final_predictions": ["category1", "category2", ...],
  "overall_confidence": 0.0-1.0
}}
"""
        
        return reasoning_prompt


class ConditionalEvaluationEngine:
    """
    Handles conditional evaluation for inter-label dependencies
    Implements conditional reasoning from Equation 25
    """
    
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.category_dependencies = self._initialize_dependencies()
    
    def _initialize_dependencies(self) -> Dict[str, List[str]]:
        """Initialize category dependencies based on domain knowledge"""
        # Define UAV technology dependencies
        dependencies = {
            "VTOL & Hybrid Flight": ["Flight Control & Stability", "Endurance & Power Systems"],
            "Flight Control & Stability": ["Surveillance & Mapping", "VTOL & Hybrid Flight"],
            "Surveillance & Mapping": ["Flight Control & Stability", "Specialized Applications"],
            "Modular & Deployable": ["Logistics & Cargo", "Multi-Environment"],
            "Endurance & Power Systems": ["VTOL & Hybrid Flight", "Surveillance & Mapping"],
            "Bionic & Flapping Wing": ["Flight Control & Stability"],
            "Logistics & Cargo": ["Modular & Deployable", "Multi-Environment"],
            "Specialized Applications": ["Surveillance & Mapping"],
            "Multi-Environment": ["Modular & Deployable", "Structural & Materials"],
            "Structural & Materials": ["Multi-Environment", "Endurance & Power Systems"]
        }
        return dependencies
    
    def get_evaluation_order(self, categories: List[str]) -> List[str]:
        """Determine optimal order for category evaluation based on dependencies"""
        # Sort by number of dependencies (fewer dependencies evaluated first)
        def dependency_count(cat):
            return len(self.category_dependencies.get(cat, []))
        
        return sorted(categories, key=dependency_count)
    
    def get_conditional_context(self, category: str, assigned_labels: List[str]) -> str:
        """Get conditional context for category evaluation"""
        if not assigned_labels:
            return ""
        
        relevant_deps = []
        category_deps = self.category_dependencies.get(category, [])
        
        for assigned_label in assigned_labels:
            if assigned_label in category_deps:
                relevant_deps.append(assigned_label)
        
        if relevant_deps:
            context = f"Note: This patent has been assigned to {', '.join(relevant_deps)}, " \
                     f"which are technologically related to {category}. " \
                     f"Consider whether the innovations span these related areas."
            return context
        
        return ""


class CoTReasoningModule:
    """
    Main Chain-of-Thought reasoning module
    Implements the complete CoT process from Section 4.4
    """
    
    def __init__(self, config):
        self.config = config
        self.categories = config.categories
        
        # Initialize components
        self.gpt_interface = GPTInterface(config)
        self.template_manager = ReasoningTemplateManager(self.categories)
        self.conditional_engine = ConditionalEvaluationEngine(self.categories)
        
        # CoT weighting parameter (β from Equation 26)
        self.beta_cot = config.beta_cot  # 0.7
    
    def reason_single_category(self, query_patent: Patent,
                              demonstration_patents: List[Patent],
                              target_category: str,
                              previously_assigned: List[str] = None) -> Dict[str, Any]:
        """
        Perform reasoning for a single category with conditional evaluation
        Implements CoT_j steps from Equation 24
        """
        if previously_assigned is None:
            previously_assigned = []
        
        # Create reasoning prompt
        reasoning_prompt = self.template_manager.create_reasoning_template(
            query_patent, demonstration_patents, [target_category], previously_assigned
        )
        
        # Add conditional context
        conditional_context = self.conditional_engine.get_conditional_context(
            target_category, previously_assigned
        )
        if conditional_context:
            reasoning_prompt += f"\n\nADDITIONAL CONTEXT:\n{conditional_context}"
        
        # System prompt
        system_prompt = self.template_manager.create_system_prompt()
        
        # Call GPT-4o
        response = self.gpt_interface.call_gpt4o(reasoning_prompt, system_prompt)
        
        if "error" in response:
            return {
                "category": target_category,
                "reasoning": None,
                "decision": False,
                "confidence": 0.0,
                "error": response["error"]
            }
        
        # Extract reasoning for the target category
        reasoning_steps = response.get("reasoning_steps", [])
        category_reasoning = None
        
        for step in reasoning_steps:
            if step.get("category") == target_category:
                category_reasoning = step
                break
        
        if not category_reasoning:
            return {
                "category": target_category,
                "reasoning": None,
                "decision": False,
                "confidence": 0.0,
                "error": "No reasoning found for category"
            }
        
        return {
            "category": target_category,
            "reasoning": category_reasoning,
            "decision": category_reasoning.get("final_decision", False),
            "confidence": category_reasoning.get("confidence_score", 0.0),
            "error": None
        }
    
    def reason_all_categories(self, query_patent: Patent,
                             demonstration_patents: List[Patent],
                             batch_size: int = 3) -> CoTResult:
        """
        Perform reasoning for all categories with conditional dependencies
        Implements the complete CoT process with inter-label dependencies
        """
        # Determine evaluation order
        evaluation_order = self.conditional_engine.get_evaluation_order(self.categories)
        
        reasoning_results = []
        assigned_labels = []
        
        # Process categories in batches to manage API costs while maintaining dependencies
        for i in range(0, len(evaluation_order), batch_size):
            batch_categories = evaluation_order[i:i + batch_size]
            
            # Create batch reasoning prompt
            reasoning_prompt = self.template_manager.create_reasoning_template(
                query_patent, demonstration_patents, batch_categories, assigned_labels
            )
            
            system_prompt = self.template_manager.create_system_prompt()
            
            # Call GPT-4o for batch
            response = self.gpt_interface.call_gpt4o(reasoning_prompt, system_prompt)
            
            if "error" not in response:
                # Process batch results
                reasoning_steps = response.get("reasoning_steps", [])
                
                for category in batch_categories:
                    # Find reasoning for this category
                    category_reasoning = None
                    for step in reasoning_steps:
                        if step.get("category") == category:
                            category_reasoning = step
                            break
                    
                    if category_reasoning:
                        reasoning_step = ReasoningStep(
                            category=category,
                            technological_features=category_reasoning.get("technological_features", []),
                            evidence_for=category_reasoning.get("evidence_for", []),
                            evidence_against=category_reasoning.get("evidence_against", []),
                            comparison_with_demos=category_reasoning.get("comparison_with_demos", ""),
                            final_decision=category_reasoning.get("final_decision", False),
                            confidence_score=category_reasoning.get("confidence_score", 0.0)
                        )
                        
                        reasoning_results.append(reasoning_step)
                        
                        # Update assigned labels for conditional evaluation
                        if reasoning_step.final_decision:
                            assigned_labels.append(category)
        
        # Compute overall confidence
        if reasoning_results:
            overall_confidence = sum(step.confidence_score for step in reasoning_results) / len(reasoning_results)
        else:
            overall_confidence = 0.0
        
        # Create final result
        final_predictions = [step.category for step in reasoning_results if step.final_decision]
        
        cot_result = CoTResult(
            query_patent_id=query_patent.id,
            reasoning_steps=reasoning_results,
            final_predictions=final_predictions,
            overall_confidence=overall_confidence,
            reasoning_quality="high" if overall_confidence > 0.7 else "medium" if overall_confidence > 0.4 else "low"
        )
        
        return cot_result
    
    def get_category_probabilities(self, cot_result: CoTResult) -> Dict[str, float]:
        """Extract category probabilities from CoT reasoning"""
        probabilities = {}
        
        for step in cot_result.reasoning_steps:
            probabilities[step.category] = step.confidence_score if step.final_decision else 0.0
        
        return probabilities
    
    def combine_with_base_predictions(self, cot_probabilities: Dict[str, float],
                                    base_probabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Combine CoT predictions with base model predictions (Equation 26)
        ŷ_CoT^j = β * p_CoT(y_q^j = 1) + (1-β) * p(y_q^j = 1 | x_q, D_k)
        """
        combined_probabilities = {}
        
        for category in self.categories:
            cot_prob = cot_probabilities.get(category, 0.0)
            base_prob = base_probabilities.get(category, 0.0)
            
            # Weighted combination (Equation 26)
            combined_prob = self.beta_cot * cot_prob + (1 - self.beta_cot) * base_prob
            combined_probabilities[category] = combined_prob
        
        return combined_probabilities
    
    def predict_with_cot(self, query_patent: Patent,
                        demonstration_patents: List[Patent],
                        base_probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Main interface for CoT-enhanced prediction
        """
        # Perform CoT reasoning
        cot_result = self.reason_all_categories(query_patent, demonstration_patents)
        
        # Get CoT probabilities
        cot_probabilities = self.get_category_probabilities(cot_result)
        
        # Combine with base predictions
        final_probabilities = self.combine_with_base_predictions(cot_probabilities, base_probabilities)
        
        # Make final predictions using original thresholds (would be passed from few_shot_predictor)
        threshold = 0.5  # Simplified threshold
        final_predictions = [cat for cat, prob in final_probabilities.items() if prob > threshold]
        
        return {
            "cot_result": cot_result,
            "cot_probabilities": cot_probabilities,
            "base_probabilities": base_probabilities,
            "final_probabilities": final_probabilities,
            "final_predictions": final_predictions,
            "reasoning_quality": cot_result.reasoning_quality
        }


class CoTEvaluator:
    """Evaluate quality of chain-of-thought reasoning"""
    
    def __init__(self):
        self.quality_dimensions = ["coherence", "factual_accuracy", "decision_support"]
    
    def evaluate_reasoning_step(self, step: ReasoningStep, ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate quality of individual reasoning step"""
        # This would involve human expert evaluation in practice
        # For now, we provide automated heuristics
        
        scores = {}
        
        # Coherence: based on length and structure of reasoning
        coherence = min(1.0, len(step.evidence_for + step.evidence_against) / 4.0)
        scores["coherence"] = coherence
        
        # Factual accuracy: simplified based on technical features identified
        factual_accuracy = min(1.0, len(step.technological_features) / 3.0)
        scores["factual_accuracy"] = factual_accuracy
        
        # Decision support: alignment between evidence and decision
        evidence_balance = len(step.evidence_for) - len(step.evidence_against)
        decision_alignment = 1.0 if (evidence_balance > 0) == step.final_decision else 0.5
        scores["decision_support"] = decision_alignment
        
        return scores
    
    def evaluate_cot_result(self, result: CoTResult, ground_truth: List[str]) -> Dict[str, Any]:
        """Evaluate overall CoT result quality"""
        if not result.reasoning_steps:
            return {"overall_quality": 0.0, "dimension_scores": {}}
        
        dimension_scores = {dim: [] for dim in self.quality_dimensions}
        
        for step in result.reasoning_steps:
            step_scores = self.evaluate_reasoning_step(step, ground_truth)
            for dim, score in step_scores.items():
                dimension_scores[dim].append(score)
        
        # Average scores across dimensions
        avg_dimension_scores = {}
        for dim, scores in dimension_scores.items():
            avg_dimension_scores[dim] = sum(scores) / len(scores) if scores else 0.0
        
        # Overall quality
        overall_quality = sum(avg_dimension_scores.values()) / len(avg_dimension_scores)
        
        return {
            "overall_quality": overall_quality,
            "dimension_scores": avg_dimension_scores,
            "num_reasoning_steps": len(result.reasoning_steps)
        }