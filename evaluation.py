"""
Evaluation utilities for patent classification framework
Implements metrics and statistical testing from Section 5.1
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from scipy import stats
from collections import defaultdict, Counter
import sklearn.metrics as sk_metrics
from dataclasses import dataclass
import math


@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    macro_f1: float
    micro_f1: float
    lrap: float
    coverage_error: float
    macro_f1_std: float
    micro_f1_std: float
    lrap_std: float
    coverage_error_std: float
    category_scores: Dict[str, Dict[str, float]]
    statistical_tests: Dict[str, Any]


class MultiLabelEvaluator:
    """
    Multi-label evaluation metrics implementation
    Implements Equations 27-30 from the methodology
    """
    
    def __init__(self, categories: List[str]):
        self.categories = categories
        self.num_categories = len(categories)
        self.category_to_idx = {cat: idx for idx, cat in enumerate(categories)}
    
    def labels_to_binary_matrix(self, label_lists: List[List[str]]) -> np.ndarray:
        """Convert list of label lists to binary matrix"""
        binary_matrix = np.zeros((len(label_lists), self.num_categories))
        
        for i, labels in enumerate(label_lists):
            for label in labels:
                if label in self.category_to_idx:
                    j = self.category_to_idx[label]
                    binary_matrix[i, j] = 1.0
        
        return binary_matrix
    
    def compute_macro_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, List[float]]:
        """
        Compute Macro-F1 score (Equation 27)
        Macro-F1 = (1/K) * Σ F1_j = (1/K) * Σ (2 * P_j * R_j) / (P_j + R_j)
        """
        f1_scores = []
        
        for j in range(self.num_categories):
            y_true_j = y_true[:, j]
            y_pred_j = y_pred[:, j]
            
            # True positives, false positives, false negatives
            tp = np.sum((y_pred_j == 1) & (y_true_j == 1))
            fp = np.sum((y_pred_j == 1) & (y_true_j == 0))
            fn = np.sum((y_pred_j == 0) & (y_true_j == 1))
            
            # Precision and recall for category j
            precision_j = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall_j = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # F1 score for category j
            f1_j = (2 * precision_j * recall_j) / (precision_j + recall_j) if (precision_j + recall_j) > 0 else 0.0
            f1_scores.append(f1_j)
        
        # Macro-F1 (equal weighting across categories)
        macro_f1 = np.mean(f1_scores)
        
        return macro_f1, f1_scores
    
    def compute_micro_f1(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Micro-F1 score (Equation 28)
        Micro-F1 = 2 * P_micro * R_micro / (P_micro + R_micro)
        """
        # Aggregate across all categories
        tp_micro = np.sum((y_pred == 1) & (y_true == 1))
        fp_micro = np.sum((y_pred == 1) & (y_true == 0))
        fn_micro = np.sum((y_pred == 0) & (y_true == 1))
        
        # Micro-averaged precision and recall
        precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
        recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
        
        # Micro-F1
        micro_f1 = (2 * precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0
        
        return micro_f1
    
    def compute_label_ranking_average_precision(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Compute Label Ranking Average Precision (Equation 29)
        LRAP = (1/N) * Σ (1/|Y_i|) * Σ (|L_ij| / rank_ij)
        """
        if y_scores is None:
            # If no scores provided, use binary predictions as scores
            y_scores = y_pred.astype(float)
        
        N = y_true.shape[0]
        lrap_scores = []
        
        for i in range(N):
            y_true_i = y_true[i]
            y_scores_i = y_scores[i]
            
            # Get true label indices
            true_label_indices = np.where(y_true_i == 1)[0]
            
            if len(true_label_indices) == 0:
                continue
            
            # Rank scores (higher scores get lower ranks)
            score_ranks = stats.rankdata(-y_scores_i, method='ordinal')
            
            # Compute LRAP for sample i
            lrap_i = 0.0
            for j in true_label_indices:
                rank_ij = score_ranks[j]
                
                # Count true labels with rank <= rank_ij
                l_ij = np.sum([
                    1 for k in true_label_indices 
                    if score_ranks[k] <= rank_ij
                ])
                
                lrap_i += l_ij / rank_ij
            
            lrap_i /= len(true_label_indices)
            lrap_scores.append(lrap_i)
        
        return np.mean(lrap_scores) if lrap_scores else 0.0
    
    def compute_coverage_error(self, y_true: np.ndarray, y_scores: np.ndarray) -> float:
        """
        Compute Coverage Error (Equation 30)
        Coverage = (1/N) * Σ max(rank_ij) - 1 for j ∈ Y_i
        """
        if y_scores is None:
            y_scores = y_pred.astype(float)
        
        N = y_true.shape[0]
        coverage_errors = []
        
        for i in range(N):
            y_true_i = y_true[i]
            y_scores_i = y_scores[i]
            
            # Get true label indices
            true_label_indices = np.where(y_true_i == 1)[0]
            
            if len(true_label_indices) == 0:
                continue
            
            # Rank scores (higher scores get lower ranks)
            score_ranks = stats.rankdata(-y_scores_i, method='ordinal')
            
            # Find maximum rank among true labels
            max_rank = max(score_ranks[j] for j in true_label_indices)
            
            # Coverage error for sample i
            coverage_error_i = max_rank - 1
            coverage_errors.append(coverage_error_i)
        
        return np.mean(coverage_errors) if coverage_errors else 0.0
    
    def compute_category_wise_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute precision, recall, F1 for each category"""
        category_metrics = {}
        
        for j, category in enumerate(self.categories):
            y_true_j = y_true[:, j]
            y_pred_j = y_pred[:, j]
            
            # Compute metrics for category j
            tp = np.sum((y_pred_j == 1) & (y_true_j == 1))
            fp = np.sum((y_pred_j == 1) & (y_true_j == 0))
            fn = np.sum((y_pred_j == 0) & (y_true_j == 1))
            tn = np.sum((y_pred_j == 0) & (y_true_j == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Support (number of true instances)
            support = int(np.sum(y_true_j))
            
            category_metrics[category] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }
        
        return category_metrics


class StatisticalTestsModule:
    """
    Statistical significance testing
    Implements Bonferroni correction and paired t-tests
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
    
    def paired_t_test(self, scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Perform paired t-test between two sets of scores"""
        if len(scores1) != len(scores2):
            raise ValueError("Score lists must have same length for paired test")
        
        if len(scores1) < 2:
            return {'t_statistic': 0.0, 'p_value': 1.0}
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        }
    
    def bonferroni_correction(self, p_values: List[float], num_comparisons: int) -> List[float]:
        """Apply Bonferroni correction for multiple comparisons"""
        corrected_alpha = self.alpha / num_comparisons
        corrected_p_values = [min(1.0, p * num_comparisons) for p in p_values]
        
        return corrected_p_values
    
    def compute_effect_size(self, scores1: List[float], scores2: List[float]) -> float:
        """Compute Cohen's d effect size"""
        if len(scores1) < 2 or len(scores2) < 2:
            return 0.0
        
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(scores1), len(scores2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        
        return float(cohens_d)


class FewShotEvaluationSuite:
    """
    Complete evaluation suite for few-shot learning
    """
    
    def __init__(self, categories: List[str], alpha: float = 0.0083):
        self.categories = categories
        self.metric_evaluator = MultiLabelEvaluator(categories)
        self.statistical_tests = StatisticalTestsModule(alpha)
        
    def evaluate_method(self, predictions: List[List[str]], 
                       ground_truth: List[List[str]],
                       prediction_scores: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Evaluate a single method"""
        # Convert to binary matrices
        y_true = self.metric_evaluator.labels_to_binary_matrix(ground_truth)
        y_pred = self.metric_evaluator.labels_to_binary_matrix(predictions)
        
        # Convert scores if provided
        y_scores = None
        if prediction_scores is not None:
            y_scores = np.array(prediction_scores)
        
        # Compute metrics
        macro_f1, category_f1s = self.metric_evaluator.compute_macro_f1(y_true, y_pred)
        micro_f1 = self.metric_evaluator.compute_micro_f1(y_true, y_pred)
        lrap = self.metric_evaluator.compute_label_ranking_average_precision(y_true, y_scores)
        coverage_error = self.metric_evaluator.compute_coverage_error(y_true, y_scores)
        
        # Category-wise metrics
        category_metrics = self.metric_evaluator.compute_category_wise_metrics(y_true, y_pred)
        
        return {
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'lrap': lrap,
            'coverage_error': coverage_error,
            'category_f1s': category_f1s,
            'category_metrics': category_metrics
        }
    
    def evaluate_episodes(self, method_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Evaluate multiple methods across episodes
        method_results: {method_name: [episode_results]}
        """
        evaluation_results = {}
        
        # Evaluate each method
        for method_name, episodes in method_results.items():
            # Aggregate predictions and ground truth across episodes
            all_predictions = []
            all_ground_truth = []
            episode_scores = defaultdict(list)
            
            for episode in episodes:
                all_predictions.extend(episode['predictions'])
                all_ground_truth.extend(episode['ground_truth'])
                
                # Compute episode-level metrics
                episode_result = self.evaluate_method(
                    episode['predictions'], 
                    episode['ground_truth']
                )
                
                for metric in ['macro_f1', 'micro_f1', 'lrap', 'coverage_error']:
                    episode_scores[metric].append(episode_result[metric])
            
            # Overall evaluation
            overall_result = self.evaluate_method(all_predictions, all_ground_truth)
            
            # Compute statistics across episodes
            method_stats = {}
            for metric in ['macro_f1', 'micro_f1', 'lrap', 'coverage_error']:
                scores = episode_scores[metric]
                method_stats[f'{metric}_mean'] = np.mean(scores)
                method_stats[f'{metric}_std'] = np.std(scores)
                method_stats[f'{metric}_episodes'] = scores
            
            evaluation_results[method_name] = {
                'overall_metrics': overall_result,
                'episode_statistics': method_stats
            }
        
        return evaluation_results
    
    def compare_methods(self, method_results: Dict[str, List[Dict[str, Any]]], 
                       baseline_method: str = None) -> Dict[str, Any]:
        """
        Compare methods with statistical significance testing
        """
        evaluation_results = self.evaluate_episodes(method_results)
        
        if baseline_method is None:
            # Use first method as baseline
            baseline_method = list(method_results.keys())[0]
        
        comparison_results = {}
        baseline_scores = evaluation_results[baseline_method]['episode_statistics']
        
        # Compare each method against baseline
        method_names = [name for name in method_results.keys() if name != baseline_method]
        p_values = []
        
        for method_name in method_names:
            method_scores = evaluation_results[method_name]['episode_statistics']
            method_comparison = {}
            
            for metric in ['macro_f1', 'micro_f1', 'lrap', 'coverage_error']:
                baseline_episodes = baseline_scores[f'{metric}_episodes']
                method_episodes = method_scores[f'{metric}_episodes']
                
                # Paired t-test
                test_result = self.statistical_tests.paired_t_test(
                    method_episodes, baseline_episodes
                )
                
                # Effect size
                effect_size = self.statistical_tests.compute_effect_size(
                    method_episodes, baseline_episodes
                )
                
                method_comparison[metric] = {
                    'improvement': method_scores[f'{metric}_mean'] - baseline_scores[f'{metric}_mean'],
                    'improvement_pct': ((method_scores[f'{metric}_mean'] - baseline_scores[f'{metric}_mean']) / baseline_scores[f'{metric}_mean']) * 100,
                    't_statistic': test_result['t_statistic'],
                    'p_value': test_result['p_value'],
                    'effect_size': effect_size
                }
                
                p_values.append(test_result['p_value'])
            
            comparison_results[method_name] = method_comparison
        
        # Apply Bonferroni correction
        if p_values:
            num_comparisons = len(method_names) * 4  # 4 metrics
            corrected_p_values = self.statistical_tests.bonferroni_correction(p_values, num_comparisons)
            
            # Update corrected p-values
            p_idx = 0
            for method_name in method_names:
                for metric in ['macro_f1', 'micro_f1', 'lrap', 'coverage_error']:
                    comparison_results[method_name][metric]['corrected_p_value'] = corrected_p_values[p_idx]
                    p_idx += 1
        
        return {
            'baseline_method': baseline_method,
            'evaluation_results': evaluation_results,
            'comparisons': comparison_results,
            'bonferroni_alpha': self.statistical_tests.alpha / num_comparisons if p_values else self.statistical_tests.alpha
        }
    
    def generate_evaluation_report(self, comparison_results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report"""
        report = []
        report.append("=== MULTI-LABEL PATENT CLASSIFICATION EVALUATION REPORT ===\n")
        
        baseline = comparison_results['baseline_method']
        evaluations = comparison_results['evaluation_results']
        comparisons = comparison_results['comparisons']
        
        # Baseline results
        report.append(f"BASELINE METHOD: {baseline}")
        baseline_stats = evaluations[baseline]['episode_statistics']
        report.append(f"  Macro-F1: {baseline_stats['macro_f1_mean']:.3f} ± {baseline_stats['macro_f1_std']:.3f}")
        report.append(f"  Micro-F1: {baseline_stats['micro_f1_mean']:.3f} ± {baseline_stats['micro_f1_std']:.3f}")
        report.append(f"  LRAP: {baseline_stats['lrap_mean']:.3f} ± {baseline_stats['lrap_std']:.3f}")
        report.append(f"  Coverage Error: {baseline_stats['coverage_error_mean']:.3f} ± {baseline_stats['coverage_error_std']:.3f}")
        report.append("")
        
        # Method comparisons
        for method_name, method_comparisons in comparisons.items():
            method_stats = evaluations[method_name]['episode_statistics']
            report.append(f"METHOD: {method_name}")
            
            for metric in ['macro_f1', 'micro_f1', 'lrap', 'coverage_error']:
                stats = method_stats[f'{metric}_mean']
                std = method_stats[f'{metric}_std']
                improvement = method_comparisons[metric]['improvement']
                improvement_pct = method_comparisons[metric]['improvement_pct']
                p_value = method_comparisons[metric]['corrected_p_value']
                effect_size = method_comparisons[metric]['effect_size']
                
                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                
                report.append(f"  {metric.replace('_', '-').title()}: {stats:.3f} ± {std:.3f}")
                report.append(f"    Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%) {significance}")
                report.append(f"    p-value: {p_value:.6f}, Effect size: {effect_size:.2f}")
            
            report.append("")
        
        return "\n".join(report)


def run_ablation_analysis(baseline_results: List[Dict[str, Any]], 
                         component_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Run ablation analysis to measure component contributions
    """
    evaluator = FewShotEvaluationSuite(["VTOL & Hybrid Flight", "Surveillance & Mapping"])  # Simplified for demo
    
    # Prepare method results
    method_results = {'Full Framework': baseline_results}
    method_results.update(component_results)
    
    # Compare methods
    comparison_results = evaluator.compare_methods(method_results, 'Full Framework')
    
    # Compute contribution scores
    contribution_scores = {}
    full_framework_score = comparison_results['evaluation_results']['Full Framework']['episode_statistics']['macro_f1_mean']
    
    for component_name, comparisons in comparison_results['comparisons'].items():
        # Contribution = Performance drop when component is removed
        component_score = comparison_results['evaluation_results'][component_name]['episode_statistics']['macro_f1_mean']
        contribution = full_framework_score - component_score
        contribution_pct = (contribution / full_framework_score) * 100
        
        contribution_scores[component_name] = {
            'contribution_absolute': contribution,
            'contribution_percentage': contribution_pct,
            'component_score': component_score,
            'statistical_significance': comparisons['macro_f1']['corrected_p_value']
        }
    
    return {
        'full_framework_score': full_framework_score,
        'component_contributions': contribution_scores,
        'comparison_results': comparison_results
    }