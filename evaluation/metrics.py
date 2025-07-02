"""
Standard metrics for knowledge graph link prediction evaluation
"""

import numpy as np
from typing import Dict, List, Union
import torch


def compute_mrr(ranks: Union[np.ndarray, List[float]]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR)
    
    Args:
        ranks: Array of ranks [num_samples]
        
    Returns:
        mrr: Mean reciprocal rank
    """
    if isinstance(ranks, list):
        ranks = np.array(ranks)
    
    reciprocal_ranks = 1.0 / ranks
    return float(reciprocal_ranks.mean())


def compute_hits_at_k(ranks: Union[np.ndarray, List[float]], k: int) -> float:
    """
    Compute Hits@K metric
    
    Args:
        ranks: Array of ranks [num_samples]
        k: Cutoff value for hits
        
    Returns:
        hits_at_k: Proportion of ranks <= k
    """
    if isinstance(ranks, list):
        ranks = np.array(ranks)
    
    hits = (ranks <= k).astype(float)
    return float(hits.mean())


def compute_ranking_metrics(ranks: Union[np.ndarray, List[float]], 
                          ks: List[int] = [1, 3, 10]) -> Dict[str, float]:
    """
    Compute standard ranking metrics for link prediction
    
    Args:
        ranks: Array of ranks [num_samples]
        ks: List of k values for Hits@K computation
        
    Returns:
        metrics: Dictionary with MRR and Hits@K values
    """
    if isinstance(ranks, list):
        ranks = np.array(ranks)
    
    metrics = {}
    
    # Mean Reciprocal Rank
    metrics['mrr'] = compute_mrr(ranks)
    
    # Hits@K for each k
    for k in ks:
        metrics[f'hits_at_{k}'] = compute_hits_at_k(ranks, k)
    
    # Additional statistics
    metrics['mean_rank'] = float(ranks.mean())
    metrics['median_rank'] = float(np.median(ranks))
    metrics['num_samples'] = len(ranks)
    
    return metrics


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Compute confidence interval for metric values
    
    Args:
        values: Array of metric values
        confidence: Confidence level (default: 0.95)
        
    Returns:
        (lower_bound, upper_bound): Confidence interval bounds
    """
    from scipy import stats
    
    n = len(values)
    mean = values.mean()
    std_err = stats.sem(values)
    
    # t-distribution for confidence interval
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    margin_error = t_value * std_err
    
    return (mean - margin_error, mean + margin_error)


def bootstrap_metric(ranks: np.ndarray, metric_fn, num_bootstrap: int = 1000, 
                    confidence: float = 0.95) -> Dict[str, float]:
    """
    Compute bootstrap confidence intervals for metrics
    
    Args:
        ranks: Array of ranks
        metric_fn: Function to compute metric (e.g., compute_mrr)
        num_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        bootstrap_results: Dictionary with metric estimate and confidence interval
    """
    n = len(ranks)
    bootstrap_values = []
    
    for _ in range(num_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(ranks, size=n, replace=True)
        bootstrap_metric = metric_fn(bootstrap_sample)
        bootstrap_values.append(bootstrap_metric)
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_values, lower_percentile)
    upper_bound = np.percentile(bootstrap_values, upper_percentile)
    
    return {
        'estimate': float(bootstrap_values.mean()),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound),
        'std': float(bootstrap_values.std())
    }


def format_results(metrics: Dict[str, float], precision: int = 4) -> str:
    """
    Format evaluation results for display
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        formatted_string: Formatted results string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("Link Prediction Evaluation Results")
    lines.append("=" * 50)
    
    # Main metrics
    if 'mrr' in metrics:
        lines.append(f"MRR: {metrics['mrr']:.{precision}f}")
    
    # Hits@K metrics
    for key in sorted(metrics.keys()):
        if key.startswith('hits_at_'):
            k = key.split('_')[-1]
            lines.append(f"Hits@{k}: {metrics[key]:.{precision}f}")
    
    # Additional statistics
    if 'mean_rank' in metrics:
        lines.append(f"Mean Rank: {metrics['mean_rank']:.2f}")
    if 'median_rank' in metrics:
        lines.append(f"Median Rank: {metrics['median_rank']:.2f}")
    if 'num_samples' in metrics:
        lines.append(f"Num Samples: {metrics['num_samples']}")
    
    lines.append("=" * 50)
    
    return "\\n".join(lines)


def compare_metrics(results1: Dict[str, float], results2: Dict[str, float], 
                   model1_name: str = "Model 1", model2_name: str = "Model 2") -> str:
    """
    Compare metrics between two models
    
    Args:
        results1: Metrics for first model
        results2: Metrics for second model
        model1_name: Name of first model
        model2_name: Name of second model
        
    Returns:
        comparison_string: Formatted comparison
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"Model Comparison: {model1_name} vs {model2_name}")
    lines.append("=" * 60)
    
    # Compare main metrics
    metrics_to_compare = ['mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10']
    
    for metric in metrics_to_compare:
        if metric in results1 and metric in results2:
            val1 = results1[metric]
            val2 = results2[metric]
            diff = val2 - val1
            pct_change = (diff / val1) * 100 if val1 != 0 else float('inf')
            
            winner = model2_name if val2 > val1 else model1_name
            
            lines.append(f"{metric.upper()}:")
            lines.append(f"  {model1_name}: {val1:.4f}")
            lines.append(f"  {model2_name}: {val2:.4f}")
            lines.append(f"  Difference: {diff:+.4f} ({pct_change:+.2f}%)")
            lines.append(f"  Winner: {winner}")
            lines.append("")
    
    lines.append("=" * 60)
    
    return "\\n".join(lines)


def aggregate_results(results_dict: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate results across multiple models for comparison
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        
    Returns:
        aggregated: Aggregated statistics and rankings
    """
    if not results_dict:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for model_results in results_dict.values():
        all_metrics.update(model_results.keys())
    
    aggregated = {
        'individual': results_dict,
        'rankings': {},
        'best_models': {}
    }
    
    # Rank models for each metric
    for metric in all_metrics:
        if metric == 'mean_rank':  # Lower is better for mean rank
            metric_values = [(name, results.get(metric, float('inf'))) 
                           for name, results in results_dict.items() 
                           if metric in results]
            metric_values.sort(key=lambda x: x[1])
        else:  # Higher is better for MRR and Hits@K
            metric_values = [(name, results.get(metric, 0.0)) 
                           for name, results in results_dict.items() 
                           if metric in results]
            metric_values.sort(key=lambda x: x[1], reverse=True)
        
        # Store rankings
        aggregated['rankings'][metric] = [name for name, _ in metric_values]
        
        # Store best model for this metric
        if metric_values:
            aggregated['best_models'][metric] = metric_values[0][0]
    
    return aggregated


def create_results_table(results_dict: Dict[str, Dict[str, float]], 
                        metrics: List[str] = ['mrr', 'hits_at_1', 'hits_at_3', 'hits_at_10']) -> str:
    """
    Create a formatted table of results for multiple models
    
    Args:
        results_dict: Dictionary mapping model names to metrics
        metrics: List of metrics to include in table
        
    Returns:
        table_string: Formatted table as string
    """
    if not results_dict:
        return "No results to display"
    
    # Prepare table data
    model_names = list(results_dict.keys())
    
    # Create header
    header = "| Model |" + "".join(f" {metric.upper()} |" for metric in metrics)
    separator = "|" + "|".join(["-" * (len(name) + 2) for name in ["Model"] + [m.upper() for m in metrics]]) + "|"
    
    # Create rows
    rows = []
    for model_name in model_names:
        model_results = results_dict[model_name]
        row = f"| {model_name} |"
        
        for metric in metrics:
            if metric in model_results:
                value = model_results[metric]
                row += f" {value:.4f} |"
            else:
                row += " N/A |"
        
        rows.append(row)
    
    # Combine table
    table_lines = [header, separator] + rows
    
    return "\\n".join(table_lines)