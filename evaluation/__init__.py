"""
Unified Evaluation Framework for Knowledge Graph Models
Provides standardized evaluation protocols for fair comparison
"""

from .evaluator import LinkPredictionEvaluator, FilteredRankingEvaluator
from .metrics import compute_mrr, compute_hits_at_k, compute_ranking_metrics, format_results, create_results_table

# Optional imports (won't break if utils not available)
try:
    from .utils import create_evaluation_dataset, load_evaluation_data
except ImportError:
    create_evaluation_dataset = None
    load_evaluation_data = None

__all__ = [
    'LinkPredictionEvaluator', 
    'FilteredRankingEvaluator',
    'compute_mrr', 
    'compute_hits_at_k', 
    'compute_ranking_metrics',
    'format_results',
    'create_results_table'
]

# Add utils functions if available
if create_evaluation_dataset is not None:
    __all__.extend(['create_evaluation_dataset', 'load_evaluation_data'])