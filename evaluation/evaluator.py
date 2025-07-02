"""
Standardized evaluator for knowledge graph link prediction
Implements filtered ranking protocol following research standards
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import time
from collections import defaultdict

from .metrics import compute_ranking_metrics


class FilteredRankingEvaluator:
    """
    Standard filtered ranking evaluator for knowledge graph link prediction
    Follows OGB and research community protocols
    """
    
    def __init__(self, filter_triples: Optional[List[torch.Tensor]] = None, 
                 batch_size: int = 1000, device: str = 'cuda'):
        """
        Initialize filtered ranking evaluator
        
        Args:
            filter_triples: List of triple tensors to filter during evaluation
                          (typically train + valid for test evaluation)
            batch_size: Batch size for evaluation
            device: Device for computation
        """
        self.filter_triples = filter_triples or []
        self.batch_size = batch_size
        self.device = device
        
        # Build filter sets for efficient lookup
        self._build_filter_sets()
        
    def _build_filter_sets(self):
        """Build efficient filter sets for O(1) lookup during evaluation"""
        self.head_filters = defaultdict(set)  # (r, t) -> set of valid heads
        self.tail_filters = defaultdict(set)  # (h, r) -> set of valid tails
        
        for triples in self.filter_triples:
            for triple in triples:
                h, r, t = triple.tolist()
                self.head_filters[(r, t)].add(h)
                self.tail_filters[(h, r)].add(t)
    
    def evaluate_model(self, model, test_triples: torch.Tensor, 
                      num_entities: int, evaluation_mode: str = 'both') -> Dict[str, float]:
        """
        Evaluate model using filtered ranking protocol
        
        Args:
            model: Model with score_heads() and score_tails() methods
            test_triples: Test triples [num_test, 3]
            num_entities: Total number of entities
            evaluation_mode: 'head', 'tail', or 'both'
            
        Returns:
            metrics: Dictionary with MRR, Hits@1, Hits@3, Hits@10
        """
        model.eval()
        all_ranks = []
        
        with torch.no_grad():
            if evaluation_mode in ['head', 'both']:
                head_ranks = self._evaluate_head_prediction(model, test_triples, num_entities)
                all_ranks.extend(head_ranks)
                
            if evaluation_mode in ['tail', 'both']:
                tail_ranks = self._evaluate_tail_prediction(model, test_triples, num_entities)
                all_ranks.extend(tail_ranks)
        
        # Compute metrics
        all_ranks = np.array(all_ranks)
        metrics = compute_ranking_metrics(all_ranks)
        
        return metrics
    
    def _evaluate_head_prediction(self, model, test_triples: torch.Tensor, 
                                 num_entities: int) -> List[float]:
        """Evaluate head prediction with filtered ranking"""
        ranks = []
        
        for i in tqdm(range(0, len(test_triples), self.batch_size), desc="Head prediction"):
            batch = test_triples[i:i + self.batch_size].to(self.device)
            batch_ranks = self._compute_head_ranks(model, batch, num_entities)
            ranks.extend(batch_ranks)
            
        return ranks
    
    def _evaluate_tail_prediction(self, model, test_triples: torch.Tensor, 
                                 num_entities: int) -> List[float]:
        """Evaluate tail prediction with filtered ranking"""
        ranks = []
        
        for i in tqdm(range(0, len(test_triples), self.batch_size), desc="Tail prediction"):
            batch = test_triples[i:i + self.batch_size].to(self.device)
            batch_ranks = self._compute_tail_ranks(model, batch, num_entities)
            ranks.extend(batch_ranks)
            
        return ranks
    
    def _compute_head_ranks(self, model, batch: torch.Tensor, num_entities: int) -> List[float]:
        """Compute filtered ranks for head prediction"""
        batch_size = batch.size(0)
        
        # Get scores for all possible heads
        relations = batch[:, 1]
        tails = batch[:, 2]
        true_heads = batch[:, 0]
        
        # Score all head candidates with chunking to avoid OOM
        chunk_size = 1000  # Process 1K entities at a time
        all_scores = torch.zeros(batch_size, num_entities, device=self.device)
        
        for chunk_start in range(0, num_entities, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_entities)
            chunk_entities = torch.arange(chunk_start, chunk_end, device=self.device)
            
            # Expand for batch
            chunk_entities_batch = chunk_entities.unsqueeze(0).expand(batch_size, -1)
            
            # Score this chunk
            chunk_scores = model.score_heads(relations, tails, chunk_entities_batch)
            all_scores[:, chunk_start:chunk_end] = chunk_scores
        
        ranks = []
        for i in range(batch_size):
            scores = all_scores[i]  # [num_entities]
            true_head = true_heads[i].item()
            r, t = relations[i].item(), tails[i].item()
            
            # Apply filtering
            filter_set = self.head_filters.get((r, t), set())
            filtered_scores = scores.clone()
            
            # Set scores of filter entities to -inf (except true head)
            for filter_head in filter_set:
                if filter_head != true_head:
                    filtered_scores[filter_head] = float('-inf')
            
            # Compute rank
            true_score = filtered_scores[true_head]
            rank = (filtered_scores > true_score).sum().item() + 1
            ranks.append(float(rank))
            
        return ranks
    
    def _compute_tail_ranks(self, model, batch: torch.Tensor, num_entities: int) -> List[float]:
        """Compute filtered ranks for tail prediction"""
        batch_size = batch.size(0)
        
        # Get scores for all possible tails
        heads = batch[:, 0]
        relations = batch[:, 1]
        true_tails = batch[:, 2]
        
        # Score all tail candidates with chunking to avoid OOM
        chunk_size = 1000  # Process 1K entities at a time
        all_scores = torch.zeros(batch_size, num_entities, device=self.device)
        
        for chunk_start in range(0, num_entities, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_entities)
            chunk_entities = torch.arange(chunk_start, chunk_end, device=self.device)
            
            # Expand for batch
            chunk_entities_batch = chunk_entities.unsqueeze(0).expand(batch_size, -1)
            
            # Score this chunk
            chunk_scores = model.score_tails(heads, relations, chunk_entities_batch)
            all_scores[:, chunk_start:chunk_end] = chunk_scores
        
        ranks = []
        for i in range(batch_size):
            scores = all_scores[i]  # [num_entities]
            true_tail = true_tails[i].item()
            h, r = heads[i].item(), relations[i].item()
            
            # Apply filtering
            filter_set = self.tail_filters.get((h, r), set())
            filtered_scores = scores.clone()
            
            # Set scores of filter entities to -inf (except true tail)
            for filter_tail in filter_set:
                if filter_tail != true_tail:
                    filtered_scores[filter_tail] = float('-inf')
            
            # Compute rank
            true_score = filtered_scores[true_tail]
            rank = (filtered_scores > true_score).sum().item() + 1
            ranks.append(float(rank))
            
        return ranks


class LinkPredictionEvaluator:
    """
    Comprehensive link prediction evaluator supporting multiple models and protocols
    """
    
    def __init__(self, train_triples: torch.Tensor, valid_triples: torch.Tensor, 
                 test_triples: torch.Tensor, num_entities: int, num_relations: int,
                 batch_size: int = 1000, device: str = 'cuda'):
        """
        Initialize comprehensive evaluator
        
        Args:
            train_triples: Training triples for filtering
            valid_triples: Validation triples for filtering  
            test_triples: Test triples for evaluation
            num_entities: Number of entities
            num_relations: Number of relations
            batch_size: Evaluation batch size
            device: Device for computation
        """
        self.train_triples = train_triples
        self.valid_triples = valid_triples
        self.test_triples = test_triples
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.batch_size = batch_size
        self.device = device
        
        # Create filtered evaluators for different scenarios
        self.test_evaluator = FilteredRankingEvaluator(
            filter_triples=[train_triples, valid_triples],
            batch_size=batch_size,
            device=device
        )
        
        self.valid_evaluator = FilteredRankingEvaluator(
            filter_triples=[train_triples],
            batch_size=batch_size,
            device=device
        )
    
    def evaluate_on_test(self, model) -> Dict[str, float]:
        """Evaluate model on test set with train+valid filtering"""
        return self.test_evaluator.evaluate_model(
            model, self.test_triples, self.num_entities
        )
    
    def evaluate_on_valid(self, model) -> Dict[str, float]:
        """Evaluate model on validation set with train filtering"""
        return self.valid_evaluator.evaluate_model(
            model, self.valid_triples, self.num_entities
        )
    
    def full_evaluation(self, model) -> Dict[str, Dict[str, float]]:
        """
        Perform full evaluation on both validation and test sets
        
        Returns:
            results: Dictionary with 'valid' and 'test' metrics
        """
        print("Evaluating on validation set...")
        valid_metrics = self.evaluate_on_valid(model)
        
        print("Evaluating on test set...")
        test_metrics = self.evaluate_on_test(model)
        
        return {
            'valid': valid_metrics,
            'test': test_metrics
        }
    
    def compare_models(self, models: Dict[str, torch.nn.Module]) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple models using the same evaluation protocol
        
        Args:
            models: Dictionary mapping model names to model instances
            
        Returns:
            comparison: Dictionary with results for each model
        """
        results = {}
        
        for model_name, model in models.items():
            print(f"\\nEvaluating {model_name}...")
            start_time = time.time()
            
            model_results = self.full_evaluation(model)
            eval_time = time.time() - start_time
            
            # Add timing information
            model_results['evaluation_time'] = eval_time
            results[model_name] = model_results
            
            # Print summary
            test_mrr = model_results['test']['mrr']
            test_hits1 = model_results['test']['hits_at_1']
            print(f"{model_name} - Test MRR: {test_mrr:.4f}, Hits@1: {test_hits1:.4f}")
        
        return results
    
    def statistical_significance_test(self, results1: List[float], results2: List[float], 
                                    test_type: str = 'wilcoxon') -> Dict[str, float]:
        """
        Perform statistical significance test between two model results
        
        Args:
            results1: Rankings from model 1
            results2: Rankings from model 2  
            test_type: 'wilcoxon' or 't_test'
            
        Returns:
            test_results: Dictionary with test statistics
        """
        from scipy import stats
        
        if test_type == 'wilcoxon':
            statistic, p_value = stats.wilcoxon(results1, results2)
        elif test_type == 't_test':
            statistic, p_value = stats.ttest_rel(results1, results2)
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }


def create_ogb_evaluator(dataset_name: str = 'ogbl-biokg', data_root: str = './data/') -> LinkPredictionEvaluator:
    """
    Create evaluator for OGB datasets with standard protocols
    
    Args:
        dataset_name: OGB dataset name
        data_root: Root directory for data
        
    Returns:
        evaluator: Configured LinkPredictionEvaluator
    """
    from ogb.linkproppred import LinkPropPredDataset
    
    # Load OGB dataset
    dataset = LinkPropPredDataset(name=dataset_name, root=data_root)
    split_edge = dataset.get_edge_split()
    
    # Convert to tensors
    train_triples = torch.from_numpy(
        np.stack([split_edge['train']['head'], 
                 split_edge['train']['relation'],
                 split_edge['train']['tail']], axis=1)
    )
    
    valid_triples = torch.from_numpy(
        np.stack([split_edge['valid']['head'],
                 split_edge['valid']['relation'], 
                 split_edge['valid']['tail']], axis=1)
    )
    
    test_triples = torch.from_numpy(
        np.stack([split_edge['test']['head'],
                 split_edge['test']['relation'],
                 split_edge['test']['tail']], axis=1)
    )
    
    # Get dataset statistics
    num_entities = dataset.graph['num_nodes']
    num_relations = int(train_triples[:, 1].max()) + 1
    
    return LinkPredictionEvaluator(
        train_triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        num_entities=num_entities,
        num_relations=num_relations
    )