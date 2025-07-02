"""
Base class for Knowledge Graph Embedding models
Standardized interface following research protocols
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import numpy as np


class BaseKGEModel(nn.Module, ABC):
    """
    Abstract base class for Knowledge Graph Embedding models
    Ensures consistent interface across all baseline implementations
    """
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int, 
                 regularization: float = 1e-5, **kwargs):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations  
        self.embedding_dim = embedding_dim
        self.regularization = regularization
        
        # Initialize embeddings - to be implemented by subclasses
        self._init_embeddings()
        
    @abstractmethod
    def _init_embeddings(self):
        """Initialize entity and relation embeddings"""
        pass
    
    @abstractmethod
    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor, 
                     tails: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for given triples
        
        Args:
            heads: Entity indices [batch_size]
            relations: Relation indices [batch_size] 
            tails: Entity indices [batch_size]
            
        Returns:
            scores: Triple scores [batch_size]
        """
        pass
    
    def forward(self, triples: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for triple scoring
        
        Args:
            triples: Tensor of shape [batch_size, 3] with (h, r, t)
            
        Returns:
            scores: Triple scores [batch_size]
        """
        heads, relations, tails = triples[:, 0], triples[:, 1], triples[:, 2]
        return self.score_triples(heads, relations, tails)
    
    def score_heads(self, relations: torch.Tensor, tails: torch.Tensor, 
                   head_candidates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Score all possible heads for given (r, t) pairs
        Used for link prediction evaluation
        
        Args:
            relations: Relation indices [batch_size]
            tails: Tail entity indices [batch_size]
            head_candidates: Optional head candidates [batch_size, num_candidates]
                           If None, score against all entities
        
        Returns:
            scores: Head scores [batch_size, num_entities or num_candidates]
        """
        batch_size = relations.size(0)
        
        if head_candidates is None:
            # Score against all entities
            all_heads = torch.arange(self.num_entities, device=relations.device)
            all_heads = all_heads.expand(batch_size, -1)  # [batch_size, num_entities]
        else:
            all_heads = head_candidates
        
        num_candidates = all_heads.size(1)
        
        # Expand relations and tails to match all head candidates
        relations_exp = relations.unsqueeze(1).expand(-1, num_candidates).contiguous().view(-1)
        tails_exp = tails.unsqueeze(1).expand(-1, num_candidates).contiguous().view(-1)
        heads_exp = all_heads.contiguous().view(-1)
        
        # Score all combinations
        scores = self.score_triples(heads_exp, relations_exp, tails_exp)
        return scores.view(batch_size, num_candidates)
    
    def score_tails(self, heads: torch.Tensor, relations: torch.Tensor,
                   tail_candidates: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Score all possible tails for given (h, r) pairs
        Used for link prediction evaluation
        
        Args:
            heads: Head entity indices [batch_size]
            relations: Relation indices [batch_size]
            tail_candidates: Optional tail candidates [batch_size, num_candidates]
                           If None, score against all entities
        
        Returns:
            scores: Tail scores [batch_size, num_entities or num_candidates]
        """
        batch_size = heads.size(0)
        
        if tail_candidates is None:
            # Score against all entities
            all_tails = torch.arange(self.num_entities, device=heads.device)
            all_tails = all_tails.expand(batch_size, -1)  # [batch_size, num_entities]
        else:
            all_tails = tail_candidates
        
        num_candidates = all_tails.size(1)
        
        # Expand heads and relations to match all tail candidates
        heads_exp = heads.unsqueeze(1).expand(-1, num_candidates).contiguous().view(-1)
        relations_exp = relations.unsqueeze(1).expand(-1, num_candidates).contiguous().view(-1)
        tails_exp = all_tails.contiguous().view(-1)
        
        # Score all combinations
        scores = self.score_triples(heads_exp, relations_exp, tails_exp)
        return scores.view(batch_size, num_candidates)
    
    def get_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        Get entity and relation embeddings
        Useful for analysis and visualization
        
        Returns:
            embeddings: Dictionary with 'entities' and 'relations' tensors
        """
        return {
            'entities': self.entity_embeddings.weight.data,
            'relations': self.relation_embeddings.weight.data
        }
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss (typically L2)
        
        Returns:
            reg_loss: Regularization loss scalar
        """
        entity_reg = torch.norm(self.entity_embeddings.weight, p=2, dim=1).mean()
        relation_reg = torch.norm(self.relation_embeddings.weight, p=2, dim=1).mean()
        return self.regularization * (entity_reg + relation_reg)
    
    def init_weights(self, init_method: str = 'xavier_uniform'):
        """
        Initialize model weights using specified method
        
        Args:
            init_method: Initialization method ('xavier_uniform', 'normal', 'uniform')
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                if init_method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif init_method == 'normal':
                    nn.init.normal_(module.weight, std=0.1)
                elif init_method == 'uniform':
                    nn.init.uniform_(module.weight, -0.1, 0.1)
                    
    def get_model_size(self) -> int:
        """
        Get total number of parameters
        
        Returns:
            num_params: Total number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_embeddings(self, filepath: str):
        """
        Save entity and relation embeddings to file
        
        Args:
            filepath: Path to save embeddings
        """
        embeddings = self.get_embeddings()
        torch.save(embeddings, filepath)
        
    def load_embeddings(self, filepath: str):
        """
        Load entity and relation embeddings from file
        
        Args:
            filepath: Path to load embeddings from
        """
        embeddings = torch.load(filepath, map_location=self.entity_embeddings.weight.device)
        self.entity_embeddings.weight.data = embeddings['entities']
        self.relation_embeddings.weight.data = embeddings['relations']