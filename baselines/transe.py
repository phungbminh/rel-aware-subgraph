"""
TransE: Translating Embeddings for Modeling Multi-relational Data
Paper: https://papers.nips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

Standard implementation following research protocols
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseKGEModel


class TransE(BaseKGEModel):
    """
    TransE model: h + r ≈ t in embedding space
    
    The model learns entity and relation embeddings such that for a valid triple (h, r, t),
    the embedding of head entity plus relation embedding should be close to tail entity embedding.
    
    Score function: -||h + r - t||_p (typically p=1 or p=2)
    """
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 200,
                 norm_p: int = 1, margin: float = 1.0, regularization: float = 1e-5, **kwargs):
        """
        Initialize TransE model
        
        Args:
            num_entities: Number of entities in KG
            num_relations: Number of relations in KG  
            embedding_dim: Dimension of embeddings (typically 200-500 for OGB-BioKG)
            norm_p: Norm for distance computation (1 or 2)
            margin: Margin for margin ranking loss
            regularization: L2 regularization coefficient
        """
        self.norm_p = norm_p
        self.margin = margin
        super().__init__(num_entities, num_relations, embedding_dim, regularization, **kwargs)
        
    def _init_embeddings(self):
        """Initialize entity and relation embeddings"""
        # Entity embeddings
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        # Relation embeddings  
        self.relation_embeddings = nn.Embedding(self.num_relations, self.embedding_dim)
        
        # Initialize with Xavier uniform (research standard)
        self.init_weights('xavier_uniform')
        
        # Normalize entity embeddings (TransE constraint)
        self._normalize_embeddings()
        
    def _normalize_embeddings(self):
        """Normalize entity embeddings to unit sphere (TransE constraint)"""
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(
                self.entity_embeddings.weight.data, p=2, dim=1
            )
    
    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor, 
                     tails: torch.Tensor) -> torch.Tensor:
        """
        Compute TransE scores for triples
        
        Score = -||h + r - t||_p (higher score = more plausible)
        
        Args:
            heads: Head entity indices [batch_size]
            relations: Relation indices [batch_size]
            tails: Tail entity indices [batch_size]
            
        Returns:
            scores: Triple scores [batch_size] (higher = better)
        """
        # Get embeddings
        h_emb = self.entity_embeddings(heads)  # [batch_size, dim]
        r_emb = self.relation_embeddings(relations)  # [batch_size, dim]
        t_emb = self.entity_embeddings(tails)  # [batch_size, dim]
        
        # Compute h + r - t
        pred = h_emb + r_emb - t_emb  # [batch_size, dim]
        
        # Compute distance (negative for scoring - higher is better)
        distances = torch.norm(pred, p=self.norm_p, dim=1)  # [batch_size]
        scores = -distances  # Convert to scores (higher = better)
        
        return scores
    
    def margin_ranking_loss(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> torch.Tensor:
        """
        Compute margin ranking loss for TransE
        
        Loss = max(0, margin + d(h,r,t) - d(h',r,t')) where d = ||h + r - t||
        
        Args:
            pos_triples: Positive triples [batch_size, 3]
            neg_triples: Negative triples [batch_size, 3]
            
        Returns:
            loss: Margin ranking loss
        """
        pos_scores = self.forward(pos_triples)  # Higher = better
        neg_scores = self.forward(neg_triples)  # Higher = better
        
        # Convert back to distances for loss computation
        pos_distances = -pos_scores
        neg_distances = -neg_scores
        
        # Margin ranking loss: margin + pos_dist - neg_dist
        loss = F.relu(self.margin + pos_distances - neg_distances).mean()
        
        # Add regularization
        reg_loss = self.regularization_loss()
        
        return loss + reg_loss
    
    def forward_with_loss(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> tuple:
        """
        Forward pass with loss computation
        
        Args:
            pos_triples: Positive triples [batch_size, 3]
            neg_triples: Negative triples [batch_size, 3]
            
        Returns:
            (loss, pos_scores, neg_scores)
        """
        # Normalize embeddings during training (TransE constraint)
        self._normalize_embeddings()
        
        loss = self.margin_ranking_loss(pos_triples, neg_triples)
        pos_scores = self.forward(pos_triples)
        neg_scores = self.forward(neg_triples)
        
        return loss, pos_scores, neg_scores
    
    def get_relation_patterns(self) -> dict:
        """
        Analyze relation patterns learned by TransE
        Useful for understanding model behavior
        
        Returns:
            patterns: Dictionary with relation pattern analysis
        """
        with torch.no_grad():
            rel_embs = self.relation_embeddings.weight.data
            
            # Compute relation norms
            rel_norms = torch.norm(rel_embs, p=2, dim=1)
            
            # Find symmetric-like relations (small norm)
            symmetric_threshold = rel_norms.mean() - rel_norms.std()
            symmetric_rels = torch.where(rel_norms < symmetric_threshold)[0]
            
            return {
                'relation_norms': rel_norms.cpu().numpy(),
                'mean_norm': rel_norms.mean().item(),
                'std_norm': rel_norms.std().item(), 
                'symmetric_candidates': symmetric_rels.cpu().numpy(),
                'num_symmetric': len(symmetric_rels)
            }
    
    def predict_missing_entity(self, known_entity: int, relation: int, 
                              position: str = 'tail') -> torch.Tensor:
        """
        Predict missing entity in a triple
        
        Args:
            known_entity: Known entity ID
            relation: Relation ID
            position: 'head' or 'tail' - position of missing entity
            
        Returns:
            scores: Scores for all entities [num_entities]
        """
        device = self.entity_embeddings.weight.device
        
        if position == 'tail':
            # Predict tail: score all t for (known_entity, relation, t)
            heads = torch.tensor([known_entity], device=device)
            relations = torch.tensor([relation], device=device)
            return self.score_tails(heads, relations).squeeze(0)
        
        elif position == 'head':
            # Predict head: score all h for (h, relation, known_entity)  
            relations = torch.tensor([relation], device=device)
            tails = torch.tensor([known_entity], device=device)
            return self.score_heads(relations, tails).squeeze(0)
        
        else:
            raise ValueError(f"Position must be 'head' or 'tail', got {position}")


# Training utilities for TransE
def create_negative_samples(pos_triples: torch.Tensor, num_entities: int, 
                          num_negatives: int = 1, corruption_mode: str = 'both') -> torch.Tensor:
    """
    Create negative samples for TransE training
    
    Args:
        pos_triples: Positive triples [batch_size, 3] 
        num_entities: Total number of entities
        num_negatives: Number of negatives per positive
        corruption_mode: 'head', 'tail', or 'both'
        
    Returns:
        neg_triples: Negative triples [batch_size * num_negatives, 3]
    """
    if len(pos_triples.shape) == 1:
        pos_triples = pos_triples.unsqueeze(0)  # Make it [1, 3] if it's [3]
    
    batch_size = pos_triples.size(0)
    device = pos_triples.device
    
    neg_triples = []
    
    for _ in range(num_negatives):
        neg_batch = pos_triples.clone()
        
        if corruption_mode == 'both':
            # Randomly choose head or tail corruption for each triple
            corrupt_head = torch.rand(batch_size, device=device) < 0.5
        elif corruption_mode == 'head':
            corrupt_head = torch.ones(batch_size, dtype=torch.bool, device=device)
        elif corruption_mode == 'tail':
            corrupt_head = torch.zeros(batch_size, dtype=torch.bool, device=device)
        else:
            raise ValueError(f"Unknown corruption mode: {corruption_mode}")
        
        # Corrupt heads - iterate to avoid indexing issues
        for i in range(batch_size):
            if corrupt_head[i]:
                # Corrupt head
                new_head = torch.randint(0, num_entities, (1,), device=device).item()
                while new_head == neg_batch[i, 0].item():
                    new_head = torch.randint(0, num_entities, (1,), device=device).item()
                neg_batch[i, 0] = new_head
            else:
                # Corrupt tail
                new_tail = torch.randint(0, num_entities, (1,), device=device).item()
                while new_tail == neg_batch[i, 2].item():
                    new_tail = torch.randint(0, num_entities, (1,), device=device).item()
                neg_batch[i, 2] = new_tail
        
        neg_triples.append(neg_batch)
    
    return torch.cat(neg_triples, dim=0)


def get_transe_config(dataset_name: str = 'ogbl-biokg') -> dict:
    """
    Get standard TransE configuration for different datasets
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        config: Dictionary with hyperparameters
    """
    configs = {
        'ogbl-biokg': {
            'embedding_dim': 2000,
            'norm_p': 1,
            'margin': 12.0,
            'regularization': 1e-7,
            'learning_rate': 0.0005,
            'batch_size': 512,
            'negative_ratio': 1,
            'epochs': 100
        },
        'fb15k-237': {
            'embedding_dim': 200,
            'norm_p': 1, 
            'margin': 1.0,
            'regularization': 1e-5,
            'learning_rate': 0.001,
            'batch_size': 128,
            'negative_ratio': 1,
            'epochs': 200
        }
    }
    
    return configs.get(dataset_name, configs['ogbl-biokg'])