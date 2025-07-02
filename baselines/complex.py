"""
ComplEx: Complex Embeddings for Simple Link Prediction
Paper: https://arxiv.org/abs/1606.06357

Standard implementation following research protocols
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseKGEModel


class ComplEx(BaseKGEModel):
    """
    ComplEx model using complex-valued embeddings
    
    The model represents entities and relations as complex vectors and uses
    the real part of trilinear product as the scoring function.
    
    Score function: Re(⟨h, r, ‾t⟩) where ‾t is complex conjugate of t
    This allows modeling of symmetric/antisymmetric relations.
    """
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 200,
                 regularization: float = 1e-5, dropout: float = 0.0, **kwargs):
        """
        Initialize ComplEx model
        
        Args:
            num_entities: Number of entities in KG
            num_relations: Number of relations in KG
            embedding_dim: Dimension of each complex component (total = 2 * embedding_dim)
            regularization: L2 regularization coefficient
            dropout: Dropout rate for embeddings
        """
        self.dropout = dropout
        super().__init__(num_entities, num_relations, embedding_dim, regularization, **kwargs)
        
    def _init_embeddings(self):
        """Initialize complex entity and relation embeddings"""
        # Entity embeddings: real and imaginary parts
        self.entity_embeddings_real = nn.Embedding(self.num_entities, self.embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(self.num_entities, self.embedding_dim)
        
        # Relation embeddings: real and imaginary parts
        self.relation_embeddings_real = nn.Embedding(self.num_relations, self.embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(self.num_relations, self.embedding_dim)
        
        # Initialize with Xavier uniform (research standard)
        self.init_weights('xavier_uniform')
        
        # Dropout layer
        if self.dropout > 0:
            self.dropout_layer = nn.Dropout(self.dropout)
        
    def init_weights(self, init_method: str = 'xavier_uniform'):
        """Initialize weights for all embedding components"""
        embeddings = [
            self.entity_embeddings_real, self.entity_embeddings_imag,
            self.relation_embeddings_real, self.relation_embeddings_imag
        ]
        
        for embedding in embeddings:
            if init_method == 'xavier_uniform':
                nn.init.xavier_uniform_(embedding.weight)
            elif init_method == 'normal':
                nn.init.normal_(embedding.weight, std=0.1)
            elif init_method == 'uniform':
                nn.init.uniform_(embedding.weight, -0.1, 0.1)
    
    def get_complex_embeddings(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor):
        """
        Get complex embeddings for entities and relations
        
        Args:
            entity_ids: Entity indices [batch_size]
            relation_ids: Relation indices [batch_size]
            
        Returns:
            entity_complex: Complex entity embeddings [batch_size, embedding_dim, 2]
            relation_complex: Complex relation embeddings [batch_size, embedding_dim, 2]
        """
        # Entity embeddings
        ent_real = self.entity_embeddings_real(entity_ids)  # [batch_size, dim]
        ent_imag = self.entity_embeddings_imag(entity_ids)  # [batch_size, dim]
        
        # Relation embeddings  
        rel_real = self.relation_embeddings_real(relation_ids)  # [batch_size, dim]
        rel_imag = self.relation_embeddings_imag(relation_ids)  # [batch_size, dim]
        
        # Apply dropout if specified
        if self.dropout > 0 and self.training:
            ent_real = self.dropout_layer(ent_real)
            ent_imag = self.dropout_layer(ent_imag)
            rel_real = self.dropout_layer(rel_real)
            rel_imag = self.dropout_layer(rel_imag)
        
        # Stack real and imaginary parts
        entity_complex = torch.stack([ent_real, ent_imag], dim=2)  # [batch_size, dim, 2]
        relation_complex = torch.stack([rel_real, rel_imag], dim=2)  # [batch_size, dim, 2]
        
        return entity_complex, relation_complex
    
    def complex_trilinear(self, h_complex: torch.Tensor, r_complex: torch.Tensor, 
                         t_complex: torch.Tensor) -> torch.Tensor:
        """
        Compute complex trilinear product: Re(⟨h, r, ‾t⟩)
        
        Args:
            h_complex: Head embeddings [batch_size, dim, 2]
            r_complex: Relation embeddings [batch_size, dim, 2] 
            t_complex: Tail embeddings [batch_size, dim, 2]
            
        Returns:
            scores: Real part of trilinear product [batch_size]
        """
        # Extract real and imaginary parts
        h_real, h_imag = h_complex[:, :, 0], h_complex[:, :, 1]  # [batch_size, dim]
        r_real, r_imag = r_complex[:, :, 0], r_complex[:, :, 1]  # [batch_size, dim]
        t_real, t_imag = t_complex[:, :, 0], t_complex[:, :, 1]  # [batch_size, dim]
        
        # Complex trilinear product: h * r * conj(t)
        # where conj(t) = t_real - i * t_imag
        
        # (h_real + i*h_imag) * (r_real + i*r_imag) * (t_real - i*t_imag)
        # Real part = h_real*r_real*t_real + h_real*r_imag*t_imag + h_imag*r_real*t_imag - h_imag*r_imag*t_real
        
        real_part = (
            h_real * r_real * t_real +
            h_real * r_imag * t_imag + 
            h_imag * r_real * t_imag -
            h_imag * r_imag * t_real
        )  # [batch_size, dim]
        
        # Sum over embedding dimension
        scores = real_part.sum(dim=1)  # [batch_size]
        
        return scores
    
    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor, 
                     tails: torch.Tensor) -> torch.Tensor:
        """
        Compute ComplEx scores for triples
        
        Args:
            heads: Head entity indices [batch_size]
            relations: Relation indices [batch_size]
            tails: Tail entity indices [batch_size]
            
        Returns:
            scores: Triple scores [batch_size] (higher = better)
        """
        # Get complex embeddings
        h_complex, r_complex = self.get_complex_embeddings(heads, relations)
        t_complex, _ = self.get_complex_embeddings(tails, torch.zeros_like(relations))
        
        # Compute trilinear product
        scores = self.complex_trilinear(h_complex, r_complex, t_complex)
        
        return scores
    
    def binary_cross_entropy_loss(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> torch.Tensor:
        """
        Compute binary cross-entropy loss for ComplEx
        
        Args:
            pos_triples: Positive triples [batch_size, 3]
            neg_triples: Negative triples [batch_size, 3]
            
        Returns:
            loss: Binary cross-entropy loss
        """
        # Positive scores
        pos_scores = self.forward(pos_triples)  # [batch_size]
        pos_labels = torch.ones_like(pos_scores)
        
        # Negative scores
        neg_scores = self.forward(neg_triples)  # [batch_size]
        neg_labels = torch.zeros_like(neg_scores)
        
        # Combine scores and labels
        all_scores = torch.cat([pos_scores, neg_scores])  # [2 * batch_size]
        all_labels = torch.cat([pos_labels, neg_labels])  # [2 * batch_size]
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(all_scores, all_labels)
        
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
        loss = self.binary_cross_entropy_loss(pos_triples, neg_triples)
        pos_scores = self.forward(pos_triples)
        neg_scores = self.forward(neg_triples)
        
        return loss, pos_scores, neg_scores
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss for all embedding components
        
        Returns:
            reg_loss: Regularization loss scalar
        """
        # Entity regularization
        entity_reg = (
            torch.norm(self.entity_embeddings_real.weight, p=2, dim=1).mean() +
            torch.norm(self.entity_embeddings_imag.weight, p=2, dim=1).mean()
        )
        
        # Relation regularization
        relation_reg = (
            torch.norm(self.relation_embeddings_real.weight, p=2, dim=1).mean() +
            torch.norm(self.relation_embeddings_imag.weight, p=2, dim=1).mean()
        )
        
        return self.regularization * (entity_reg + relation_reg) / 2
    
    def get_embeddings(self) -> dict:
        """
        Get all embedding components
        
        Returns:
            embeddings: Dictionary with all embedding tensors
        """
        return {
            'entities_real': self.entity_embeddings_real.weight.data,
            'entities_imag': self.entity_embeddings_imag.weight.data,
            'relations_real': self.relation_embeddings_real.weight.data,
            'relations_imag': self.relation_embeddings_imag.weight.data
        }
    
    def get_relation_patterns(self) -> dict:
        """
        Analyze relation patterns learned by ComplEx
        
        Returns:
            patterns: Dictionary with relation pattern analysis
        """
        with torch.no_grad():
            rel_real = self.relation_embeddings_real.weight.data
            rel_imag = self.relation_embeddings_imag.weight.data
            
            # Compute relation magnitudes
            rel_magnitudes = torch.sqrt(rel_real**2 + rel_imag**2).sum(dim=1)
            
            # Compute real/imaginary ratios (for pattern analysis)
            real_norms = torch.norm(rel_real, p=2, dim=1)
            imag_norms = torch.norm(rel_imag, p=2, dim=1)
            
            # Relations with high imaginary component (antisymmetric candidates)
            antisymmetric_ratio = imag_norms / (real_norms + 1e-8)
            antisymmetric_threshold = antisymmetric_ratio.mean() + antisymmetric_ratio.std()
            antisymmetric_rels = torch.where(antisymmetric_ratio > antisymmetric_threshold)[0]
            
            # Relations with high real component (symmetric candidates)
            symmetric_ratio = real_norms / (imag_norms + 1e-8)
            symmetric_threshold = symmetric_ratio.mean() + symmetric_ratio.std()
            symmetric_rels = torch.where(symmetric_ratio > symmetric_threshold)[0]
            
            return {
                'relation_magnitudes': rel_magnitudes.cpu().numpy(),
                'real_norms': real_norms.cpu().numpy(),
                'imaginary_norms': imag_norms.cpu().numpy(),
                'antisymmetric_candidates': antisymmetric_rels.cpu().numpy(),
                'symmetric_candidates': symmetric_rels.cpu().numpy(),
                'num_antisymmetric': len(antisymmetric_rels),
                'num_symmetric': len(symmetric_rels)
            }
    
    def predict_relation_type(self, relation_id: int) -> str:
        """
        Predict relation type based on learned embeddings
        
        Args:
            relation_id: Relation ID to analyze
            
        Returns:
            relation_type: 'symmetric', 'antisymmetric', or 'general'
        """
        with torch.no_grad():
            real_emb = self.relation_embeddings_real.weight[relation_id]
            imag_emb = self.relation_embeddings_imag.weight[relation_id]
            
            real_norm = torch.norm(real_emb, p=2)
            imag_norm = torch.norm(imag_emb, p=2)
            
            ratio = real_norm / (imag_norm + 1e-8)
            
            if ratio > 3.0:  # Predominantly real
                return 'symmetric'
            elif ratio < 0.33:  # Predominantly imaginary
                return 'antisymmetric'
            else:
                return 'general'


def get_complex_config(dataset_name: str = 'ogbl-biokg') -> dict:
    """
    Get standard ComplEx configuration for different datasets
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        config: Dictionary with hyperparameters
    """
    configs = {
        'ogbl-biokg': {
            'embedding_dim': 1000,  # Per component (total = 2000)
            'regularization': 1e-5,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 512,
            'negative_ratio': 1,
            'epochs': 100
        },
        'fb15k-237': {
            'embedding_dim': 200,  # Per component (total = 400)
            'regularization': 1e-3,
            'dropout': 0.1,
            'learning_rate': 0.001,
            'batch_size': 128,
            'negative_ratio': 1,
            'epochs': 200
        }
    }
    
    return configs.get(dataset_name, configs['ogbl-biokg'])