"""
RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
Paper: https://arxiv.org/abs/1902.10197

Standard implementation following research protocols
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseKGEModel


class RotatE(BaseKGEModel):
    """
    RotatE model using complex-valued embeddings with rotational relations
    
    The model represents entities as complex vectors and relations as rotations
    in complex space. The scoring function models relations as rotations from
    head to tail entity.
    
    Score function: -||h ∘ r - t||₂ where ∘ is complex multiplication (rotation)
    """
    
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int = 500,
                 margin: float = 6.0, epsilon: float = 2.0, regularization: float = 1e-5, 
                 adversarial_temperature: float = 1.0, **kwargs):
        """
        Initialize RotatE model
        
        Args:
            num_entities: Number of entities in KG
            num_relations: Number of relations in KG
            embedding_dim: Dimension of real/imaginary parts (total complex dim = embedding_dim)
            margin: Margin for self-adversarial negative sampling
            epsilon: Fixed margin for loss computation  
            regularization: L2 regularization coefficient
            adversarial_temperature: Temperature for self-adversarial sampling
        """
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for RotatE")
            
        self.complex_dim = embedding_dim // 2  # Half for real, half for imaginary
        self.margin = margin
        self.epsilon = epsilon
        self.adversarial_temperature = adversarial_temperature
        
        super().__init__(num_entities, num_relations, embedding_dim, regularization, **kwargs)
        
    def _init_embeddings(self):
        """Initialize entity and relation embeddings"""
        # Entity embeddings (complex-valued)
        self.entity_embeddings = nn.Embedding(self.num_entities, self.embedding_dim)
        
        # Relation embeddings (phases for rotation)
        # Relations are represented as phases in [0, 2π)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.complex_dim)
        
        # Initialize embeddings
        self.init_weights()
        
        # Constraint: entity embeddings should be on unit circle (optional, often omitted)
        # self._normalize_entities()
        
    def init_weights(self, init_method: str = 'uniform'):
        """Initialize weights following RotatE protocol"""
        # Entity embeddings: uniform initialization
        nn.init.uniform_(self.entity_embeddings.weight, -self.epsilon/self.margin, self.epsilon/self.margin)
        
        # Relation embeddings: uniform phases in [0, 2π)
        nn.init.uniform_(self.relation_embeddings.weight, 0, 2 * math.pi)
        
    def _normalize_entities(self):
        """Normalize entity embeddings to unit circle (optional constraint)"""
        with torch.no_grad():
            # Reshape to complex view: [num_entities, complex_dim, 2]
            entity_complex = self.entity_embeddings.weight.view(-1, self.complex_dim, 2)
            
            # Compute magnitudes
            magnitudes = torch.sqrt(entity_complex[:, :, 0]**2 + entity_complex[:, :, 1]**2)
            magnitudes = magnitudes.unsqueeze(2)  # [num_entities, complex_dim, 1]
            
            # Normalize (avoid division by zero)
            entity_complex = entity_complex / (magnitudes + 1e-8)
            
            # Update weights
            self.entity_embeddings.weight.data = entity_complex.view(-1, self.embedding_dim)
    
    def get_complex_representations(self, entity_ids: torch.Tensor, relation_ids: torch.Tensor):
        """
        Get complex representations for entities and relation phases
        
        Args:
            entity_ids: Entity indices [batch_size]
            relation_ids: Relation indices [batch_size]
            
        Returns:
            entity_complex: Complex entity embeddings [batch_size, complex_dim, 2]
            relation_phases: Relation phases [batch_size, complex_dim]
        """
        # Entity embeddings - reshape to complex view
        entity_emb = self.entity_embeddings(entity_ids)  # [batch_size, embedding_dim]
        entity_complex = entity_emb.view(-1, self.complex_dim, 2)  # [batch_size, complex_dim, 2]
        
        # Relation phases
        relation_phases = self.relation_embeddings(relation_ids)  # [batch_size, complex_dim]
        
        return entity_complex, relation_phases
    
    def complex_rotation(self, entity_complex: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        """
        Apply complex rotation to entity embeddings
        
        Rotation: e^(iθ) * z = (cos(θ) + i*sin(θ)) * (a + i*b)
                              = (a*cos(θ) - b*sin(θ)) + i*(a*sin(θ) + b*cos(θ))
        
        Args:
            entity_complex: Complex entity embeddings [batch_size, complex_dim, 2]
            phases: Rotation phases [batch_size, complex_dim]
            
        Returns:
            rotated: Rotated complex embeddings [batch_size, complex_dim, 2]
        """
        # Extract real and imaginary parts
        real = entity_complex[:, :, 0]  # [batch_size, complex_dim]
        imag = entity_complex[:, :, 1]  # [batch_size, complex_dim]
        
        # Compute cos and sin of phases
        cos_phase = torch.cos(phases)  # [batch_size, complex_dim]
        sin_phase = torch.sin(phases)  # [batch_size, complex_dim]
        
        # Apply rotation
        rotated_real = real * cos_phase - imag * sin_phase
        rotated_imag = real * sin_phase + imag * cos_phase
        
        # Stack back to complex representation
        rotated = torch.stack([rotated_real, rotated_imag], dim=2)  # [batch_size, complex_dim, 2]
        
        return rotated
    
    def score_triples(self, heads: torch.Tensor, relations: torch.Tensor, 
                     tails: torch.Tensor) -> torch.Tensor:
        """
        Compute RotatE scores for triples
        
        Score = -||h ∘ r - t||₂ (higher score = more plausible)
        
        Args:
            heads: Head entity indices [batch_size]
            relations: Relation indices [batch_size]
            tails: Tail entity indices [batch_size]
            
        Returns:
            scores: Triple scores [batch_size] (higher = better)
        """
        # Get complex representations
        h_complex, r_phases = self.get_complex_representations(heads, relations)
        t_complex, _ = self.get_complex_representations(tails, torch.zeros_like(relations))
        
        # Apply rotation: h ∘ r
        h_rotated = self.complex_rotation(h_complex, r_phases)  # [batch_size, complex_dim, 2]
        
        # Compute difference: h ∘ r - t
        diff = h_rotated - t_complex  # [batch_size, complex_dim, 2]
        
        # Compute L2 norm in complex space
        # ||z||₂ = sqrt(real²+ imag²) for each complex dimension, then sum
        diff_norms = torch.sqrt(diff[:, :, 0]**2 + diff[:, :, 1]**2)  # [batch_size, complex_dim]
        distances = diff_norms.sum(dim=1)  # [batch_size]
        
        # Convert to scores (negative distance - higher is better)
        scores = self.margin - distances
        
        return scores
    
    def self_adversarial_loss(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> torch.Tensor:
        """
        Compute self-adversarial negative sampling loss for RotatE
        
        Args:
            pos_triples: Positive triples [batch_size, 3]
            neg_triples: Negative triples [batch_size * neg_ratio, 3]
            
        Returns:
            loss: Self-adversarial loss
        """
        # Positive scores
        pos_scores = self.forward(pos_triples)  # [batch_size]
        pos_loss = F.logsigmoid(-pos_scores).mean()
        
        # Negative scores with self-adversarial weighting
        neg_scores = self.forward(neg_triples)  # [batch_size * neg_ratio]
        
        # Self-adversarial weights (softmax with temperature)
        neg_weights = F.softmax(neg_scores * self.adversarial_temperature, dim=0).detach()
        
        # Weighted negative loss
        neg_loss = (neg_weights * F.logsigmoid(neg_scores)).sum()
        
        # Total loss
        loss = (pos_loss + neg_loss) / 2
        
        # Add regularization
        reg_loss = self.regularization_loss()
        
        return loss + reg_loss
    
    def margin_loss(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor) -> torch.Tensor:
        """
        Compute margin-based loss (alternative to self-adversarial)
        
        Args:
            pos_triples: Positive triples [batch_size, 3]
            neg_triples: Negative triples [batch_size * neg_ratio, 3]
            
        Returns:
            loss: Margin-based loss
        """
        pos_scores = self.forward(pos_triples)  # Higher = better
        neg_scores = self.forward(neg_triples)  # Higher = better
        
        # For margin ranking loss, we want: margin + neg_score - pos_score > 0
        # This means pos_score should be higher than neg_score by at least margin
        batch_size = pos_triples.size(0)
        neg_ratio = neg_triples.size(0) // batch_size
        
        # Reshape negative scores to [batch_size, neg_ratio]
        neg_scores = neg_scores.view(batch_size, neg_ratio)
        
        # Expand positive scores to match: [batch_size, neg_ratio]
        pos_scores = pos_scores.unsqueeze(1).expand(-1, neg_ratio)
        
        # Standard margin ranking loss: max(0, margin + neg_score - pos_score)
        loss = F.relu(self.margin + neg_scores - pos_scores).mean()
        
        # Add regularization
        reg_loss = self.regularization_loss()
        
        return loss + reg_loss
    
    def forward_with_loss(self, pos_triples: torch.Tensor, neg_triples: torch.Tensor, 
                         loss_type: str = 'adversarial') -> tuple:
        """
        Forward pass with loss computation
        
        Args:
            pos_triples: Positive triples [batch_size, 3]
            neg_triples: Negative triples [batch_size * neg_ratio, 3]
            loss_type: 'adversarial' or 'margin'
            
        Returns:
            (loss, pos_scores, neg_scores)
        """
        if loss_type == 'adversarial':
            loss = self.self_adversarial_loss(pos_triples, neg_triples)
        elif loss_type == 'margin':
            loss = self.margin_loss(pos_triples, neg_triples)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        pos_scores = self.forward(pos_triples)
        neg_scores = self.forward(neg_triples)
        
        return loss, pos_scores, neg_scores
    
    def get_relation_patterns(self) -> dict:
        """
        Analyze relation patterns learned by RotatE
        
        Returns:
            patterns: Dictionary with relation pattern analysis
        """
        with torch.no_grad():
            phases = self.relation_embeddings.weight.data  # [num_relations, complex_dim]
            
            # Compute phase statistics
            phase_means = phases.mean(dim=1)  # Average phase per relation
            phase_stds = phases.std(dim=1)   # Phase variance per relation
            
            # Identify special patterns
            # Near-zero phases (identity-like relations)
            identity_threshold = 0.1
            identity_rels = torch.where(phase_means.abs() < identity_threshold)[0]
            
            # Near-π phases (inversion-like relations)  
            inversion_threshold = 0.1
            inversion_rels = torch.where((phase_means - math.pi).abs() < inversion_threshold)[0]
            
            # High variance (complex rotational patterns)
            complex_threshold = phase_stds.mean() + phase_stds.std()
            complex_rels = torch.where(phase_stds > complex_threshold)[0]
            
            return {
                'phase_means': phase_means.cpu().numpy(),
                'phase_stds': phase_stds.cpu().numpy(),
                'identity_candidates': identity_rels.cpu().numpy(),
                'inversion_candidates': inversion_rels.cpu().numpy(),
                'complex_candidates': complex_rels.cpu().numpy(),
                'num_identity': len(identity_rels),
                'num_inversion': len(inversion_rels),
                'num_complex': len(complex_rels)
            }
    
    def get_embeddings(self) -> dict:
        """
        Get entity and relation embeddings
        
        Returns:
            embeddings: Dictionary with embedding tensors
        """
        return {
            'entities': self.entity_embeddings.weight.data,
            'relations': self.relation_embeddings.weight.data
        }
    
    def predict_composition(self, rel1_id: int, rel2_id: int) -> torch.Tensor:
        """
        Predict composition of two relations: r1 ∘ r2
        In RotatE, this is simply addition of phases
        
        Args:
            rel1_id: First relation ID
            rel2_id: Second relation ID
            
        Returns:
            composed_phases: Composed relation phases [complex_dim]
        """
        with torch.no_grad():
            rel1_phases = self.relation_embeddings.weight[rel1_id]
            rel2_phases = self.relation_embeddings.weight[rel2_id]
            
            # Composition = addition of phases (modulo 2π)
            composed = (rel1_phases + rel2_phases) % (2 * math.pi)
            
            return composed


def create_adversarial_negatives(pos_triples: torch.Tensor, scores: torch.Tensor, 
                               num_entities: int, temperature: float = 1.0) -> torch.Tensor:
    """
    Create self-adversarial negative samples for RotatE
    
    Args:
        pos_triples: Positive triples [batch_size, 3]
        scores: Current model scores for sampling weights [batch_size]
        num_entities: Total number of entities
        temperature: Temperature for sampling
        
    Returns:
        neg_triples: Adversarial negative triples [batch_size, 3]
    """
    batch_size = pos_triples.size(0)
    device = pos_triples.device
    
    neg_triples = pos_triples.clone()
    
    # Sample corruption type (head or tail) for each triple
    corrupt_head = torch.rand(batch_size, device=device) < 0.5
    
    # Create adversarial weights based on current scores
    weights = F.softmax(scores * temperature, dim=0)
    
    # Sample negative entities based on weights
    entity_distribution = torch.multinomial(weights, num_entities, replacement=True)
    
    for i in range(batch_size):
        if corrupt_head[i]:
            # Corrupt head
            neg_head = torch.randint(0, num_entities, (1,), device=device).item()
            neg_triples[i, 0] = neg_head
        else:
            # Corrupt tail
            neg_tail = torch.randint(0, num_entities, (1,), device=device).item()
            neg_triples[i, 2] = neg_tail
    
    return neg_triples


def get_rotate_config(dataset_name: str = 'ogbl-biokg') -> dict:
    """
    Get standard RotatE configuration for different datasets
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        config: Dictionary with hyperparameters
    """
    configs = {
        'ogbl-biokg': {
            'embedding_dim': 2000,  # Total complex embedding dimension
            'margin': 12.0,
            'epsilon': 2.0,
            'regularization': 1e-7,
            'adversarial_temperature': 1.0,
            'learning_rate': 0.0005,
            'batch_size': 512,
            'negative_ratio': 256,  # High for self-adversarial sampling
            'epochs': 100,
            'loss_type': 'adversarial'
        },
        'fb15k-237': {
            'embedding_dim': 1000,
            'margin': 6.0,
            'epsilon': 2.0,
            'regularization': 1e-5,
            'adversarial_temperature': 1.0,
            'learning_rate': 0.001,
            'batch_size': 128,
            'negative_ratio': 64,
            'epochs': 200,
            'loss_type': 'adversarial'
        }
    }
    
    return configs.get(dataset_name, configs['ogbl-biokg'])