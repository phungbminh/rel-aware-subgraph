import torch
import numpy as np
from torch_geometric.data import Data

class RuntimeNegativeSampler:
    """
    Runtime negative sampling for training following RGCN/CompGCN style.
    Sample negative triples by corrupting head or tail entities.
    """
    def __init__(self, all_entities, corruption_rate=0.5):
        """
        Args:
            all_entities: List of all entity IDs
            corruption_rate: Probability of corrupting head vs tail
        """
        self.all_entities = torch.tensor(all_entities, dtype=torch.long)
        self.num_entities = len(all_entities)
        self.corruption_rate = corruption_rate
        
    def sample_negatives(self, pos_graphs, relations, num_negatives_per_pos=1):
        """
        Sample negative graphs by corrupting positive triples.
        
        Args:
            pos_graphs: List of positive Data objects
            relations: List of relation IDs  
            num_negatives_per_pos: Number of negatives per positive
            
        Returns:
            neg_graphs: List of negative Data objects
            neg_relations: List of relation IDs for negatives
        """
        neg_graphs = []
        neg_relations = []
        
        for pos_graph, relation in zip(pos_graphs, relations):
            # Extract original triple info
            original_nodes = pos_graph.original_nodes
            head_idx = pos_graph.head_idx
            tail_idx = pos_graph.tail_idx
            
            if head_idx < 0 or tail_idx < 0:
                # Skip if head/tail not found
                continue
                
            original_head = original_nodes[head_idx].item()
            original_tail = original_nodes[tail_idx].item()
            
            for _ in range(num_negatives_per_pos):
                if torch.rand(1).item() < self.corruption_rate:
                    # Corrupt head
                    new_head = self._sample_random_entity(exclude=original_head)
                    new_tail = original_tail
                else:
                    # Corrupt tail  
                    new_head = original_head
                    new_tail = self._sample_random_entity(exclude=original_tail)
                
                # Create negative graph with same structure but different head/tail
                neg_graph = self._create_negative_graph(
                    pos_graph, new_head, new_tail, head_idx, tail_idx
                )
                
                neg_graphs.append(neg_graph)
                neg_relations.append(relation)
                
        return neg_graphs, neg_relations
    
    def _sample_random_entity(self, exclude=None):
        """Sample random entity excluding the given one"""
        while True:
            entity = self.all_entities[torch.randint(0, self.num_entities, (1,))].item()
            if entity != exclude:
                return entity
    
    def _create_negative_graph(self, pos_graph, new_head, new_tail, head_idx, tail_idx):
        """Create negative graph by replacing head/tail in original graph"""
        # Use deepcopy approach to ensure all attributes are copied
        import copy
        neg_graph = copy.deepcopy(pos_graph)
        
        # Update the corrupted entities
        neg_graph.original_nodes[head_idx] = new_head
        neg_graph.original_nodes[tail_idx] = new_tail
        
        # Update masks for new head/tail
        neg_graph.head_mask.fill_(False)
        neg_graph.tail_mask.fill_(False)
        neg_graph.head_mask[head_idx] = True
        neg_graph.tail_mask[tail_idx] = True
        
        # Note: We keep the same graph structure (nodes, edges) but change the semantic meaning
        # This is a simplification - ideally we'd rebuild the subgraph around new entities
        # But for efficiency, we assume the subgraph structure is still informative
        
        return neg_graph

class FilteredNegativeSampler(RuntimeNegativeSampler):
    """
    Enhanced negative sampler that avoids sampling known positive triples.
    Following filtered evaluation setting.
    """
    def __init__(self, all_entities, known_triples=None, corruption_rate=0.5):
        super().__init__(all_entities, corruption_rate)
        
        # Convert known triples to set for fast lookup
        if known_triples is not None:
            self.known_triples = set(map(tuple, known_triples))
        else:
            self.known_triples = set()
    
    def _sample_random_entity(self, exclude=None, head=None, relation=None, tail=None):
        """Sample entity while avoiding known positive triples"""
        max_attempts = 50  # Prevent infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            entity = self.all_entities[torch.randint(0, self.num_entities, (1,))].item()
            
            if entity == exclude:
                attempts += 1
                continue
                
            # Check if this would create a known positive triple
            if head is not None and tail is not None:
                # Corrupting head
                candidate_triple = (entity, relation, tail)
            elif head is not None and relation is not None:
                # Corrupting tail
                candidate_triple = (head, relation, entity)
            else:
                # No filtering needed
                return entity
                
            if candidate_triple not in self.known_triples:
                return entity
                
            attempts += 1
        
        # Fallback: return random entity even if it might be positive
        return self.all_entities[torch.randint(0, self.num_entities, (1,))].item()