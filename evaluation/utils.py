"""
Evaluation utilities for knowledge graph link prediction
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import pickle
import lmdb
from pathlib import Path


def create_evaluation_dataset(triples: torch.Tensor, num_entities: int, 
                            num_relations: int, batch_size: int = 1000) -> torch.utils.data.DataLoader:
    """
    Create evaluation dataset from triples
    
    Args:
        triples: Triple tensor [num_triples, 3]
        num_entities: Number of entities
        num_relations: Number of relations
        batch_size: Batch size for evaluation
        
    Returns:
        dataloader: DataLoader for evaluation
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(triples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader


def load_evaluation_data(data_root: str, split: str = 'test') -> Tuple[torch.Tensor, Dict]:
    """
    Load evaluation data from processed format
    
    Args:
        data_root: Root directory with data
        split: Data split ('train', 'valid', 'test')
        
    Returns:
        triples: Triple tensor
        metadata: Dataset metadata
    """
    data_root = Path(data_root)
    
    # Load from OGB format (fallback)
    try:
        from ogb.linkproppred import LinkPropPredDataset
        dataset = LinkPropPredDataset(name='ogbl-biokg', root=str(data_root / 'ogb'))
        split_edge = dataset.get_edge_split()
        
        triples = torch.from_numpy(
            np.stack([split_edge[split]['head'],
                     split_edge[split]['relation'], 
                     split_edge[split]['tail']], axis=1)
        )
        
        metadata = {
            'num_entities': dataset.graph['num_nodes'],
            'num_relations': int(max(split_edge['train']['relation'].max(),
                                   split_edge['valid']['relation'].max(), 
                                   split_edge['test']['relation'].max())) + 1
        }
        
        return triples, metadata
        
    except Exception as e:
        raise FileNotFoundError(f"Could not load evaluation data from {data_root}: {e}")


def load_lmdb_data(lmdb_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """
    Load data from LMDB database
    
    Args:
        lmdb_path: Path to LMDB database
        max_samples: Maximum number of samples to load
        
    Returns:
        data_list: List of loaded data items
    """
    import lmdb
    import pickle
    
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    data_list = []
    
    with env.begin() as txn:
        cursor = txn.cursor()
        count = 0
        
        for key, value in cursor:
            if key == b'_progress':
                continue
                
            try:
                data = pickle.loads(value)
                if 'error' not in data:
                    data_list.append(data)
                    count += 1
                    
                if max_samples and count >= max_samples:
                    break
                    
            except Exception:
                continue
    
    env.close()
    return data_list


def save_evaluation_results(results: Dict, output_path: str):
    """
    Save evaluation results to file
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    import json
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)


def load_mappings(mapping_dir: str) -> Dict[str, Dict]:
    """
    Load entity and relation mappings
    
    Args:
        mapping_dir: Directory containing mapping files
        
    Returns:
        mappings: Dictionary with entity2id, relation2id, etc.
    """
    mapping_dir = Path(mapping_dir)
    mappings = {}
    
    mapping_files = [
        'entity2id.pkl', 'relation2id.pkl', 
        'id2entity.pkl', 'id2relation.pkl'
    ]
    
    for filename in mapping_files:
        filepath = mapping_dir / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                key = filename.replace('.pkl', '')
                mappings[key] = pickle.load(f)
    
    return mappings


def filter_triples_by_entities(triples: torch.Tensor, 
                              valid_entities: set) -> torch.Tensor:
    """
    Filter triples to only include valid entities
    
    Args:
        triples: Triple tensor [num_triples, 3]
        valid_entities: Set of valid entity IDs
        
    Returns:
        filtered_triples: Filtered triple tensor
    """
    # Check which triples have both head and tail in valid entities
    valid_mask = torch.tensor([
        (h.item() in valid_entities and t.item() in valid_entities)
        for h, r, t in triples
    ])
    
    return triples[valid_mask]


def compute_triple_statistics(triples: torch.Tensor) -> Dict[str, float]:
    """
    Compute basic statistics for triples
    
    Args:
        triples: Triple tensor [num_triples, 3]
        
    Returns:
        stats: Dictionary with statistics
    """
    heads, relations, tails = triples[:, 0], triples[:, 1], triples[:, 2]
    
    stats = {
        'num_triples': len(triples),
        'num_unique_entities': len(torch.unique(torch.cat([heads, tails]))),
        'num_unique_relations': len(torch.unique(relations)),
        'avg_head_degree': len(triples) / len(torch.unique(heads)),
        'avg_tail_degree': len(triples) / len(torch.unique(tails)),
        'avg_relation_frequency': len(triples) / len(torch.unique(relations))
    }
    
    return stats


def create_negative_samples_evaluation(positive_triples: torch.Tensor, 
                                     num_entities: int,
                                     num_negatives: int = 100,
                                     mode: str = 'tail') -> torch.Tensor:
    """
    Create negative samples for evaluation
    
    Args:
        positive_triples: Positive triples [num_pos, 3]
        num_entities: Total number of entities
        num_negatives: Number of negatives per positive
        mode: 'head', 'tail', or 'both'
        
    Returns:
        negative_triples: Negative triples [num_pos * num_negatives, 3]
    """
    num_pos = len(positive_triples)
    negatives = []
    
    for i in range(num_pos):
        h, r, t = positive_triples[i]
        
        for _ in range(num_negatives):
            if mode == 'tail' or (mode == 'both' and np.random.rand() < 0.5):
                # Corrupt tail
                neg_t = torch.randint(0, num_entities, (1,)).item()
                while neg_t == t.item():
                    neg_t = torch.randint(0, num_entities, (1,)).item()
                negatives.append([h.item(), r.item(), neg_t])
            else:
                # Corrupt head
                neg_h = torch.randint(0, num_entities, (1,)).item()
                while neg_h == h.item():
                    neg_h = torch.randint(0, num_entities, (1,)).item()
                negatives.append([neg_h, r.item(), t.item()])
    
    return torch.tensor(negatives, dtype=torch.long)