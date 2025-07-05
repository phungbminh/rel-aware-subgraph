import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict
import math

class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler based on graph sizes to ensure uniform memory usage.
    Groups graphs with similar sizes to avoid memory issues.
    """
    def __init__(self, dataset, max_tokens_per_batch=50000, shuffle=True):
        self.dataset = dataset
        self.max_tokens_per_batch = max_tokens_per_batch
        self.shuffle = shuffle
        
        # Pre-compute graph sizes (number of nodes)
        print("[INFO] Pre-computing graph sizes for dynamic batching...")
        self.graph_sizes = []
        for i in range(len(dataset)):
            # Get sample without loading full graph
            key = dataset.keys[i]
            with dataset.env.begin() as txn:
                raw = txn.get(key)
                if raw:
                    import pickle
                    data = pickle.loads(raw)
                    num_nodes = len(data['nodes'])
                    self.graph_sizes.append(num_nodes)
                else:
                    self.graph_sizes.append(100)  # fallback
        
        self.graph_sizes = np.array(self.graph_sizes)
        print(f"[INFO] Graph sizes: min={self.graph_sizes.min()}, max={self.graph_sizes.max()}, "
              f"mean={self.graph_sizes.mean():.1f}, median={np.median(self.graph_sizes):.1f}")
        
        # Create size-based bins
        self._create_batches()
    
    def _create_batches(self):
        """Create batches based on graph sizes"""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        # Sort by size for better batching
        size_idx_pairs = [(self.graph_sizes[i], i) for i in indices]
        size_idx_pairs.sort(key=lambda x: x[0])
        
        self.batches = []
        current_batch = []
        current_tokens = 0
        
        for size, idx in size_idx_pairs:
            # Estimate tokens (nodes + edges roughly)
            estimated_tokens = size * 3  # rough estimate: nodes + edges
            
            if current_tokens + estimated_tokens > self.max_tokens_per_batch and current_batch:
                # Start new batch
                if self.shuffle:
                    np.random.shuffle(current_batch)
                self.batches.append(current_batch)
                current_batch = [idx]
                current_tokens = estimated_tokens
            else:
                current_batch.append(idx)
                current_tokens += estimated_tokens
        
        # Add last batch
        if current_batch:
            if self.shuffle:
                np.random.shuffle(current_batch)
            self.batches.append(current_batch)
        
        print(f"[INFO] Created {len(self.batches)} dynamic batches, "
              f"sizes: {[len(b) for b in self.batches[:5]]}...")
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

class FixedSizeBatchSampler(Sampler):
    """
    Simple batch sampler that caps maximum batch size while maintaining dynamic sizing.
    """
    def __init__(self, dataset, max_batch_size=16, max_nodes_per_batch=30000, shuffle=True, is_full_dataset=False):
        self.dataset = dataset
        self.max_batch_size = max_batch_size
        self.max_nodes_per_batch = max_nodes_per_batch
        self.shuffle = shuffle
        self.is_full_dataset = is_full_dataset
        
        # Optimize for full dataset - much more aggressive reduction
        if is_full_dataset:
            # MUCH more conservative limits for large datasets to prevent OOM
            self.max_batch_size = min(max_batch_size, 2)  # Reduce from 8 to 2
            self.max_nodes_per_batch = min(max_nodes_per_batch, 500)  # Reduce from 15000 to 500
            print(f"[AUTO] Adjusted for full dataset: max_batch_size={self.max_batch_size}, max_nodes_per_batch={self.max_nodes_per_batch}")
        
        # Pre-compute graph sizes with progress bar for large datasets
        print("[INFO] Pre-computing graph sizes...")
        self.graph_sizes = []
        dataset_size = len(dataset)
        
        if is_full_dataset and dataset_size > 1000:
            # Use sampling for very large datasets
            sample_size = min(1000, dataset_size // 10)
            print(f"[INFO] Sampling {sample_size} graphs for size estimation (full dataset mode)")
            indices = np.random.choice(dataset_size, sample_size, replace=False)
        else:
            indices = range(dataset_size)
        
        sample_sizes = []
        for i in indices:
            key = dataset.keys[i]
            with dataset.env.begin() as txn:
                raw = txn.get(key)
                if raw:
                    import pickle
                    data = pickle.loads(raw)
                    num_nodes = len(data['nodes'])
                    sample_sizes.append(num_nodes)
                else:
                    sample_sizes.append(100)
        
        if is_full_dataset and len(indices) < dataset_size:
            # Extrapolate sizes for all graphs
            sample_sizes = np.array(sample_sizes)
            mean_size = sample_sizes.mean()
            std_size = sample_sizes.std()
            print(f"[INFO] Extrapolating sizes: mean={mean_size:.1f}, std={std_size:.1f}")
            
            # Generate sizes for all graphs based on sample distribution
            np.random.seed(42)  # Reproducible
            self.graph_sizes = np.random.normal(mean_size, std_size, dataset_size)
            self.graph_sizes = np.clip(self.graph_sizes, sample_sizes.min(), sample_sizes.max()).astype(int)
        else:
            self.graph_sizes = np.array(sample_sizes)
        
        print(f"[INFO] Graph sizes: min={self.graph_sizes.min()}, max={self.graph_sizes.max()}")
        
        self._create_batches()
    
    def _create_batches(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(indices)
        
        self.batches = []
        current_batch = []
        current_nodes = 0
        
        for idx in indices:
            nodes = self.graph_sizes[idx]
            
            # Check if adding this graph exceeds limits
            if (len(current_batch) >= self.max_batch_size or 
                current_nodes + nodes > self.max_nodes_per_batch) and current_batch:
                self.batches.append(current_batch)
                current_batch = [idx]
                current_nodes = nodes
            else:
                current_batch.append(idx)
                current_nodes += nodes
        
        if current_batch:
            self.batches.append(current_batch)
        
        print(f"[INFO] Created {len(self.batches)} batches with size control")
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)