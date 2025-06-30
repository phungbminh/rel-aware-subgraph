import numpy as np
import torch
from collections import deque, defaultdict, Counter
from typing import List, Tuple, Dict

# ======= CSRGraph kiểm tra hợp lệ ===========
class CSRGraph:
    """Đồ thị dạng CSR với kiểm tra chỉ số node hợp lệ"""
    __slots__ = ('indptr', 'indices', 'num_nodes')
    def __init__(self, edge_index: np.ndarray, num_nodes: int):
        sources = edge_index[0]
        targets = edge_index[1]
        if sources.max() >= num_nodes or targets.max() >= num_nodes:
            raise ValueError(f"Edge index out of range: max node id {max(sources.max(), targets.max())} >= num_nodes {num_nodes}")
        sorted_idx = np.argsort(sources)
        sources = sources[sorted_idx]
        targets = targets[sorted_idx]
        counts = np.bincount(sources, minlength=num_nodes)
        self.indptr = np.zeros(num_nodes + 1, dtype=np.int32)
        self.indptr[1:] = np.cumsum(counts)
        self.indices = targets.astype(np.int32)
        self.num_nodes = num_nodes


