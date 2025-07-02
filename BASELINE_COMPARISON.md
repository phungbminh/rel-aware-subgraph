# Baseline Comparison for RASG Research

This document provides comprehensive guidelines for training and evaluating baseline knowledge graph embedding models against the RASG approach.

## Overview

The baseline comparison framework implements three standard KGE models:

- **TransE**: Translating embeddings for simple relational modeling
- **ComplEx**: Complex embeddings for symmetric/antisymmetric relations  
- **RotatE**: Rotational embeddings in complex space

All models follow research standards for fair comparison with identical:
- Evaluation protocols (filtered ranking)
- Dataset splits and preprocessing
- Hyperparameter optimization strategies
- Computational resource allocation

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_baselines.txt
```

### 2. Prepare Data
```bash
# Ensure you have processed subgraph data
python build_subgraph.py --data-root ./test_5k_db/ --k 2 --tau 3 --max-triples 5000
```

### 3. Run Comparison
```bash
# Quick comparison (5 epochs each)
bash scripts/run_baseline_comparison.sh ./test_5k_db/ ./baseline_results/ cuda

# Full comparison (20 epochs each)
bash scripts/run_baseline_comparison.sh ./test_5k_db/ ./baseline_results/ cuda full

# Individual model training
bash scripts/run_baseline_comparison.sh ./test_5k_db/ ./baseline_results/ cuda individual
```

## Implementation Details

### Model Architectures

#### TransE
```python
# Score function: -||h + r - t||_p
class TransE(BaseKGEModel):
    def score_triples(self, heads, relations, tails):
        h_emb = self.entity_embeddings(heads)
        r_emb = self.relation_embeddings(relations)
        t_emb = self.entity_embeddings(tails)
        
        pred = h_emb + r_emb - t_emb
        distances = torch.norm(pred, p=self.norm_p, dim=1)
        return -distances  # Higher score = better
```

**Key Features:**
- Simple additive composition: h + r ≈ t
- L1 or L2 norm for distance computation
- Entity embedding normalization constraint
- Margin ranking loss for training

#### ComplEx  
```python
# Score function: Re(⟨h, r, conj(t)⟩)
class ComplEx(BaseKGEModel):
    def complex_trilinear(self, h_complex, r_complex, t_complex):
        # Extract real/imaginary parts
        h_real, h_imag = h_complex[:, :, 0], h_complex[:, :, 1]
        r_real, r_imag = r_complex[:, :, 0], r_complex[:, :, 1]
        t_real, t_imag = t_complex[:, :, 0], t_complex[:, :, 1]
        
        # Complex trilinear product
        real_part = (h_real * r_real * t_real + 
                    h_real * r_imag * t_imag + 
                    h_imag * r_real * t_imag - 
                    h_imag * r_imag * t_real)
        
        return real_part.sum(dim=1)
```

**Key Features:**
- Complex-valued embeddings (real + imaginary)
- Handles symmetric and antisymmetric relations
- Binary cross-entropy loss
- Dropout regularization

#### RotatE
```python
# Score function: -||h ∘ r - t||₂ (rotation in complex space)
class RotatE(BaseKGEModel):
    def complex_rotation(self, entity_complex, phases):
        real = entity_complex[:, :, 0]
        imag = entity_complex[:, :, 1]
        
        cos_phase = torch.cos(phases)
        sin_phase = torch.sin(phases)
        
        # Apply rotation
        rotated_real = real * cos_phase - imag * sin_phase
        rotated_imag = real * sin_phase + imag * cos_phase
        
        return torch.stack([rotated_real, rotated_imag], dim=2)
```

**Key Features:**
- Relations as rotations in complex space
- Models composition, inversion, symmetry patterns
- Self-adversarial negative sampling
- Uniform phase initialization

### Training Protocols

#### Standard Hyperparameters

| Model | Embedding Dim | Learning Rate | Batch Size | Epochs | Loss Function |
|-------|---------------|---------------|------------|--------|---------------|
| TransE | 2000 | 0.0005 | 512 | 100 | Margin Ranking |
| ComplEx | 1000 (×2) | 0.001 | 512 | 100 | Binary Cross-Entropy |
| RotatE | 2000 | 0.0005 | 512 | 100 | Self-Adversarial |

*Note: Dimensions reduced for 5K test dataset*

#### Training Strategies

**TransE Training:**
```python
def train_transe(train_triples, config):
    for epoch in range(config['epochs']):
        for batch in train_loader:
            # Create negative samples
            neg_triples = create_negative_samples(batch, num_entities, ratio=1)
            
            # Margin ranking loss
            loss = margin_ranking_loss(batch, neg_triples, margin=config['margin'])
            
            # Normalize embeddings (TransE constraint)
            normalize_entity_embeddings()
```

**ComplEx Training:**
```python
def train_complex(train_triples, config):
    for epoch in range(config['epochs']):
        for batch in train_loader:
            # Create negative samples
            neg_triples = create_negative_samples(batch, num_entities, ratio=1)
            
            # Binary cross-entropy loss
            pos_scores = model(batch)
            neg_scores = model(neg_triples)
            loss = bce_loss(pos_scores, neg_scores)
```

**RotatE Training:**
```python
def train_rotate(train_triples, config):
    for epoch in range(config['epochs']):
        for batch in train_loader:
            # Self-adversarial negative sampling
            neg_triples = create_adversarial_negatives(batch, current_scores, temperature=1.0)
            
            # Self-adversarial loss
            loss = self_adversarial_loss(batch, neg_triples)
```

### Evaluation Framework

#### Filtered Ranking Protocol
```python
class FilteredRankingEvaluator:
    def __init__(self, filter_triples):
        # Build filter sets: (r,t) -> valid heads, (h,r) -> valid tails
        self.head_filters = defaultdict(set)
        self.tail_filters = defaultdict(set)
        
        for h, r, t in filter_triples:
            self.head_filters[(r, t)].add(h)
            self.tail_filters[(h, r)].add(t)
    
    def compute_filtered_rank(self, model, test_triple):
        h, r, t = test_triple
        
        # Score all possible tails
        scores = model.score_tails([h], [r])  # [1, num_entities]
        
        # Filter known positive tails (except true tail)
        filter_set = self.tail_filters[(h, r)]
        for filter_tail in filter_set:
            if filter_tail != t:
                scores[filter_tail] = float('-inf')
        
        # Compute rank
        true_score = scores[t]
        rank = (scores > true_score).sum() + 1
        return rank
```

#### Metrics Computation
```python
def compute_ranking_metrics(ranks):
    ranks = np.array(ranks)
    
    metrics = {
        'mrr': (1.0 / ranks).mean(),
        'hits_at_1': (ranks <= 1).mean(),
        'hits_at_3': (ranks <= 3).mean(),
        'hits_at_10': (ranks <= 10).mean(),
        'mean_rank': ranks.mean(),
        'median_rank': np.median(ranks)
    }
    
    return metrics
```

## Usage Examples

### Example 1: Train Single Model
```python
from baselines import TransE
from evaluation import LinkPredictionEvaluator

# Initialize model
model = TransE(num_entities=45085, num_relations=51, embedding_dim=200)

# Train model
trainer = BaselineTrainer(num_entities, num_relations)
trained_model = trainer.train_transe(train_triples, valid_triples, config, output_dir)

# Evaluate
evaluator = LinkPredictionEvaluator(train_triples, valid_triples, test_triples, 
                                  num_entities, num_relations)
results = evaluator.evaluate_on_test(trained_model)
print(f"Test MRR: {results['mrr']:.4f}")
```

### Example 2: Compare Multiple Models
```python
from run_baseline_comparison import BaselineTrainer

# Load data
train_triples, valid_triples, test_triples, num_entities, num_relations = load_data_for_baselines(data_root)

# Initialize trainer
trainer = BaselineTrainer(num_entities, num_relations)

# Train models
models = {}
models['TransE'] = trainer.train_transe(train_triples, valid_triples, transe_config, output_dir)
models['ComplEx'] = trainer.train_complex(train_triples, valid_triples, complex_config, output_dir)
models['RotatE'] = trainer.train_rotate(train_triples, valid_triples, rotate_config, output_dir)

# Compare results
evaluator = LinkPredictionEvaluator(train_triples, valid_triples, test_triples, 
                                  num_entities, num_relations)
comparison_results = evaluator.compare_models(models)
```

### Example 3: Hyperparameter Analysis
```python
# Test different embedding dimensions
embedding_dims = [100, 200, 500, 1000]
results = {}

for dim in embedding_dims:
    config = get_transe_config()
    config['embedding_dim'] = dim
    
    model = trainer.train_transe(train_triples, valid_triples, config, output_dir)
    metrics = evaluator.evaluate_on_test(model)
    results[f'TransE_dim_{dim}'] = metrics

# Analyze results
best_dim = max(results.keys(), key=lambda k: results[k]['mrr'])
print(f"Best embedding dimension: {best_dim}")
```

## Results Analysis

### Expected Performance Ranges

Based on literature and OGB-BioKG characteristics:

| Model | Expected MRR | Expected Hits@1 | Expected Hits@10 | Notes |
|-------|-------------|----------------|------------------|-------|
| TransE | 0.35-0.42 | 0.25-0.32 | 0.55-0.65 | Poor on symmetric relations |
| ComplEx | 0.40-0.45 | 0.28-0.35 | 0.60-0.70 | Good on biomedical relations |
| RotatE | 0.38-0.44 | 0.26-0.33 | 0.58-0.68 | Balanced performance |
| **RASG** | **0.42-0.48** | **0.30-0.38** | **0.62-0.72** | **Our method** |

### Performance Analysis

#### Relation Pattern Analysis
```python
# Analyze which relations each model handles well
def analyze_relation_performance(model, test_triples, relation2id):
    relation_ranks = defaultdict(list)
    
    for triple in test_triples:
        h, r, t = triple
        rank = compute_rank(model, triple)
        relation_ranks[r].append(rank)
    
    relation_mrr = {}
    for rel_id, ranks in relation_ranks.items():
        relation_mrr[rel_id] = (1.0 / np.array(ranks)).mean()
    
    return relation_mrr

# Compare relation-specific performance
transe_rel_mrr = analyze_relation_performance(transe_model, test_triples, relation2id)
complex_rel_mrr = analyze_relation_performance(complex_model, test_triples, relation2id)
rasg_rel_mrr = analyze_relation_performance(rasg_model, test_triples, relation2id)
```

#### Statistical Significance
```python
from scipy import stats

# Compare RASG vs best baseline
rasg_ranks = evaluate_detailed(rasg_model, test_triples)
baseline_ranks = evaluate_detailed(best_baseline_model, test_triples)

# Wilcoxon signed-rank test
statistic, p_value = stats.wilcoxon(rasg_ranks, baseline_ranks)
print(f"Statistical significance: p = {p_value:.4f}")
print(f"Significant improvement: {p_value < 0.05}")
```

## File Organization

```
rel-aware-subgraph/
├── baselines/
│   ├── __init__.py
│   ├── base_model.py          # Abstract base class
│   ├── transe.py             # TransE implementation
│   ├── complex.py            # ComplEx implementation
│   └── rotate.py             # RotatE implementation
├── evaluation/
│   ├── __init__.py
│   ├── evaluator.py          # Evaluation framework
│   ├── metrics.py            # Metric computations
│   └── utils.py              # Evaluation utilities
├── run_baseline_comparison.py # Main comparison script
├── scripts/
│   └── run_baseline_comparison.sh
├── requirements_baselines.txt
└── BASELINE_COMPARISON.md    # This file
```

## Research Standards Compliance

### Reproducibility
- Fixed random seeds for all experiments
- Identical data preprocessing across models
- Standardized hyperparameter grids
- Same computational resources and training time

### Fair Comparison
- Consistent evaluation protocol (filtered ranking)
- Same negative sampling strategies during evaluation
- Identical train/valid/test splits
- Same stopping criteria and validation procedures

### Reporting Standards
- Complete hyperparameter specifications
- Training time and computational requirements
- Statistical significance testing
- Confidence intervals for metrics
- Model size and parameter counts

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch-size 64`
   - Reduce embedding dimensions
   - Use gradient accumulation

2. **Slow Training**
   - Enable mixed precision training
   - Reduce negative sampling ratio
   - Use smaller validation sets

3. **Poor Convergence**
   - Adjust learning rate
   - Increase regularization
   - Check data preprocessing

### Debug Mode
```bash
# Run with debug output
python run_baseline_comparison.py \
    --data-root ./test_5k_db/ \
    --output-dir ./debug_results/ \
    --quick-mode \
    --models transe \
    --debug
```

## Citation

When using these baseline implementations, please cite the original papers:

```bibtex
@inproceedings{bordes2013translating,
  title={Translating embeddings for modeling multi-relational data},
  author={Bordes, Antoine and Usunier, Nicolas and Garcia-Duran, Alberto and Weston, Jason and Yakhnenko, Oksana},
  booktitle={NeurIPS},
  year={2013}
}

@inproceedings{trouillon2016complex,
  title={Complex embeddings for simple link prediction},
  author={Trouillon, Théo and Welbl, Johannes and Riedel, Sebastian and Gaussier, Éric and Bouchard, Guillaume},
  booktitle={ICML},
  year={2016}
}

@inproceedings{sun2019rotate,
  title={RotatE: Knowledge graph embedding by relational rotation in complex space},
  author={Sun, Zhiqing and Deng, Zhi-Hong and Nie, Jian-Yun and Tang, Jian},
  booktitle={ICLR},
  year={2019}
}
```