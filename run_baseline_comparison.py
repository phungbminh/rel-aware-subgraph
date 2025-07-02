#!/usr/bin/env python3
"""
Baseline Comparison Script for RASG Research
Train and evaluate TransE, ComplEx, RotatE against RASG model

Usage:
    python run_baseline_comparison.py --data-root ./test_5k_db/ --output-dir ./baseline_results/
"""

import argparse
import os
import json
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm

# Import baseline models
from baselines import TransE, ComplEx, RotatE
from baselines.transe import create_negative_samples, get_transe_config
from baselines.complex import get_complex_config  
from baselines.rotate import get_rotate_config, create_adversarial_negatives

# Import evaluation framework
from evaluation import LinkPredictionEvaluator, compute_ranking_metrics, format_results, create_results_table

# Import RASG for comparison
from model.model import RASG
from trainer.trainer import run_training
from extraction.datasets import SubGraphDataset


class BaselineTrainer:
    """Trainer for baseline KGE models with standardized protocols"""
    
    def __init__(self, num_entities: int, num_relations: int, device: str = 'cuda'):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.device = device
        
    def create_triple_dataset(self, triples: torch.Tensor, batch_size: int = 512) -> DataLoader:
        """Create DataLoader for triple data"""
        dataset = TensorDataset(triples)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    def train_transe(self, train_triples: torch.Tensor, valid_triples: torch.Tensor,
                    config: dict, output_dir: str) -> TransE:
        """Train TransE model"""
        print("\\n" + "="*50)
        print("Training TransE")
        print("="*50)
        
        # Initialize model
        model = TransE(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            **{k: v for k, v in config.items() if k not in ['learning_rate', 'epochs', 'batch_size', 'negative_ratio']}
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Training loop
        train_loader = self.create_triple_dataset(train_triples, config['batch_size'])
        
        best_mrr = 0
        patience = 5  # Reduced patience for 10 epochs
        no_improve = 0
        
        for epoch in range(config['epochs']):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_triples, in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch_triples = batch_triples.to(self.device)
                
                # Create negative samples
                neg_triples = create_negative_samples(
                    batch_triples, self.num_entities, config['negative_ratio']
                )
                
                optimizer.zero_grad()
                loss, _, _ = model.forward_with_loss(batch_triples, neg_triples)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation every 2 epochs for better early stopping
            if (epoch + 1) % 2 == 0:
                val_metrics = self._quick_evaluate(model, valid_triples[:500])  # Quick eval
                val_mrr = val_metrics['mrr']
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val MRR={val_mrr:.4f}")
                
                if val_mrr > best_mrr:
                    best_mrr = val_mrr
                    no_improve = 0
                    # Save best model
                    torch.save(model.state_dict(), os.path.join(output_dir, 'transe_best.pt'))
                else:
                    no_improve += 1
                    
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(output_dir, 'transe_best.pt')))
        return model
    
    def train_complex(self, train_triples: torch.Tensor, valid_triples: torch.Tensor,
                     config: dict, output_dir: str) -> ComplEx:
        """Train ComplEx model"""
        print("\\n" + "="*50)
        print("Training ComplEx")
        print("="*50)
        
        # Initialize model
        model = ComplEx(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            **{k: v for k, v in config.items() if k not in ['learning_rate', 'epochs', 'batch_size', 'negative_ratio']}
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Training loop
        train_loader = self.create_triple_dataset(train_triples, config['batch_size'])
        
        best_mrr = 0
        patience = 5  # Reduced patience for 10 epochs
        no_improve = 0
        
        for epoch in range(config['epochs']):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_triples, in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch_triples = batch_triples.to(self.device)
                
                # Create negative samples
                neg_triples = create_negative_samples(
                    batch_triples, self.num_entities, config['negative_ratio']
                )
                
                optimizer.zero_grad()
                loss, _, _ = model.forward_with_loss(batch_triples, neg_triples)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation every 2 epochs for better early stopping
            if (epoch + 1) % 2 == 0:
                val_metrics = self._quick_evaluate(model, valid_triples[:500])
                val_mrr = val_metrics['mrr']
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val MRR={val_mrr:.4f}")
                
                if val_mrr > best_mrr:
                    best_mrr = val_mrr
                    no_improve = 0
                    torch.save(model.state_dict(), os.path.join(output_dir, 'complex_best.pt'))
                else:
                    no_improve += 1
                    
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(output_dir, 'complex_best.pt')))
        return model
    
    def train_rotate(self, train_triples: torch.Tensor, valid_triples: torch.Tensor,
                    config: dict, output_dir: str) -> RotatE:
        """Train RotatE model"""
        print("\\n" + "="*50)
        print("Training RotatE")
        print("="*50)
        
        # Initialize model
        model = RotatE(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            **{k: v for k, v in config.items() if k not in ['learning_rate', 'epochs', 'batch_size', 'negative_ratio', 'loss_type']}
        ).to(self.device)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        
        # Training loop
        train_loader = self.create_triple_dataset(train_triples, config['batch_size'])
        
        best_mrr = 0
        patience = 5  # Reduced patience for 10 epochs
        no_improve = 0
        
        for epoch in range(config['epochs']):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch_triples, in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                batch_triples = batch_triples.to(self.device)
                
                # Create negative samples (more for RotatE adversarial training)
                neg_triples = create_negative_samples(
                    batch_triples, self.num_entities, config['negative_ratio']
                )
                
                optimizer.zero_grad()
                loss, _, _ = model.forward_with_loss(batch_triples, neg_triples, config.get('loss_type', 'margin'))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            
            # Validation every 2 epochs for better early stopping
            if (epoch + 1) % 2 == 0:
                val_metrics = self._quick_evaluate(model, valid_triples[:500])
                val_mrr = val_metrics['mrr']
                
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val MRR={val_mrr:.4f}")
                
                if val_mrr > best_mrr:
                    best_mrr = val_mrr
                    no_improve = 0
                    torch.save(model.state_dict(), os.path.join(output_dir, 'rotate_best.pt'))
                else:
                    no_improve += 1
                    
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(os.path.join(output_dir, 'rotate_best.pt')))
        return model
    
    def _quick_evaluate(self, model, test_triples: torch.Tensor) -> dict:
        """Quick evaluation for validation during training"""
        model.eval()
        ranks = []
        
        with torch.no_grad():
            # Sample larger set for more stable validation
            num_samples = min(len(test_triples), 50)  # Increased for stability
            sample_indices = torch.randperm(len(test_triples))[:num_samples]
            
            for i in sample_indices:
                triple = test_triples[i:i+1].to(self.device)
                h, r, t = triple[0, 0], triple[0, 1], triple[0, 2]
                
                # Score subset of entities to avoid OOM
                num_candidates = min(2000, self.num_entities)  # Increased candidates for better evaluation
                candidate_entities = torch.randperm(self.num_entities)[:num_candidates].to(self.device)
                
                # Ensure true tail is always in candidates
                if t not in candidate_entities:
                    candidate_entities[0] = t  # Replace first candidate with true tail
                
                heads = torch.tensor([h], device=self.device).expand(len(candidate_entities))
                relations = torch.tensor([r], device=self.device).expand(len(candidate_entities))
                
                scores = model.score_triples(heads, relations, candidate_entities)
                
                # Find rank of true tail
                true_idx = (candidate_entities == t).nonzero(as_tuple=True)[0][0]
                true_score = scores[true_idx]
                rank = (scores > true_score).sum().item() + 1
                # Normalize rank by number of candidates for comparable metrics
                normalized_rank = rank / len(candidate_entities) * self.num_entities
                ranks.append(normalized_rank)
        
        return compute_ranking_metrics(ranks)


def load_data_for_baselines(data_root: str) -> tuple:
    """Load data in format suitable for baseline models"""
    data_root = Path(data_root)
    mapping_dir = data_root / "mappings"
    
    # Load mappings
    with open(mapping_dir / "entity2id.pkl", 'rb') as f:
        entity2id = pickle.load(f)
    with open(mapping_dir / "relation2id.pkl", 'rb') as f:
        relation2id = pickle.load(f)
    
    num_entities = len(entity2id)
    num_relations = len(relation2id)
    
    # Load OGB data directly for baselines (they don't use subgraphs)
    from ogb.linkproppred import LinkPropPredDataset
    dataset = LinkPropPredDataset(name='ogbl-biokg', root='./data/ogb/')
    split_edge = dataset.get_edge_split()
    
    # Convert to tensors
    train_triples = torch.from_numpy(
        np.stack([split_edge['train']['head'][:5000],  # Use same 5K subset
                 split_edge['train']['relation'][:5000],
                 split_edge['train']['tail'][:5000]], axis=1)
    )
    
    valid_triples = torch.from_numpy(
        np.stack([split_edge['valid']['head'][:500],  # Use same 500 subset
                 split_edge['valid']['relation'][:500],
                 split_edge['valid']['tail'][:500]], axis=1)
    )
    
    test_triples = torch.from_numpy(
        np.stack([split_edge['test']['head'][:500],   # Use same 500 subset
                 split_edge['test']['relation'][:500],
                 split_edge['test']['tail'][:500]], axis=1)
    )
    
    return train_triples, valid_triples, test_triples, num_entities, num_relations


def train_rasg_baseline(data_root: str, output_dir: str) -> dict:
    """Train RASG model for comparison"""
    print("\\n" + "="*50)
    print("Training RASG (Our Method)")
    print("="*50)
    
    # Use existing RASG training pipeline
    import subprocess
    import sys
    
    rasg_output_dir = os.path.join(output_dir, "rasg_results")
    os.makedirs(rasg_output_dir, exist_ok=True)
    
    # Run RASG training
    cmd = [
        sys.executable, "main.py",
        "--data-root", data_root,
        "--output-dir", rasg_output_dir,
        "--epochs", "10",  # Consistent with baselines
        "--batch-size", "64",
        "--gnn-hidden", "128",  # Increased for better comparison
        "--num-layers", "3",    # Increased for better comparison
        "--lr", "0.001",
        "--patience", "8"       # Increased patience for 10 epochs
    ]
    
    print(f"Running RASG training: {' '.join(cmd)}")
    
    # Run with real-time output
    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=3600)  # 1 hour timeout
        
        if result.returncode != 0:
            print(f"RASG training failed with return code: {result.returncode}")
            return {}
            
    except subprocess.TimeoutExpired:
        print("RASG training timed out after 1 hour")
        return {}
    except Exception as e:
        print(f"RASG training failed with exception: {e}")
        return {}
    
    # Load RASG results
    results_file = os.path.join(rasg_output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            rasg_results = json.load(f)
        return rasg_results
    else:
        print("RASG results file not found")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Baseline Comparison for RASG Research")
    parser.add_argument("--data-root", type=str, required=True, help="Root directory with processed data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training")
    parser.add_argument("--quick-mode", action="store_true", help="Quick mode with reduced epochs")
    parser.add_argument("--models", nargs="+", default=["transe", "complex", "rotate", "rasg"], 
                       help="Models to train and compare")
    
    args = parser.parse_args()
    
    print("🔬 RASG Baseline Comparison")
    print("="*60)
    print(f"Data root: {args.data_root}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\\n📊 Loading data...")
    train_triples, valid_triples, test_triples, num_entities, num_relations = load_data_for_baselines(args.data_root)
    
    print(f"Dataset statistics:")
    print(f"  Entities: {num_entities}")
    print(f"  Relations: {num_relations}")
    print(f"  Train triples: {len(train_triples)}")
    print(f"  Valid triples: {len(valid_triples)}")
    print(f"  Test triples: {len(test_triples)}")
    
    # Initialize trainer
    trainer = BaselineTrainer(num_entities, num_relations, args.device)
    
    # Initialize evaluator with smaller batch size to avoid OOM
    evaluator = LinkPredictionEvaluator(
        train_triples=train_triples,
        valid_triples=valid_triples,
        test_triples=test_triples,
        num_entities=num_entities,
        num_relations=num_relations,
        batch_size=50,  # Much smaller for 45K entities
        device=args.device
    )
    
    # Train and evaluate models
    results = {}
    models = {}
    
    # Adjust configs for quick mode  
    epochs = 5 if args.quick_mode else 10  # Changed from 20 to 10 for balanced comparison
    
    if "transe" in args.models:
        # Train TransE
        config = get_transe_config('ogbl-biokg')
        config['epochs'] = epochs
        config['batch_size'] = 256  # Optimized for 5K dataset
        config['embedding_dim'] = 400  # Optimized for 5K dataset
        config['learning_rate'] = 0.005  # Higher for faster convergence
        config['margin'] = 5.0  # Reduced margin for smaller dataset
        config['regularization'] = 1e-6  # Reduced regularization
        
        start_time = time.time()
        transe_model = trainer.train_transe(train_triples, valid_triples, config, args.output_dir)
        training_time = time.time() - start_time
        
        # Evaluate TransE
        print("\\n📈 Evaluating TransE...")
        torch.cuda.empty_cache()  # Clear cache before evaluation
        transe_results = evaluator.full_evaluation(transe_model)
        transe_results['training_time'] = training_time
        transe_results['model_params'] = transe_model.get_model_size()
        
        results['TransE'] = transe_results
        models['TransE'] = transe_model
        torch.cuda.empty_cache()  # Clear cache after evaluation
    
    if "complex" in args.models:
        # Train ComplEx
        config = get_complex_config('ogbl-biokg')
        config['epochs'] = epochs
        config['batch_size'] = 256  # Optimized for 5K dataset
        config['embedding_dim'] = 200  # Per component, reduced for stability
        config['learning_rate'] = 0.01   # Much higher for ComplEx convergence
        config['regularization'] = 1e-7  # Much lower regularization
        config['negative_ratio'] = 5     # Increased negatives for better learning
        
        start_time = time.time()
        complex_model = trainer.train_complex(train_triples, valid_triples, config, args.output_dir)
        training_time = time.time() - start_time
        
        # Evaluate ComplEx
        print("\\n📈 Evaluating ComplEx...")
        torch.cuda.empty_cache()  # Clear cache before evaluation
        complex_results = evaluator.full_evaluation(complex_model)
        complex_results['training_time'] = training_time
        complex_results['model_params'] = complex_model.get_model_size()
        
        results['ComplEx'] = complex_results
        models['ComplEx'] = complex_model
        torch.cuda.empty_cache()  # Clear cache after evaluation
    
    if "rotate" in args.models:
        # Train RotatE
        config = get_rotate_config('ogbl-biokg')
        config['epochs'] = epochs
        config['batch_size'] = 256  # Optimized for 5K dataset
        config['embedding_dim'] = 400  # Total complex embedding dimension
        config['negative_ratio'] = 10  # Balanced negatives
        config['learning_rate'] = 0.002  # Conservative for RotatE stability
        config['loss_type'] = 'margin'   # Use simpler margin loss instead of adversarial
        config['margin'] = 3.0          # Smaller margin for 5K dataset
        config['regularization'] = 1e-6  # Reduced regularization
        
        start_time = time.time()
        rotate_model = trainer.train_rotate(train_triples, valid_triples, config, args.output_dir)
        training_time = time.time() - start_time
        
        # Evaluate RotatE
        print("\\n📈 Evaluating RotatE...")
        torch.cuda.empty_cache()  # Clear cache before evaluation
        rotate_results = evaluator.full_evaluation(rotate_model)
        rotate_results['training_time'] = training_time
        rotate_results['model_params'] = rotate_model.get_model_size()
        
        results['RotatE'] = rotate_results
        models['RotatE'] = rotate_model
        torch.cuda.empty_cache()  # Clear cache after evaluation
    
    if "rasg" in args.models:
        # Train RASG
        print(f"\\n⚠️  RASG training may take 30-60 minutes...")
        rasg_results = train_rasg_baseline(args.data_root, args.output_dir)
        if rasg_results:
            results['RASG'] = rasg_results
        else:
            print("⚠️  RASG training failed or timed out - continuing with baseline results only")
    
    # Save all results
    results_file = os.path.join(args.output_dir, "baseline_comparison.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print comparison table
    print("\\n" + "="*80)
    print("FINAL RESULTS COMPARISON")
    print("="*80)
    
    # Extract test metrics for comparison
    test_results = {}
    for model_name, model_results in results.items():
        if 'test' in model_results:
            test_results[model_name] = model_results['test']
        elif 'test_mrr' in model_results:  # RASG format
            test_results[model_name] = {
                'mrr': model_results['test_mrr'],
                'hits_at_1': model_results['test_hits'][0],
                'hits_at_3': model_results['test_hits'][1],
                'hits_at_10': model_results['test_hits'][2]
            }
    
    if test_results:
        table = create_results_table(test_results)
        print(table)
        
        # Save table to file
        with open(os.path.join(args.output_dir, "results_table.txt"), 'w') as f:
            f.write(table)
    
    # Print summary
    print(f"\\n✅ Baseline comparison completed!")
    print(f"📁 Results saved to: {args.output_dir}")
    print(f"📊 Detailed results: {results_file}")
    print(f"📋 Results table: {os.path.join(args.output_dir, 'results_table.txt')}")


if __name__ == "__main__":
    main()