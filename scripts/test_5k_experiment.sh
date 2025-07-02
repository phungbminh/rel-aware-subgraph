#!/bin/bash

# Test script cho 5000 triples với multiple configs
echo "🧪 Testing RASG with 5000 triples - Multiple configs"

# Build test dataset
echo "📦 Building test dataset (5000 triples)..."
python build_subgraph.py \
    --ogb-root './data/ogb/' \
    --output-dir './test_5k_db/' \
    --k 2 \
    --tau 3 \
    --num-workers 4 \
    --batch-size 100 \
    --max-triples 5000 \
    --max-eval-triples 500 \
    --num-negatives 5 \
    --max-nodes-per-graph 1000 \
    --rel-degree-dense

echo "✅ Dataset built! Starting experiments..."

# Test Config A: Conservative
echo "🔬 Test A: Conservative config"
python main.py \
    --data-root 'test_5k_db/' \
    --output-dir results_test_5k_conservative \
    --epochs 8 \
    --batch-size 4 \
    --num-negatives 5 \
    --num-train-negatives 1 \
    --gnn-hidden 64 \
    --node-emb-dim 16 \
    --rel-emb-dim 32 \
    --att-dim 32 \
    --num-layers 2 \
    --lr 0.001 \
    --patience 4 \
    --num-workers 2

# Test Config B: Balanced  
echo "🔬 Test B: Balanced config"
python main.py \
    --data-root 'test_5k_db/' \
    --output-dir results_test_5k_balanced \
    --epochs 8 \
    --batch-size 8 \
    --num-negatives 5 \
    --num-train-negatives 2 \
    --gnn-hidden 96 \
    --node-emb-dim 24 \
    --rel-emb-dim 48 \
    --att-dim 48 \
    --num-layers 3 \
    --lr 0.001 \
    --patience 4 \
    --num-workers 2

# Test Config C: Aggressive (Simulate 2x T4)
echo "🔬 Test C: Aggressive config (2x T4 simulation)"
python main.py \
    --data-root 'test_5k_db/' \
    --output-dir results_test_5k_aggressive \
    --epochs 8 \
    --batch-size 16 \
    --num-negatives 5 \
    --num-train-negatives 2 \
    --gnn-hidden 128 \
    --node-emb-dim 32 \
    --rel-emb-dim 64 \
    --att-dim 64 \
    --att-heads 4 \
    --num-layers 3 \
    --lr 0.001 \
    --patience 4 \
    --num-workers 4

echo "✅ All experiments completed!"
echo ""
echo "📊 Results comparison:"
echo "Conservative: $(cat results_test_5k_conservative/results.json | grep test_mrr)"
echo "Balanced:     $(cat results_test_5k_balanced/results.json | grep test_mrr)"  
echo "Aggressive:   $(cat results_test_5k_aggressive/results.json | grep test_mrr)"
echo ""
echo "⏱️  Training times:"
echo "Conservative: ~30-45 minutes"
echo "Balanced:     ~45-60 minutes"
echo "Aggressive:   ~60-90 minutes"