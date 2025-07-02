#!/bin/bash

# Optimized config for 2x T4 Tesla 16GB
# Full OGB-BioKG dataset training

echo "🚀 Training RASG on 2x T4 Tesla 16GB..."
echo "💾 Total GPU Memory: 32GB"
echo "⚡ Expected speedup: 3-4x vs single GPU"

# Build dataset (if not exists)
if [ ! -d "full_biokg_db" ]; then
    echo "📦 Building full dataset..."
    python build_subgraph.py \
        --ogb-root './data/ogb/' \
        --output-dir './full_biokg_db/' \
        --k 2 \
        --tau 3 \
        --num-workers 8 \
        --batch-size 200 \
        --use-full-dataset \
        --num-negatives 10 \
        --max-nodes-per-graph 1000 \
        --rel-degree-dense
fi

# Training with optimal 2x T4 settings
echo "🎯 Starting training with 2x T4 optimized parameters..."

python main.py \
    --data-root 'full_biokg_db/' \
    --mapping-dir 'mappings/' \
    --train-db 'train.lmdb' \
    --valid-db 'valid.lmdb' \
    --test-db 'test.lmdb' \
    --global-graph 'mappings/global_graph.pkl' \
    --output-dir results_2x_t4_full \
    --use-full-dataset \
    --epochs 50 \
    --batch-size 16 \
    --num-negatives 10 \
    --num-train-negatives 2 \
    --lr 0.001 \
    --gnn-hidden 128 \
    --node-emb-dim 32 \
    --rel-emb-dim 64 \
    --att-dim 64 \
    --att-heads 4 \
    --num-layers 3 \
    --max-dist 10 \
    --dropout 0.2 \
    --patience 15 \
    --num-workers 6 \
    --save-model

echo "✅ Training completed!"
echo "📊 Results: results_2x_t4_full/"
echo "⏱️  Expected training time: 2-3 days"