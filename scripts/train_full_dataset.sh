#!/bin/bash

# Train RASG on full OGB-BioKG dataset
# Usage: bash scripts/train_full_dataset.sh

echo "🚀 Training RASG on FULL OGB-BioKG dataset..."

python main.py \
    --data-root 'full_biokg_db/' \
    --mapping-dir 'mappings/' \
    --train-db 'train.lmdb' \
    --valid-db 'valid.lmdb' \
    --test-db 'test.lmdb' \
    --global-graph 'mappings/global_graph.pkl' \
    --output-dir results_full_biokg \
    --use-full-dataset \
    --epochs 50 \
    --batch-size 16 \
    --num-negatives 10 \
    --num-train-negatives 1 \
    --lr 0.001 \
    --gnn-hidden 128 \
    --node-emb-dim 32 \
    --rel-emb-dim 64 \
    --att-dim 64 \
    --att-heads 4 \
    --num-layers 3 \
    --max-dist 10 \
    --dropout 0.3 \
    --patience 15 \
    --num-workers 4 \
    --save-model

echo "✅ Full dataset training completed!"
echo "📊 Results saved to: results_full_biokg/"