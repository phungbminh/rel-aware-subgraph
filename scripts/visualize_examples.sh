#!/bin/bash

# Example script để visualize subgraphs
echo "🎨 RASG Subgraph Visualization Examples"

# Tạo thư mục output
mkdir -p visualizations

# Example 1: Visualize valid set subgraphs (default)
echo "📊 Example 1: Visualizing validation subgraphs..."
python visualize_subgraphs.py \
    --data-root ./test_5k_db/ \
    --lmdb-file valid.lmdb \
    --num-samples 6 \
    --output-dir visualizations/valid_set

# Example 2: Visualize training subgraphs
echo "📊 Example 2: Visualizing training subgraphs..."
python visualize_subgraphs.py \
    --data-root ./test_5k_db/ \
    --lmdb-file train.lmdb \
    --num-samples 8 \
    --output-dir visualizations/train_set

# Example 3: Individual detailed plots only
echo "📊 Example 3: Creating detailed individual plots..."
python visualize_subgraphs.py \
    --data-root ./test_5k_db/ \
    --lmdb-file test.lmdb \
    --num-samples 3 \
    --output-dir visualizations/detailed_only \
    --individual-only \
    --figsize 12 10

# Example 4: Large grid visualization
echo "📊 Example 4: Large grid visualization..."
python visualize_subgraphs.py \
    --data-root ./test_5k_db/ \
    --lmdb-file valid.lmdb \
    --num-samples 12 \
    --output-dir visualizations/large_grid

echo "✅ All visualizations completed!"
echo "📁 Check the following directories:"
echo "   - visualizations/valid_set/"
echo "   - visualizations/train_set/" 
echo "   - visualizations/detailed_only/"
echo "   - visualizations/large_grid/"