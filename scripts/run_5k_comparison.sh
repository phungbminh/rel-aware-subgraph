#!/bin/bash

# RASG Baseline Comparison Script
# 5K triples, tau=2, k=2, no node limit per subgraph, 10 epochs

echo "🔬 RASG Baseline Comparison - 5K Dataset"
echo "=========================================="
echo "Configuration:"
echo "  - 5K training triples"
echo "  - tau=2 (relation degree threshold)"
echo "  - k=2 (hop distance)" 
echo "  - No node limit per subgraph"
echo "  - 10 epochs training"
echo ""

# Step 1: Build optimized 5K dataset
echo "📦 Building optimized 5K dataset..."
python build_subgraph.py \
    --ogb-root './data/ogb/' \
    --output-dir './comparison_5k_db/' \
    --k 2 \
    --tau 2 \
    --num-workers 6 \
    --batch-size 150 \
    --max-triples 5000 \
    --max-eval-triples 500 \
    --num-negatives 5 \
    --max-nodes-per-graph 2000 \
    --rel-degree-dense

if [ $? -ne 0 ]; then
    echo "❌ Dataset building failed!"
    exit 1
fi

echo "✅ Dataset built successfully!"
echo ""

# Step 2: Run baseline comparison
echo "🚀 Starting baseline model comparison..."
python run_baseline_comparison.py \
    --data-root './comparison_5k_db/' \
    --output-dir './comparison_results_5k/' \
    --device cuda \
    --models transe complex rotate rasg

if [ $? -ne 0 ]; then
    echo "❌ Baseline comparison failed!"
    exit 1
fi

echo ""
echo "✅ Baseline comparison completed!"

# Step 3: Display results
echo ""
echo "📊 COMPARISON RESULTS:"
echo "=" * 60

if [ -f "./comparison_results_5k/results_table.txt" ]; then
    echo "Results Table:"
    cat "./comparison_results_5k/results_table.txt"
else
    echo "Results table not found, checking JSON..."
    if [ -f "./comparison_results_5k/baseline_comparison.json" ]; then
        echo "JSON results available at: ./comparison_results_5k/baseline_comparison.json"
    fi
fi

echo ""
echo "📁 Detailed results saved to: ./comparison_results_5k/"
echo "📊 Dataset info: ./comparison_5k_db/"
echo ""
echo "🎯 Key improvements in this configuration:"
echo "  - Lower tau=2 for more inclusive subgraphs"
echo "  - Increased max nodes (2000) for richer context"
echo "  - 10 epochs for meaningful comparison"
echo "  - Dense relation degree computation for speed"