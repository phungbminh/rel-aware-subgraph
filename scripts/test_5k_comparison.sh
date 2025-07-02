#!/bin/bash

# Quick test comparison với 5K dataset 
# Test configuration trước khi chạy full comparison

echo "🧪 Quick Test - 5K Comparison Configuration"
echo "==========================================="

# Test dataset building
echo "📦 Testing dataset building..."
python build_subgraph.py \
    --ogb-root './data/ogb/' \
    --output-dir './test_comparison_5k_db/' \
    --k 2 \
    --tau 2 \
    --num-workers 4 \
    --batch-size 100 \
    --max-triples 1000 \
    --max-eval-triples 100 \
    --num-negatives 5 \
    --max-nodes-per-graph 2000 \
    --rel-degree-dense

if [ $? -ne 0 ]; then
    echo "❌ Test dataset building failed!"
    exit 1
fi

echo "✅ Test dataset built successfully!"

# Test one baseline model (ComplEx - usually fastest)
echo ""
echo "🔬 Testing ComplEx baseline (1 epoch)..."
python run_baseline_comparison.py \
    --data-root './test_comparison_5k_db/' \
    --output-dir './test_results/' \
    --device cuda \
    --quick-mode \
    --models complex

if [ $? -ne 0 ]; then
    echo "❌ Baseline test failed!"
    exit 1
fi

echo "✅ Baseline test completed!"

# Check results
echo ""
echo "📊 Test Results:"
if [ -f "./test_results/baseline_comparison.json" ]; then
    echo "✅ Results file created successfully"
    
    # Extract ComplEx MRR if available
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('./test_results/baseline_comparison.json', 'r') as f:
        results = json.load(f)
    if 'ComplEx' in results and 'test' in results['ComplEx']:
        mrr = results['ComplEx']['test']['mrr']
        hits1 = results['ComplEx']['test']['hits_at_1']
        print(f'ComplEx Test Results: MRR={mrr:.4f}, Hits@1={hits1:.4f}')
    else:
        print('ComplEx results not found in expected format')
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
else
    echo "⚠️ Results file not found"
fi

# Cleanup test files
echo ""
echo "🧹 Cleaning up test files..."
rm -rf ./test_comparison_5k_db/
rm -rf ./test_results/

echo ""
echo "✅ Configuration test completed!"
echo "🚀 Ready to run full comparison with:"
echo "   bash scripts/run_5k_comparison.sh"