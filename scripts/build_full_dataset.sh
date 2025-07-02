#!/bin/bash

# Build full OGB-BioKG dataset for RASG
# Usage: bash scripts/build_full_dataset.sh

echo "🚀 Building FULL OGB-BioKG dataset..."

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

echo "✅ Full dataset build completed!"
echo "📊 Dataset stats:"
echo "   - Location: ./full_biokg_db/"
echo "   - Training: ~5M triples" 
echo "   - Validation: ~500K triples"
echo "   - Test: ~500K triples"
echo ""
echo "🚀 Next step: Train with:"
echo "   python main.py --data-root full_biokg_db/ --use-full-dataset"