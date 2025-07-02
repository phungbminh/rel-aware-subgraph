#!/bin/bash

# Baseline Comparison Script for RASG Research
# Trains and evaluates TransE, ComplEx, RotatE against RASG

echo "🔬 RASG Baseline Comparison"
echo "=========================================="

# Configuration
DATA_ROOT="${1:-./test_5k_db/}"
OUTPUT_DIR="${2:-./baseline_results/}"
DEVICE="${3:-cuda}"

echo "📊 Configuration:"
echo "  Data root: $DATA_ROOT"
echo "  Output dir: $OUTPUT_DIR"  
echo "  Device: $DEVICE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Quick comparison (reduced epochs for testing)
echo ""
echo "🚀 Running quick baseline comparison..."
python run_baseline_comparison.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR/quick_comparison" \
    --device "$DEVICE" \
    --quick-mode \
    --models transe complex rotate rasg

echo ""
echo "✅ Quick comparison completed!"
echo "📁 Results: $OUTPUT_DIR/quick_comparison/"

# Full comparison (if requested)
if [ "$4" = "full" ]; then
    echo ""
    echo "🔬 Running full baseline comparison..."
    python run_baseline_comparison.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR/full_comparison" \
        --device "$DEVICE" \
        --models transe complex rotate rasg
    
    echo ""
    echo "✅ Full comparison completed!"
    echo "📁 Results: $OUTPUT_DIR/full_comparison/"
fi

# Individual model training (if requested)
if [ "$4" = "individual" ]; then
    echo ""
    echo "🎯 Training individual baselines..."
    
    # TransE only
    python run_baseline_comparison.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR/transe_only" \
        --device "$DEVICE" \
        --models transe
    
    # ComplEx only  
    python run_baseline_comparison.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR/complex_only" \
        --device "$DEVICE" \
        --models complex
    
    # RotatE only
    python run_baseline_comparison.py \
        --data-root "$DATA_ROOT" \
        --output-dir "$OUTPUT_DIR/rotate_only" \
        --device "$DEVICE" \
        --models rotate
    
    echo ""
    echo "✅ Individual training completed!"
fi

echo ""
echo "📊 Summary of results:"
if [ -f "$OUTPUT_DIR/quick_comparison/results_table.txt" ]; then
    echo "Quick comparison results:"
    cat "$OUTPUT_DIR/quick_comparison/results_table.txt"
fi

if [ -f "$OUTPUT_DIR/full_comparison/results_table.txt" ]; then
    echo ""
    echo "Full comparison results:"
    cat "$OUTPUT_DIR/full_comparison/results_table.txt"
fi

echo ""
echo "🎉 Baseline comparison pipeline completed!"
echo "📚 For detailed analysis, check the JSON files in each output directory."