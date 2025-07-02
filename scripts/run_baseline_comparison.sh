#!/bin/bash

# Updated RASG Baseline Comparison Script
# Uses the new flexible run_baseline_comparison.py

echo "🔬 RASG Baseline Comparison - Updated Script"
echo "============================================"
echo "This script now uses the flexible baseline comparison."
echo "For more options, use run_comparison.sh"
echo ""

# Configuration
DATA_ROOT="${1:-./test_5k_db/}"
OUTPUT_DIR="${2:-./baseline_results/}"
DEVICE="${3:-cuda}"
MODE="${4:-quick}"

echo "📊 Configuration:"
echo "  Data root: $DATA_ROOT"
echo "  Output dir: $OUTPUT_DIR"  
echo "  Device: $DEVICE"
echo "  Mode: $MODE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

case $MODE in
    quick)
        echo ""
        echo "🚀 Running quick baseline comparison (5 epochs)..."
        python run_baseline_comparison.py \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR/quick_comparison" \
            --device "$DEVICE" \
            --quick-mode \
            --models transe complex rotate rasg
        ;;
    
    full)
        echo ""
        echo "🔬 Running full baseline comparison (10 epochs)..."
        python run_baseline_comparison.py \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR/full_comparison" \
            --device "$DEVICE" \
            --epochs 10 \
            --models transe complex rotate rasg
        ;;
    
    individual)
        echo ""
        echo "🎯 Training individual baselines..."
        
        # TransE only
        python run_baseline_comparison.py \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR/transe_only" \
            --device "$DEVICE" \
            --epochs 10 \
            --models transe
        
        # ComplEx only  
        python run_baseline_comparison.py \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR/complex_only" \
            --device "$DEVICE" \
            --epochs 10 \
            --models complex
        
        # RotatE only
        python run_baseline_comparison.py \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR/rotate_only" \
            --device "$DEVICE" \
            --epochs 10 \
            --models rotate
        ;;
    
    *)
        echo "❌ Unknown mode: $MODE"
        echo "Valid modes: quick, full, individual"
        exit 1
        ;;
esac

if [ $? -ne 0 ]; then
    echo "❌ Baseline comparison failed!"
    exit 1
fi

echo ""
echo "✅ Baseline comparison completed!"

# Display results
echo ""
echo "📊 Summary of results:"
case $MODE in
    quick)
        RESULT_DIR="$OUTPUT_DIR/quick_comparison"
        ;;
    full)
        RESULT_DIR="$OUTPUT_DIR/full_comparison"
        ;;
    individual)
        echo "Individual results saved in separate directories"
        RESULT_DIR=""
        ;;
esac

if [ -n "$RESULT_DIR" ] && [ -f "$RESULT_DIR/results_table.txt" ]; then
    cat "$RESULT_DIR/results_table.txt"
elif [ -n "$RESULT_DIR" ] && [ -f "$RESULT_DIR/baseline_comparison.json" ]; then
    echo "JSON results available at: $RESULT_DIR/baseline_comparison.json"
fi

echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "💡 For more flexible options with custom dataset sizes, use:"
echo "   ./scripts/run_comparison.sh --help"
echo ""
echo "Usage examples of this script:"
echo "  $0                                    # Quick 5K comparison"
echo "  $0 ./data ./results cuda full        # Full 5K comparison" 
echo "  $0 ./data ./results cuda individual  # Individual model training"