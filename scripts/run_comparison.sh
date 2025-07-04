#!/bin/bash

# RASG Flexible Baseline Comparison Script
# Configurable dataset size and training parameters

# Default parameters
DATASET_SIZE="5k"
TRAIN_LIMIT=5000
VALID_LIMIT=500
TEST_LIMIT=500
EPOCHS=10
BATCH_SIZE=256
MODELS="transe complex rotate rasg"
DEVICE="cuda"
OUTPUT_PREFIX="comparison"

# Help function
show_help() {
    echo "🔬 RASG Flexible Baseline Comparison Script"
    echo "==========================================="
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -s, --size SIZE        Dataset size: 1k, 5k, 10k, full (default: 5k)"
    echo "  -t, --train NUM        Training samples limit (default: 5000, -1 for no limit)"
    echo "  -v, --valid NUM        Validation samples limit (default: 500, -1 for no limit)"
    echo "  -e, --test NUM         Test samples limit (default: 500, -1 for no limit)"
    echo "  -p, --epochs NUM       Number of epochs (default: 10)"
    echo "  -b, --batch-size NUM   Batch size (default: 256)"
    echo "  -m, --models MODELS    Space-separated model list (default: 'transe complex rotate rasg')"
    echo "  -d, --device DEVICE    Device: cuda/cpu (default: cuda)"
    echo "  -o, --output PREFIX    Output directory prefix (default: comparison)"
    echo "  -q, --quick            Quick mode (5 epochs, small eval batch)"
    echo "  -h, --help             Show this help"
    echo ""
    echo "Preset configurations:"
    echo "  1k:   1K train, 100 valid/test, 8 epochs (quick testing)"
    echo "  5k:   5K train, 500 valid/test, 10 epochs (development)"
    echo "  10k:  8K train, 800 valid/test, 12 epochs (research)"
    echo "  full: No limits, 25 epochs (complete evaluation)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default 5K comparison"
    echo "  $0 -s 1k                             # Quick 1K test"
    echo "  $0 -s 10k                            # Research 10K dataset"
    echo "  $0 -s full -p 30                     # Full dataset, 30 epochs"
    echo "  $0 -m 'transe complex' -q            # Quick test with 2 models"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--size)
            DATASET_SIZE="$2"
            shift 2
            ;;
        -t|--train)
            TRAIN_LIMIT="$2"
            shift 2
            ;;
        -v|--valid)
            VALID_LIMIT="$2"
            shift 2
            ;;
        -e|--test)
            TEST_LIMIT="$2"
            shift 2
            ;;
        -p|--epochs)
            EPOCHS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_PREFIX="$2"
            shift 2
            ;;
        -q|--quick)
            EPOCHS=5
            BATCH_SIZE=128
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Apply preset configurations
case $DATASET_SIZE in
    1k)
        TRAIN_LIMIT=1000
        VALID_LIMIT=100
        TEST_LIMIT=100
        EPOCHS=8
        ;;
    5k)
        TRAIN_LIMIT=5000
        VALID_LIMIT=500
        TEST_LIMIT=500
        EPOCHS=10
        ;;
    10k)
        TRAIN_LIMIT=10000
        VALID_LIMIT=1000
        TEST_LIMIT=1000
        EPOCHS=15
        ;;
    full)
        TRAIN_LIMIT=-1
        VALID_LIMIT=-1
        TEST_LIMIT=-1
        EPOCHS=25
        ;;
    custom)
        # Keep user-specified values
        ;;
    *)
        echo "❌ Unknown dataset size: $DATASET_SIZE"
        echo "Valid options: 1k, 5k, 10k, full"
        exit 1
        ;;
esac

# Set output directories
DATA_DIR="./comparison_${DATASET_SIZE}_db"
OUTPUT_DIR="./${OUTPUT_PREFIX}_results_${DATASET_SIZE}"

echo "🔬 RASG Baseline Comparison - ${DATASET_SIZE^^} Dataset"
echo "=" $(printf '=%.0s' {1..50})
echo "Configuration:"
echo "  📊 Dataset: ${DATASET_SIZE} (Train: ${TRAIN_LIMIT}, Valid: ${VALID_LIMIT}, Test: ${TEST_LIMIT})"
echo "  🚀 Training: ${EPOCHS} epochs, batch_size=${BATCH_SIZE}"
echo "  🔧 Models: ${MODELS}"
echo "  💻 Device: ${DEVICE}"
echo "  📁 Output: ${OUTPUT_DIR}"
echo ""

# Step 1: Build dataset if needed
if [ ! -d "$DATA_DIR" ]; then
    echo "📦 Building ${DATASET_SIZE} dataset..."
    
    # Set parameters based on dataset size
    if [ "$DATASET_SIZE" = "1k" ]; then
        MAX_TRIPLES=1000
        MAX_EVAL=100
        MAX_NODES=1000
    elif [ "$DATASET_SIZE" = "5k" ]; then
        MAX_TRIPLES=5000
        MAX_EVAL=500
        MAX_NODES=2000
    elif [ "$DATASET_SIZE" = "10k" ]; then
        MAX_TRIPLES=8000   # Giảm để tránh memory issues
        MAX_EVAL=800       # Giảm eval size  
        MAX_NODES=2000     # Giảm nodes per subgraph
    else
        # Full dataset
        MAX_TRIPLES=-1
        MAX_EVAL=-1
        MAX_NODES=5000
    fi
    
    python build_subgraph.py \
        --ogb-root './data/ogb/' \
        --output-dir "$DATA_DIR" \
        --k 2 \
        --tau 2 \
        --num-workers 6 \
        --batch-size 150 \
        --max-triples $MAX_TRIPLES \
        --max-eval-triples $MAX_EVAL \
        --num-negatives 50 \
        --max-nodes-per-graph $MAX_NODES \
        --rel-degree-dense

    if [ $? -ne 0 ]; then
        echo "❌ Dataset building failed!"
        exit 1
    fi
    
    echo "✅ Dataset built successfully!"
    echo ""
else
    echo "📦 Using existing dataset: $DATA_DIR"
    echo ""
fi

# Step 2: Run baseline comparison
echo "🚀 Starting baseline model comparison..."
python run_baseline_comparison.py \
    --data-root "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --train-limit $TRAIN_LIMIT \
    --valid-limit $VALID_LIMIT \
    --test-limit $TEST_LIMIT \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --models $MODELS

if [ $? -ne 0 ]; then
    echo "❌ Baseline comparison failed!"
    exit 1
fi

echo ""
echo "✅ Baseline comparison completed!"

# Step 3: Display results
echo ""
echo "📊 COMPARISON RESULTS:"
echo "=" $(printf '=%.0s' {1..60})

if [ -f "$OUTPUT_DIR/results_table.txt" ]; then
    echo "Results Table:"
    cat "$OUTPUT_DIR/results_table.txt"
else
    echo "Results table not found, checking JSON..."
    if [ -f "$OUTPUT_DIR/baseline_comparison.json" ]; then
        echo "JSON results available at: $OUTPUT_DIR/baseline_comparison.json"
        # Show basic results
        python -c "
import json
try:
    with open('$OUTPUT_DIR/baseline_comparison.json') as f:
        results = json.load(f)
    print('\\nQuick Results:')
    for model, data in results.items():
        if 'test' in data and 'mrr' in data['test']:
            mrr = data['test']['mrr']
            h1 = data['test'].get('hits_at_1', 0)
            print(f'  {model}: MRR={mrr:.4f}, Hits@1={h1:.4f}')
except Exception as e:
    print(f'Could not parse results: {e}')
"
    fi
fi

echo ""
echo "📁 Results saved to: $OUTPUT_DIR"
echo "📊 Dataset: $DATA_DIR"
echo ""
echo "🎯 Configuration summary:"
echo "  - Dataset size: ${DATASET_SIZE} (${TRAIN_LIMIT} train, ${VALID_LIMIT} valid, ${TEST_LIMIT} test)"
echo "  - Training: ${EPOCHS} epochs, batch_size=${BATCH_SIZE}"
echo "  - Models: ${MODELS}"
echo "  - Device: ${DEVICE}"
echo ""
echo "💡 Kaggle-optimized presets:"
echo "  - 1K: ~15min total (quick testing)"
echo "  - 5K: ~6h total (development)"  
echo "  - 10K: ~12h total (research)"