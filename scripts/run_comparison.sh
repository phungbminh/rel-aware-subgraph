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
    echo "  -s, --size SIZE        Dataset size: 5k, 20k, 50k, full (default: 5k)"
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
    echo "  5k:   5K train, 500 valid/test, 10 epochs"
    echo "  20k:  20K train, 2K valid/test, 15 epochs"  
    echo "  50k:  50K train, 5K valid/test, 20 epochs"
    echo "  full: No limits, 25 epochs"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Default 5K comparison"
    echo "  $0 -s 20k                            # 20K preset"
    echo "  $0 -t 10000 -v 1000 -e 1000 -p 15   # Custom 10K dataset"
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
    5k)
        TRAIN_LIMIT=5000
        VALID_LIMIT=500
        TEST_LIMIT=500
        EPOCHS=10
        ;;
    20k)
        TRAIN_LIMIT=20000
        VALID_LIMIT=2000
        TEST_LIMIT=2000
        EPOCHS=15
        ;;
    50k)
        TRAIN_LIMIT=50000
        VALID_LIMIT=5000
        TEST_LIMIT=5000
        EPOCHS=20
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
        echo "Valid options: 5k, 20k, 50k, full"
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
    if [ "$DATASET_SIZE" = "5k" ]; then
        MAX_TRIPLES=5000
        MAX_EVAL=500
        MAX_NODES=2000
    elif [ "$DATASET_SIZE" = "20k" ]; then
        MAX_TRIPLES=20000
        MAX_EVAL=2000
        MAX_NODES=3000
    elif [ "$DATASET_SIZE" = "50k" ]; then
        MAX_TRIPLES=50000
        MAX_EVAL=5000
        MAX_NODES=4000
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
        --num-negatives 5 \
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