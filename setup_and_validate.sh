#!/bin/bash
# MineSLAM Setup and Validation Script
# Runs all validation checks before training

echo "=================================================="
echo "MineSLAM PyTorch Scaffold - Real Data Validation"
echo "=================================================="

# Step 1: Environment Check
echo "🔍 Step 1: Checking environment..."
python scripts/check_env.py
if [ $? -ne 0 ]; then
    echo "❌ Environment check failed!"
    exit 1
fi

# Step 2: Path Validation
echo ""
echo "📁 Step 2: Validating real data paths..."
python scripts/check_paths.py --config configs/mineslam.yaml
if [ $? -ne 0 ]; then
    echo "❌ Path validation failed!"
    exit 1
fi

# Step 3: Bootstrap Test
echo ""
echo "🧪 Step 3: Running bootstrap test with real data..."
python tests/test_bootstrap.py
if [ $? -ne 0 ]; then
    echo "❌ Bootstrap test failed!"
    exit 1
fi

echo ""
echo "✅ ALL VALIDATIONS PASSED!"
echo "Ready for MineSLAM training with verified real data."
echo ""
echo "Next steps:"
echo "  1. Train: python train.py --config configs/mineslam.yaml"
echo "  2. Validate: python val.py --config configs/mineslam.yaml --checkpoint checkpoints/best_model.pth"
echo "  3. Inference: python inference.py --config configs/mineslam.yaml --checkpoint checkpoints/best_model.pth --data_dir /path/to/real/data"