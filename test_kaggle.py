#!/usr/bin/env python3
"""
Test script for Kaggle coding environment
Kiểm tra trainer compatibility và GPU setup
"""
import torch
import sys
import os
from pathlib import Path

def test_kaggle_compatibility():
    """Test basic compatibility for Kaggle environment"""
    print("🧪 Testing RASG compatibility on Kaggle...")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # GPU detection test
    print(f"\n🖥️ GPU Detection:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory/1e9:.1f}GB)")
    else:
        print("Running on CPU")
    
    # PyTorch Geometric test
    try:
        import torch_geometric
        from torch_geometric.data import Data, Batch
        print(f"\n✅ PyG version: {torch_geometric.__version__}")
    except ImportError as e:
        print(f"\n❌ PyG import failed: {e}")
        return False
    
    # Test trainer import
    try:
        from trainer.trainer import run_training, train_one_epoch
        print("✅ Trainer import successful")
    except ImportError as e:
        print(f"❌ Trainer import failed: {e}")
        return False
    
    # Test model import
    try:
        from model.model import RASG
        print("✅ Model import successful")
    except ImportError as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    return True

def test_small_batch():
    """Test với batch nhỏ để verify không có CUDA errors"""
    print("\n🔬 Testing small batch processing...")
    
    try:
        from torch_geometric.data import Data, Batch
        from model.model import RASG
        
        # Tạo dummy data
        num_nodes = 10
        num_relations = 5
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create dummy graph
        x = torch.randn(num_nodes, 16)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        edge_attr = torch.randint(0, num_relations, (20,))
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        batch = Batch.from_data_list([data])
        
        # Create model
        model = RASG(
            num_entities=100,
            num_relations=num_relations,
            node_emb_dim=16,
            rel_emb_dim=16,
            gnn_hidden=32,
            att_dim=32,
            num_layers=2
        )
        
        # Test forward pass
        batch = batch.to(device)
        model = model.to(device)
        relations = torch.randint(0, num_relations, (1,)).to(device)
        
        with torch.no_grad():
            output = model(batch, relations)
            print(f"✅ Forward pass successful: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🚀 RASG Kaggle Compatibility Test")
    print("=" * 60)
    
    # Basic compatibility
    if not test_kaggle_compatibility():
        print("\n❌ Basic compatibility test failed!")
        return False
    
    # Small batch test
    if not test_small_batch():
        print("\n❌ Small batch test failed!")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! RASG is ready for Kaggle")
    print("=" * 60)
    
    # Recommendations for Kaggle
    print("\n📋 Kaggle Setup Recommendations:")
    print("1. Use GPU accelerator if available")
    print("2. Start with small test dataset (5000 triples)")
    print("3. Monitor memory usage during training")
    print("4. Use checkpointing for long training runs")
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if gpu_memory >= 15:
            print(f"\n🎯 Your GPU ({gpu_memory:.1f}GB) is suitable for:")
            print("   - Full dataset training with batch_size=8-16")
            print("   - Model dimensions: gnn_hidden=128, layers=3")
        else:
            print(f"\n⚠️  Your GPU ({gpu_memory:.1f}GB) needs conservative settings:")
            print("   - Reduce batch_size to 4-8")
            print("   - Use gnn_hidden=64-96, layers=2")
    
    return True

if __name__ == "__main__":
    main()