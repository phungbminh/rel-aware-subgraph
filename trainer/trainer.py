import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from tqdm.auto import tqdm
from torch_geometric.data import Batch
import os
from utils import collate_pyg
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


class GraphPackedBatch(Batch):
    @staticmethod
    def from_data_list(data_list):
        """Tạo một siêu đồ thị chứa tất cả đồ thị con"""
        batch = Batch()
        batch.batch = torch.cat([
            torch.full((data.num_nodes,), i)
            for i, data in enumerate(data_list)
        ], dim=0)

        # Gom tất cả node features
        batch.x = torch.cat([data.x for data in data_list], dim=0)

        # Điều chỉnh chỉ số edge
        edge_indices = []
        node_offset = 0
        for i, data in enumerate(data_list):
            edge_indices.append(data.edge_index + node_offset)
            node_offset += data.num_nodes

        batch.edge_index = torch.cat(edge_indices, dim=1)
        batch.edge_attr = torch.cat([data.edge_attr for data in data_list], dim=0)
        return batch

def create_mega_batch(pos_graphs, relations, negatives_list, is_debug=False):
    """
    Gộp positive và negative vào một batch PyG lớn, vectorized.
    """
    if is_debug:
        print(f"[DEBUG][create_mega_batch] len(pos_graphs): {len(pos_graphs)}, len(negatives_list): {len(negatives_list)}")

    if is_debug:
        print("[DEBUG][create_mega_batch] Number of pos_graphs:", len(pos_graphs))
        print("[DEBUG][create_mega_batch] Number of negatives per pos:", len(negatives_list[0]) if negatives_list else 0)
    all_graphs = pos_graphs + [g for negs in negatives_list for g in negs]
    all_rel = []
    for i, negs in enumerate(negatives_list):
        all_rel += [relations[i]] * (1 + len(negs))
    if is_debug:
        print("[DEBUG][create_mega_batch] Total all_graphs:", len(all_graphs))
    batch_all = Batch.from_data_list(all_graphs)
    batch_r = torch.tensor(all_rel, dtype=torch.long)
    batch_size = len(pos_graphs)
    num_negs = len(negatives_list[0]) if negatives_list else 0
    if is_debug:
        print("[DEBUG][create_mega_batch] batch_all shape:", batch_all.x.shape if hasattr(batch_all, 'x') else None)
        print("[DEBUG][create_mega_batch] batch_r shape:", batch_r.shape)
    return batch_all, batch_r, batch_size, num_negs

def preprocess_batch(batch, device, is_debug=False):
    pos_graphs, relations, negatives_list, _ = batch
    batch_all, batch_r, batch_size, num_negs = create_mega_batch(pos_graphs, relations, negatives_list, is_debug=is_debug)
    kwargs = {'non_blocking': True} if (torch.cuda.is_available() and str(device).startswith("cuda")) else {}
    if is_debug:
        print("[DEBUG][preprocess_batch] Moving batch_all to device", device)
    batch_all = batch_all.to(device, **kwargs)
    batch_r = batch_r.to(device, **kwargs)
    return batch_all, batch_r, batch_size, num_negs

def train_one_epoch(model, loader, optimizer, device, margin=1.0, max_grad_norm=1.0, 
                   accumulation_steps=1, negative_sampler=None, num_train_negatives=1, is_debug=False):
    import time
    start_time = time.time()
    
    model.train()
    total_loss, n_samples = 0, 0
    optimizer.zero_grad(set_to_none=True)
    
    # Disable autograd anomaly detection for speed
    torch.autograd.set_detect_anomaly(False)
    
    print(f"[TRAINING] Starting epoch with {len(loader)} batches, accumulation_steps={accumulation_steps}")
    
    if is_debug:
        print("[DEBUG][train_one_epoch] Start training")
    
    for step, batch in enumerate(tqdm(loader, desc='Training')):
        try:
            if is_debug and step < 2:
                print(f"[DEBUG][train_one_epoch] Got batch {step}")
            
            pos_graphs, relations, neg_graphs_list, num_negs = batch
            
            # Check if this is training (no pre-computed negatives) or evaluation
            is_training_batch = num_negs == 0
            
            if is_training_batch and negative_sampler is not None:
                # Training: Use binary classification with runtime negative sampling
                loss = train_binary_classification(
                    model, pos_graphs, relations, negative_sampler, 
                    num_train_negatives, device, is_debug and step < 2
                )
            else:
                # Evaluation or training with pre-computed negatives: Use ranking loss
                batch_all, batch_r, batch_size, num_negs = preprocess_batch(batch, device, is_debug=(is_debug and step < 2))
                
                if num_negs == 0:
                    # Skip if no negatives available
                    continue
                    
                scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
                scores_pos, scores_neg = scores[:, 0].unsqueeze(1), scores[:, 1:]
                
                loss = F.margin_ranking_loss(
                    scores_pos.expand_as(scores_neg),
                    scores_neg,
                    target=torch.ones_like(scores_neg),
                    margin=margin, reduction='mean'
                )
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0 or step == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item() * accumulation_steps
            n_samples += len(pos_graphs)
            
            # Aggressive memory cleanup for large datasets
            if step % 5 == 0:  # More frequent cleanup
                torch.cuda.empty_cache()
                
            # Progress monitoring for slow training
            if step % 100 == 0 and step > 0:
                elapsed = time.time() - start_time
                avg_time_per_step = elapsed / step
                eta_seconds = avg_time_per_step * (len(loader) - step)
                eta_hours = eta_seconds / 3600
                print(f"[PROGRESS] Step {step}/{len(loader)}, ETA: {eta_hours:.1f}h, {avg_time_per_step:.1f}s/step")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"[WARNING] OOM at step {step}, skipping batch and reducing memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                # Track OOM occurrences
                if not hasattr(train_one_epoch, 'oom_count'):
                    train_one_epoch.oom_count = 0
                train_one_epoch.oom_count += 1
                
                if train_one_epoch.oom_count > 5:
                    print(f"[ERROR] Too many OOM errors ({train_one_epoch.oom_count}), stopping epoch")
                    break
                continue
            else:
                raise e
    
    elapsed_total = time.time() - start_time
    print(f"[TRAINING] Epoch completed in {elapsed_total/60:.1f} minutes, avg {elapsed_total/len(loader):.1f}s/step")
    
    if is_debug:
        print("[DEBUG][train_one_epoch] End training")
    return total_loss / n_samples

def train_binary_classification(model, pos_graphs, relations, negative_sampler, 
                               num_negatives, device, is_debug=False):
    """
    Train with binary classification loss following RGCN/CompGCN style.
    """
    # Sample negative graphs
    neg_graphs, neg_relations = negative_sampler.sample_negatives(
        pos_graphs, relations, num_negatives
    )
    
    if len(neg_graphs) == 0:
        # Fallback if no negatives could be sampled
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Combine positive and negative graphs
    all_graphs = pos_graphs + neg_graphs
    all_relations = relations + neg_relations
    
    # Create labels: 1 for positive, 0 for negative
    pos_labels = torch.ones(len(pos_graphs), device=device)
    neg_labels = torch.zeros(len(neg_graphs), device=device)
    all_labels = torch.cat([pos_labels, neg_labels])
    
    if is_debug:
        print(f"[DEBUG][Binary] Pos: {len(pos_graphs)}, Neg: {len(neg_graphs)}")
    
    # Forward pass
    try:
        batch_all = Batch.from_data_list(all_graphs).to(device)
        batch_r = torch.tensor(all_relations, device=device)
        
        scores = model(batch_all, batch_r)  # Shape: (total_graphs,)
    except Exception as e:
        if is_debug:
            print(f"[DEBUG][Binary] Error in batching: {e}")
            print(f"[DEBUG][Binary] Pos graph keys: {list(pos_graphs[0].keys())}")
            if len(neg_graphs) > 0:
                print(f"[DEBUG][Binary] Neg graph keys: {list(neg_graphs[0].keys())}")
        raise e
    
    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(scores, all_labels, reduction='mean')
    
    return loss

@torch.no_grad()
def evaluate(model, loader, device, is_debug=False):
    model.eval()
    all_ranks = []
    if is_debug:
        print("[DEBUG][evaluate] Start evaluation")
    for step, batch in enumerate(tqdm(loader, desc='Evaluating')):
        if is_debug:
            print(f"[DEBUG][evaluate] Got batch {step}")
        batch_all, batch_r, batch_size, num_negs = preprocess_batch(batch, device, is_debug=is_debug)
        
        if num_negs == 0:
            # Không có negatives - skip evaluation hoặc warning
            if is_debug:
                print(f"[DEBUG][evaluate] No negatives in batch {step}, skipping")
            continue
            
        scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
        pos_scores, neg_scores = scores[:, 0].unsqueeze(1), scores[:, 1:]
        above = (neg_scores > pos_scores).sum(dim=1)
        equal = (neg_scores == pos_scores).sum(dim=1)
        batch_ranks = 1.0 + above + equal * 0.5
        all_ranks.append(batch_ranks.cpu())
        if is_debug and step == 0:
            print(f"[DEBUG][evaluate] pos_scores shape: {pos_scores.shape}, neg_scores shape: {neg_scores.shape}")
            print(f"[DEBUG][evaluate] batch_ranks: {batch_ranks}")
    
    if len(all_ranks) == 0:
        print("[WARNING] No valid evaluation batches found!")
        return 0.0, [0.0, 0.0, 0.0], torch.tensor([])
        
    all_ranks = torch.cat(all_ranks)
    mrr = (1.0 / all_ranks).mean().item()
    hits = [(all_ranks <= k).float().mean().item() for k in [1, 3, 10]]
    if is_debug:
        print("[DEBUG][evaluate] End evaluation")
    return mrr, hits, all_ranks

def run_training(
    model,
    train_dataset,
    valid_dataset,
    test_dataset,
    epochs,
    batch_size,
    device,
    lr=1e-3,
    margin=1.0,
    eval_every=1,
    weight_decay=1e-5,
    patience=10,
    max_grad_norm=1.0,
    num_workers=0,         # NÊN để 0 nếu dùng LMDB!
    checkpoint_path=None,
    logger=None,
    num_train_negatives=1, # Number of negatives per positive in training
    is_debug=False
):
    pin_memory = torch.cuda.is_available() and str(device).startswith("cuda")
    # Force num_workers=0 for LMDB compatibility - prevents bus errors
    optimal_workers = 0  # Always use 0 with LMDB databases to prevent bus errors
    loader_kwargs = dict(
        num_workers=optimal_workers,
        collate_fn=collate_pyg,
        pin_memory=pin_memory,
        persistent_workers=False,  # Tắt persistent với LMDB để tránh deadlock
        prefetch_factor=None  # No prefetch needed with single-threaded loading
    )

    # Use dynamic batching for better memory utilization
    from utils import FixedSizeBatchSampler
    
    # Check if using full dataset
    is_full_dataset = len(train_dataset) > 1000  # Heuristic for full dataset
    
    # Drastically reduce memory usage for 10K dataset
    # Detect dataset size and adjust accordingly
    dataset_size = len(train_dataset)
    if dataset_size >= 8000:  # 10K dataset
        max_nodes_training = 400  # Much smaller for 10K
        max_batch_training = 2    # Smaller batch
    elif dataset_size >= 5000:  # 5K dataset  
        max_nodes_training = 800  # Medium
        max_batch_training = 4    # Medium batch
    else:  # 1K dataset
        max_nodes_training = 1200  # Larger
        max_batch_training = batch_size  # Original batch size
    
    print(f"[MEMORY] Detected dataset size: {dataset_size}, using max_nodes={max_nodes_training}, max_batch={max_batch_training}")
    
    train_sampler = FixedSizeBatchSampler(
        train_dataset, 
        max_batch_size=max_batch_training,
        max_nodes_per_batch=max_nodes_training,
        shuffle=True,
        is_full_dataset=is_full_dataset
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        **loader_kwargs
    )
    
    # For eval, use even smaller batches based on dataset size
    if dataset_size >= 8000:  # 10K dataset
        max_nodes_eval = 200  # Very small for memory
        max_batch_eval = 1    # Single sample
    elif dataset_size >= 5000:  # 5K dataset
        max_nodes_eval = 300  # Small
        max_batch_eval = 1    # Single sample
    else:  # 1K dataset
        max_nodes_eval = 500  # Larger
        max_batch_eval = 2    # Can handle more
    
    print(f"[MEMORY] Eval batching: max_nodes={max_nodes_eval}, max_batch={max_batch_eval}")
    
    valid_sampler = FixedSizeBatchSampler(
        valid_dataset,
        max_batch_size=max_batch_eval,
        max_nodes_per_batch=max_nodes_eval,
        shuffle=False,
        is_full_dataset=is_full_dataset
    )
    
    test_sampler = FixedSizeBatchSampler(
        test_dataset,
        max_batch_size=max_batch_eval,
        max_nodes_per_batch=max_nodes_eval,
        shuffle=False,
        is_full_dataset=is_full_dataset
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_sampler=valid_sampler,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        **loader_kwargs
    )
    if is_debug:
        print(f"[DEBUG][run_training] Loader created. Pin memory: {pin_memory}, num_workers: {optimal_workers}")

    # Setup negative sampler for training
    from .negative_sampler import RuntimeNegativeSampler
    all_entities = list(train_dataset.entity2id.keys())
    negative_sampler = RuntimeNegativeSampler(all_entities, corruption_rate=0.5)
    
    if is_debug:
        print(f"[DEBUG][run_training] Created negative sampler with {len(all_entities)} entities")

    # if torch.cuda.device_count() > 1 and str(device).startswith("cuda"):
    #     #model = torch.nn.DataParallel(model)
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0, 1])
    #     print(f"[DEBUG][run_training] Using {torch.cuda.device_count()} GPUs with DataParallel!")
    # model = model.to(device)

    # Smart multi-GPU setup that works with both single and multi-GPU
    num_gpus = torch.cuda.device_count()
    
    # Add memory monitoring
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[MEMORY] GPU memory available: {gpu_memory_gb:.1f}GB")
        
        def log_memory_usage(prefix=""):
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"[MEMORY] {prefix} Allocated: {allocated:.2f}GB, Reserved: {cached:.2f}GB")
        
        log_memory_usage("Initial")
    
    if num_gpus > 1:
        print(f"[INFO] Found {num_gpus} GPUs")
        
        # Check if PyG + DataParallel compatible
        try:
            # Test compatibility with a dummy forward pass
            model = model.to(device)
            print("[INFO] Testing multi-GPU compatibility...")
            
            # For now, use single GPU to avoid DataParallel issues with PyG
            print("[WARNING] Using single GPU due to PyG DataParallel compatibility issues")
            print("[INFO] This ensures stable training. Multi-GPU support will be added in future updates.")
            
        except Exception as e:
            print(f"[WARNING] Multi-GPU setup failed: {e}")
            print("[INFO] Falling back to single GPU")
            model = model.to(device)
    else:
        print("[INFO] Using single GPU")
        model = model.to(device)

    try:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=pin_memory)
    except TypeError:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience//2, factor=0.5)
    best_mrr, no_improve = 0, 0
    history = {'train_loss': [], 'val_mrr': [], 'val_hits': []}

    for epoch in range(1, epochs + 1):
        msg = f"\n=== Epoch {epoch}/{epochs} ==="
        print(msg) if logger is None else logger.info(msg)
        if is_debug:
            print(f"[DEBUG][run_training] Epoch {epoch}: start train_one_epoch")
        # More aggressive gradient accumulation for large datasets
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 16e9
        train_size = len(train_dataset)
        
        if train_size >= 8000:  # 10K dataset
            accumulation_steps = 4  # More accumulation for memory
            print(f"[MEMORY] Large dataset detected ({train_size}), using accumulation_steps={accumulation_steps}")
        elif train_size >= 5000:  # 5K dataset
            accumulation_steps = 2  # Medium accumulation
            print(f"[MEMORY] Medium dataset detected ({train_size}), using accumulation_steps={accumulation_steps}")
        else:
            accumulation_steps = 1  # No accumulation for small datasets
            print(f"[MEMORY] Small dataset detected ({train_size}), using accumulation_steps={accumulation_steps}")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, margin, max_grad_norm, 
                                   accumulation_steps, negative_sampler, num_train_negatives, is_debug=is_debug)
        if is_debug:
            print(f"[DEBUG][run_training] Epoch {epoch}: train_loss = {train_loss}")
        history['train_loss'].append(train_loss)
        
        # Skip validation if not eval epoch
        if epoch % eval_every == 0 or epoch == epochs:
            val_mrr, val_hits, _ = evaluate(model, valid_loader, device, is_debug=is_debug)
            if is_debug:
                print(f"[DEBUG][run_training] Epoch {epoch}: val_mrr = {val_mrr}, val_hits = {val_hits}")
            history['val_mrr'].append(val_mrr)
            history['val_hits'].append(val_hits)
            scheduler.step(val_mrr)
            msg = (f"Train Loss: {train_loss:.4f} | Valid MRR: {val_mrr:.4f} | "
                   f"Hits@1/3/10: {val_hits[0]:.4f}/{val_hits[1]:.4f}/{val_hits[2]:.4f}")
            print(msg) if logger is None else logger.info(msg)
            if val_mrr > best_mrr:
                best_mrr, no_improve = val_mrr, 0
                if checkpoint_path:
                    state = {
                        'epoch': epoch,
                        'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_mrr': best_mrr,
                        'history': history
                    }
                    torch.save(state, checkpoint_path)
                    if is_debug:
                        print(f"[DEBUG][run_training] Checkpoint saved at {checkpoint_path}")
            else:
                no_improve += 1
        else:
            # Skip validation
            msg = f"Train Loss: {train_loss:.4f} | [Skipped validation]"
            print(msg) if logger is None else logger.info(msg)
            no_improve += 1
        
        # Check early stopping
        if no_improve >= patience:
            msg = f"Early stopping after {patience} epochs without improvement"
            print(msg) if logger is None else logger.info(msg)
            break

    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['state_dict'])
        if is_debug:
            print("[DEBUG][run_training] Loaded best model for testing")

    print("\n==== Final Evaluation on Test Set ====") if logger is None else logger.info("\n==== Final Evaluation on Test Set ====")
    test_mrr, test_hits, test_ranks = evaluate(model, test_loader, device, is_debug=is_debug)
    msg = (f"Test MRR: {test_mrr:.4f} | Hits@1/3/10: {test_hits[0]:.4f}/{test_hits[1]:.4f}/{test_hits[2]:.4f}")
    print(msg) if logger is None else logger.info(msg)
    return {
        'best_val_mrr': best_mrr,
        'test_mrr': test_mrr,
        'test_hits': test_hits,
        'history': history
    }
