import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from tqdm.auto import tqdm
from torch_geometric.data import Batch
import os
from utils import collate_pyg

def create_mega_batch(pos_graphs, relations, negatives_list):
    """
    Gộp positive và negative vào một batch PyG lớn, vectorized.
    """
    all_graphs = pos_graphs + [g for negs in negatives_list for g in negs]
    all_rel = []
    for i, negs in enumerate(negatives_list):
        all_rel += [relations[i]] * (1 + len(negs))
    batch_all = Batch.from_data_list(all_graphs)
    batch_r = torch.tensor(all_rel, dtype=torch.long)
    batch_size = len(pos_graphs)
    num_negs = len(negatives_list[0]) if negatives_list else 0
    return batch_all, batch_r, batch_size, num_negs

def preprocess_batch(batch, device):
    pos_graphs, relations, negatives_list, _ = batch
    batch_all, batch_r, batch_size, num_negs = create_mega_batch(pos_graphs, relations, negatives_list)
    # CUDA: dùng non_blocking, CPU/MPS bỏ qua param này
    kwargs = {'non_blocking': True} if (torch.cuda.is_available() and str(device).startswith("cuda")) else {}
    batch_all = batch_all.to(device, **kwargs)
    batch_r = batch_r.to(device, **kwargs)
    return batch_all, batch_r, batch_size, num_negs

def train_one_epoch(model, loader, optimizer, device, margin=1.0, max_grad_norm=1.0):
    model.train()
    total_loss, n_samples = 0, 0
    for batch in tqdm(loader, desc='Training'):
        batch_all, batch_r, batch_size, num_negs = preprocess_batch(batch, device)
        scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
        scores_pos, scores_neg = scores[:, 0].unsqueeze(1), scores[:, 1:]
        loss = F.margin_ranking_loss(
            scores_pos.expand_as(scores_neg),
            scores_neg,
            target=torch.ones_like(scores_neg),
            margin=margin, reduction='sum'
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_samples += batch_size
    return total_loss / n_samples

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_ranks = []
    for batch in tqdm(loader, desc='Evaluating'):
        batch_all, batch_r, batch_size, num_negs = preprocess_batch(batch, device)
        scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
        pos_scores, neg_scores = scores[:, 0].unsqueeze(1), scores[:, 1:]
        above = (neg_scores > pos_scores).sum(dim=1)
        equal = (neg_scores == pos_scores).sum(dim=1)
        batch_ranks = 1.0 + above + equal * 0.5
        all_ranks.append(batch_ranks.cpu())
    all_ranks = torch.cat(all_ranks)
    mrr = (1.0 / all_ranks).mean().item()
    hits = [(all_ranks <= k).float().mean().item() for k in [1, 3, 10]]
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
    weight_decay=1e-5,
    patience=10,
    max_grad_norm=1.0,
    num_workers=4,
    checkpoint_path=None,
    logger=None
):
    # DataLoader params auto-tuned cho CUDA/CPU
    pin_memory = torch.cuda.is_available() and str(device).startswith("cuda")
    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=collate_pyg,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=min(batch_size * 4, 256),
        shuffle=False,
        **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size * 4, 256),
        shuffle=False,
        **loader_kwargs
    )
    # Multi-GPU support (2 card trở lên)
    if torch.cuda.device_count() > 1 and str(device).startswith("cuda"):
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = model.to(device)

    # Optimizer: fused chỉ dùng được khi CUDA, torch>=2.0, nên kiểm tra an toàn
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device, margin, max_grad_norm)
        history['train_loss'].append(train_loss)
        val_mrr, val_hits, _ = evaluate(model, valid_loader, device)
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
                print(f"Checkpoint saved at {checkpoint_path}") if logger is None else logger.info(f"Checkpoint saved at {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                msg = f"Early stopping after {patience} epochs without improvement"
                print(msg) if logger is None else logger.info(msg)
                break

    # Final test
    if checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['state_dict'])
        print("Loaded best model for testing") if logger is None else logger.info("Loaded best model for testing")

    print("\n==== Final Evaluation on Test Set ====") if logger is None else logger.info("\n==== Final Evaluation on Test Set ====")
    test_mrr, test_hits, test_ranks = evaluate(model, test_loader, device)
    msg = (f"Test MRR: {test_mrr:.4f} | Hits@1/3/10: {test_hits[0]:.4f}/{test_hits[1]:.4f}/{test_hits[2]:.4f}")
    print(msg) if logger is None else logger.info(msg)
    return {
        'best_val_mrr': best_mrr,
        'test_mrr': test_mrr,
        'test_hits': test_hits,
        'history': history
    }
