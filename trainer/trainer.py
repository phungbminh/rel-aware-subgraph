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

def train_one_epoch(model, loader, optimizer, device, margin=1.0, max_grad_norm=1.0, is_debug=False):
    model.train()
    total_loss, n_samples = 0, 0
    if is_debug:
        print("[DEBUG][train_one_epoch] Start training")
    for step, batch in enumerate(tqdm(loader, desc='Training')):
        print(torch.cuda.memory_allocated() / 1024 ** 3, "GB allocated")
        print(torch.cuda.memory_reserved() / 1024 ** 3, "GB reserved")
        if is_debug:
            print(f"[DEBUG][train_one_epoch] Got batch {step}")
        batch_all, batch_r, batch_size, num_negs = preprocess_batch(batch, device, is_debug=is_debug)

        if step == 0 or is_debug:
            print("[DEBUG][check] Model device:", next(model.parameters()).device)
            if hasattr(batch_all, 'x'):
                print("[DEBUG][check] batch_all.x device:", batch_all.x.device)
            if hasattr(batch_all, 'edge_index'):
                print("[DEBUG][check] batch_all.edge_index device:", batch_all.edge_index.device)
            print("[DEBUG][check] batch_r device:", batch_r.device)

        if is_debug:
            print(f"[DEBUG][train_one_epoch] batch_all: {batch_all.x.shape if hasattr(batch_all, 'x') else None}, batch_r: {batch_r.shape}, batch_size: {batch_size}, num_negs: {num_negs}")
        scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
        if is_debug:
            print(f"[DEBUG][train_one_epoch] scores shape: {scores.shape}")
        scores_pos, scores_neg = scores[:, 0].unsqueeze(1), scores[:, 1:]
        loss = F.margin_ranking_loss(
            scores_pos.expand_as(scores_neg),
            scores_neg,
            target=torch.ones_like(scores_neg),
            margin=margin, reduction='sum'
        )
        if is_debug:
            print(f"[DEBUG][train_one_epoch] loss: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
        n_samples += batch_size
    if is_debug:
        print("[DEBUG][train_one_epoch] End training")
    return total_loss / n_samples

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
        scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
        pos_scores, neg_scores = scores[:, 0].unsqueeze(1), scores[:, 1:]
        above = (neg_scores > pos_scores).sum(dim=1)
        equal = (neg_scores == pos_scores).sum(dim=1)
        batch_ranks = 1.0 + above + equal * 0.5
        all_ranks.append(batch_ranks.cpu())
        if is_debug and step == 0:
            print(f"[DEBUG][evaluate] batch_ranks: {batch_ranks}")
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
    weight_decay=1e-5,
    patience=10,
    max_grad_norm=1.0,
    num_workers=0,         # NÊN để 0 nếu dùng LMDB!
    checkpoint_path=None,
    logger=None,
    is_debug=False
):
    pin_memory = torch.cuda.is_available() and str(device).startswith("cuda")
    loader_kwargs = dict(
        num_workers=num_workers,
        collate_fn=GraphPackedBatch.from_data_list,
        pin_memory=pin_memory,
        persistent_workers=True,  # Không persistent khi debug hoặc với LMDB
        prefetch_factor=2
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
    if is_debug:
        print(f"[DEBUG][run_training] Loader created. Pin memory: {pin_memory}, num_workers: {num_workers}")

    # if torch.cuda.device_count() > 1 and str(device).startswith("cuda"):
    #     #model = torch.nn.DataParallel(model)
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0, 1])
    #     print(f"[DEBUG][run_training] Using {torch.cuda.device_count()} GPUs with DataParallel!")
    # model = model.to(device)

    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
        print(f"[DEBUG][run_training] Using DistributedDataParallel on {torch.cuda.device_count()} GPUs!")
    else:
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
        train_loss = train_one_epoch(model, train_loader, optimizer, device, margin, max_grad_norm, is_debug=is_debug)
        if is_debug:
            print(f"[DEBUG][run_training] Epoch {epoch}: train_loss = {train_loss}")
        history['train_loss'].append(train_loss)
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
