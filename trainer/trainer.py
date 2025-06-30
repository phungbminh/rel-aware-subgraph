import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, lr_scheduler
from tqdm import tqdm
from torch_geometric.data import Batch
from utils.pyg_utils import collate_pyg, move_batch_to_device_pyg
from torch_geometric.data import Batch, Data
# =======================
# HÀM GỘP BATCH
# =======================
def create_mega_batch(pos_graphs, relations, negatives_list, num_negs):
    """
    Tạo batch lớn gồm tất cả positive và negative graphs.
    - pos_graphs: list[Data]
    - relations: list[int]
    - negatives_list: list[list[Data]]
    - num_negs: int (số negative mỗi positive)

    Return:
        - batch_all: Batch
        - batch_r: tensor relation ids
        - batch_size: số positive
        - num_negs: số negative mỗi positive
    """
    all_graphs = []
    all_rel = []
    batch_size = len(pos_graphs)
    for i in range(batch_size):
        # Positive
        all_graphs.append(pos_graphs[i])
        all_rel.append(relations[i])
        # Negative
        for neg in negatives_list[i]:
            all_graphs.append(neg)
            all_rel.append(relations[i])
    # Đảm bảo all_graphs chỉ toàn Data object
    for idx, g in enumerate(all_graphs):
        if not isinstance(g, Data):
            print(f"ERROR: all_graphs[{idx}] is not Data but {type(g)}")
            raise TypeError(f"all_graphs[{idx}] is {type(g)}")
    batch_all = Batch.from_data_list(all_graphs)
    batch_r = torch.tensor(all_rel, dtype=torch.long)
    return batch_all, batch_r, batch_size, num_negs

def get_example(batch, i):
    """
    Lấy graph thứ i từ PyG Batch (hàm get_example nếu có, hoặc index thông thường)
    """
    return batch.get_example(i) if hasattr(batch, 'get_example') else batch[i]

# =======================
# HUẤN LUYỆN 1 EPOCH
# =======================
def train_one_epoch(model, loader, optimizer, device, margin=1.0, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    n_samples = 0
    for pos_graphs, relations, negatives_list, num_negs in tqdm(loader, desc='Training'):
        # Chuyển relations thành tensor
        relations = torch.tensor(relations, dtype=torch.long)
        batch_all, batch_r, batch_size, num_negs = create_mega_batch(pos_graphs, relations, negatives_list, num_negs)
        batch_all = batch_all.to(device)
        batch_r = batch_r.to(device)
        scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
        scores_pos = scores[:, 0].unsqueeze(1)
        scores_neg = scores[:, 1:]
        loss = F.margin_ranking_loss(
            scores_pos.expand_as(scores_neg),
            scores_neg,
            target=torch.ones_like(scores_neg),
            margin=margin
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * batch_size
        n_samples += batch_size
    return total_loss / n_samples
# =======================
# ĐÁNH GIÁ (VALID/TEST)
# =======================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_ranks = []
    for pos_graphs, relations, negatives_list, num_negs in tqdm(loader, desc='Evaluating'):
        relations = torch.tensor(relations, dtype=torch.long)
        batch_all, batch_r, batch_size, num_negs = create_mega_batch(pos_graphs, relations, negatives_list, num_negs)
        batch_all = batch_all.to(device)
        batch_r = batch_r.to(device)
        scores = model(batch_all, batch_r).reshape(batch_size, 1 + num_negs)
        pos_scores = scores[:, 0].unsqueeze(1)
        neg_scores = scores[:, 1:]
        above = (neg_scores > pos_scores).sum(dim=1)
        equal = (neg_scores == pos_scores).sum(dim=1)
        batch_ranks = 1.0 + above + equal * 0.5
        all_ranks.append(batch_ranks.cpu())
    all_ranks = torch.cat(all_ranks)
    mrr = (1.0 / all_ranks).mean().item()
    hits = [(all_ranks <= k).float().mean().item() for k in [1, 3, 10]]
    return mrr, hits, all_ranks
# =======================
# QUY TRÌNH TRAIN FULL
# =======================
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
    patience=6,
    max_grad_norm=1.0,
    num_workers=4,
    use_amp=False,
    checkpoint_path="best_model.pt"
):
    """
    Toàn bộ quy trình train/validate/test + early stopping + checkpoint.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_pyg,
        num_workers=0,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=min(batch_size*4, 256),
        shuffle=False,
        collate_fn=collate_pyg,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=min(batch_size*4, 256),
        shuffle=False,
        collate_fn=collate_pyg,
        num_workers=0
    )
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs!")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience//2, factor=0.5)
    best_mrr, no_improve = 0, 0
    history = {'train_loss': [], 'val_mrr': [], 'val_hits': []}
    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, margin, max_grad_norm
        )
        history['train_loss'].append(train_loss)
        val_mrr, val_hits, _ = evaluate(model, valid_loader, device)
        history['val_mrr'].append(val_mrr)
        history['val_hits'].append(val_hits)
        scheduler.step(val_mrr)
        print(f"Train Loss: {train_loss:.4f} | Valid MRR: {val_mrr:.4f}")
        print(f"Hits@1/3/10: {val_hits[0]:.4f}/{val_hits[1]:.4f}/{val_hits[2]:.4f}")
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            no_improve = 0
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mrr': best_mrr,
                'history': history
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    # Đánh giá cuối cùng trên test set
    print("\n==== Final Evaluation on Test Set ====")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    test_mrr, test_hits, test_ranks = evaluate(model, test_loader, device)
    print(f"Test MRR: {test_mrr:.4f}, Hits@1/3/10: {test_hits[0]:.4f}/{test_hits[1]:.4f}/{test_hits[2]:.4f}")
    rank_counts = torch.bincount(test_ranks.int(), minlength=1).float()
    rank_dist = rank_counts / rank_counts.sum()
    print("Rank Distribution:", rank_dist.tolist()[:20])
    return {
        'best_val_mrr': best_mrr,
        'test_mrr': test_mrr,
        'test_hits': test_hits,
        'history': history
    }
