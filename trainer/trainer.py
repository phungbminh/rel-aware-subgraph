import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from utils.pyg_utils import collate_pyg, move_batch_to_device_pyg

def train_one_epoch(model, loader, optimizer, device, margin=1.0):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in tqdm(loader, desc='Train'):
        batch = move_batch_to_device_pyg(batch, device)
        batch_pos, g_label, r_label, batch_negs, batch_neg_g_labels, batch_neg_r_labels = batch

        batch_size = batch_pos.num_graphs
        loss = 0

        # Lặp từng sample trong batch (hoặc bạn có thể tối ưu song song)
        for i in range(batch_size):
            # Dương
            pos_data = batch_pos.get_example(i)
            pos_score = model(pos_data, int(r_label[i]))
            # Âm (có thể nhiều negative cho mỗi dương)
            neg_scores = []
            for neg_data in batch_negs[i]:
                neg_scores.append(model(neg_data, int(r_label[i])))  # hoặc neg_r_label[i][j] nếu cần

            neg_scores = torch.stack(neg_scores)
            pos_score = pos_score.expand_as(neg_scores)
            ones = torch.ones_like(neg_scores)
            loss += torch.nn.functional.margin_ranking_loss(pos_score, neg_scores, ones, margin=margin)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_batches += 1

    return total_loss / n_batches

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_mrr = 0
    total_hits = [0, 0, 0]  # Hits@1, @3, @10
    count = 0

    for batch in tqdm(loader, desc='Eval'):
        batch = move_batch_to_device_pyg(batch, device)
        batch_pos, g_label, r_label, batch_negs, batch_neg_g_labels, batch_neg_r_labels = batch
        batch_size = batch_pos.num_graphs

        for i in range(batch_size):
            pos_data = batch_pos.get_example(i)
            pos_score = model(pos_data, int(r_label[i])).item()
            neg_scores = [model(neg_data, int(r_label[i])).item() for neg_data in batch_negs[i]]
            scores = [pos_score] + neg_scores
            rank = 1 + sum([s > pos_score for s in neg_scores])
            total_mrr += 1.0 / rank
            for idx, K in enumerate([1,3,10]):
                total_hits[idx] += int(rank <= K)
            count += 1

    mrr = total_mrr / count
    hits = [h / count for h in total_hits]
    return mrr, hits

def run_training(
    model, train_dataset, valid_dataset, test_dataset,
    epochs, batch_size, device, lr=1e-3, margin=1.0, patience=6
):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_pyg
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_pyg
    )

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_mrr = 0
    patience_cnt = 0

    for epoch in range(1, epochs+1):
        print(f"\n=== Epoch {epoch} ===")
        loss = train_one_epoch(model, train_loader, optimizer, device, margin)
        print(f"Train loss: {loss:.4f}")

        mrr, hits = evaluate(model, valid_loader, device)
        print(f"Valid MRR: {mrr:.4f}, Hits@1/3/10: {hits}")

        if mrr > best_mrr:
            best_mrr = mrr
            patience_cnt = 0
            # Bạn có thể lưu model tốt nhất tại đây
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early stopping!")
                break

    print("\n==== Final evaluation on Test set ====")
    model.load_state_dict(torch.load("best_model.pt"))
    mrr, hits = evaluate(model, test_loader, device)
    print(f"Test MRR: {mrr:.4f}, Hits@1/3/10: {hits}")

    return mrr, hits

