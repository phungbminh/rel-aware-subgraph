import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics
import os
from tqdm import tqdm
class Trainer:
    def __init__(self, model, train_dataset, valid_dataset=None,
                 batch_size=32, lr=1e-3, margin=1.0,
                 optimizer_name='Adam', device='cuda',
                 collate_fn=None, exp_dir='.', save_every=5):

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.batch_size = batch_size
        self.device = device
        self.collate_fn = collate_fn
        self.save_every = save_every
        self.exp_dir = exp_dir

        self.optimizer = getattr(torch.optim, optimizer_name)(
            self.model.parameters(), lr=lr)
        self.criterion = nn.MarginRankingLoss(margin, reduction='mean')
        os.makedirs(self.exp_dir, exist_ok=True)
        self.best_auc = 0

    @staticmethod
    def move_batch_to_device(batch, device):
        pos_graph, pos_label, neg_graph, neg_label = batch
        pos_graph = pos_graph.to(device)
        neg_graph = neg_graph.to(device)
        pos_label = pos_label.to(device)
        neg_label = neg_label.to(device)
        return pos_graph, pos_label, neg_graph, neg_label

    def train_epoch(self):
        self.model.train()
        dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                                collate_fn=self.collate_fn)
        total_loss = 0
        all_scores = []
        all_labels = []
        pbar = tqdm(dataloader, desc="Training", leave=False)
        for batch in pbar:
            pos_graph, pos_label, neg_graph, neg_label = self.move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad()
            score_pos = self.model(pos_graph)
            score_neg = self.model(neg_graph)
            # Score shape: [batch_size]
            target = torch.ones_like(score_pos)
            loss = self.criterion(score_pos, score_neg, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * score_pos.size(0)
            # Logging for AUC
            all_scores.extend(score_pos.detach().cpu().tolist())
            all_scores.extend(score_neg.detach().cpu().tolist())
            all_labels.extend([1]*score_pos.size(0) + [0]*score_neg.size(0))

        # Calculate metrics
        auc = metrics.roc_auc_score(all_labels, all_scores)
        ap = metrics.average_precision_score(all_labels, all_scores)
        return total_loss / len(self.train_dataset), auc, ap

    def eval_epoch(self):
        self.model.eval()
        dataloader = DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False,
                                collate_fn=self.collate_fn)
        all_scores = []
        all_labels = []
        with torch.no_grad():
            for batch in dataloader:
                pos_graph, pos_label, neg_graph, neg_label = self.move_batch_to_device(batch, self.device)
                score_pos = self.model(pos_graph)
                score_neg = self.model(neg_graph)
                all_scores.extend(score_pos.cpu().tolist())
                all_scores.extend(score_neg.cpu().tolist())
                all_labels.extend([1]*score_pos.size(0) + [0]*score_neg.size(0))
        auc = metrics.roc_auc_score(all_labels, all_scores)
        ap = metrics.average_precision_score(all_labels, all_scores)
        return auc, ap

    def save_model(self, filename):
        torch.save(self.model.state_dict(), os.path.join(self.exp_dir, filename))

    def train(self, num_epochs=20):
        for epoch in range(1, num_epochs+1):
            train_loss, train_auc, train_ap = self.train_epoch()
            print(f"Epoch {epoch}/{num_epochs}: Train Loss={train_loss:.4f}, AUC={train_auc:.4f}, AP={train_ap:.4f}")

            # Validation
            if self.valid_dataset is not None:
                val_auc, val_ap = self.eval_epoch()
                print(f"           [Validation] AUC={val_auc:.4f}, AP={val_ap:.4f}")
                if val_auc > self.best_auc:
                    self.save_model("best_model.pt")
                    self.best_auc = val_auc

            if epoch % self.save_every == 0:
                self.save_model(f"model_epoch_{epoch}.pt")

