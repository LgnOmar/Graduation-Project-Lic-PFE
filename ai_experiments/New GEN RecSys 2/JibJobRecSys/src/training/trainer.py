"""
trainer.py

Purpose:
    Train the HetGCNRecommender model for link prediction on (Professional, interacted_with, Job) edges.

Key Functions:
    - train_model(model, data, train_edges, val_edges, optimizer, loss_fn, sampler, epochs)
        - Handles batching, negative sampling, optimizer, and early stopping.
        - Uses BCE loss.

Inputs:
    - model: HetGCNRecommender instance.
    - data: HeteroData graph.
    - train_edges: Training edge_label_index.
    - val_edges: Validation edge_label_index.
    - optimizer: Optimizer instance.
    - loss_fn: Loss function (BCE).
    - sampler: Negative sampler function.
    - epochs: Number of epochs.

Outputs:
    - Trained model, training/validation loss history.

High-Level Logic:
    1. For each epoch, sample negatives, run forward/backward, update optimizer.
    2. Evaluate on validation set, implement early stopping.
"""

import torch
import numpy as np
from tqdm import tqdm
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, mae, rmse

def train_model(
    model,
    data,
    train_edges,
    val_edges,
    optimizer,
    loss_fn,
    sampler,
    epochs=10,
    device='cpu',
    num_neg_per_pos=1,
    all_positive_interactions_map=None
):
    model.to(device)
    data = data.to(device)
    history = {
        'train_loss': [], 'val_loss': [],
        'val_ndcg@10': [], 'val_precision@10': [], 'val_recall@10': [],
        'val_mae': [], 'val_rmse': []
    }
    k_epoch_eval = 100  # K value for epoch-wise metrics (was 10)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        # Sample negatives for training
        pos_src, pos_dst = train_edges
        neg_src, neg_dst = sampler(
            (pos_src, pos_dst),
            data['Job'].num_nodes,
            num_neg_per_pos,
            all_positive_interactions_map
        )
        edge_label_index = (
            torch.cat([pos_src, neg_src]),
            torch.cat([pos_dst, neg_dst])
        )
        edge_label = torch.cat([
            torch.ones(len(pos_src)),
            torch.zeros(len(neg_src))
        ]).to(device)
        preds = model(data, {('Professional', 'interacted_with', 'Job'): edge_label_index})
        loss = loss_fn(preds, edge_label)
        loss.backward()
        optimizer.step()
        history['train_loss'].append(loss.item())
        # Validation
        model.eval()
        with torch.no_grad():
            # --- Per-user validation ranking metrics ---
            pos_src, pos_dst = val_edges
            unique_profs = torch.unique(pos_src)
            k_epoch_eval = 10  # or 100, as desired
            num_neg_eval = 20  # for speed; can use config if passed
            per_prof_precisions = []
            per_prof_recalls = []
            per_prof_ndcgs = []
            for prof in unique_profs:
                # True jobs for this professional in validation
                true_jobs = pos_dst[(pos_src == prof)].cpu().numpy()
                if len(true_jobs) == 0:
                    continue
                # Sample negatives for this professional
                all_jobs = np.arange(data['Job'].num_nodes)
                already_pos = set(true_jobs)
                if all_positive_interactions_map is not None:
                    already_pos = already_pos.union(set(all_positive_interactions_map.get(int(prof), [])))
                negatives = np.setdiff1d(all_jobs, list(already_pos))
                if len(negatives) > num_neg_eval:
                    negatives = np.random.choice(negatives, num_neg_eval, replace=False)
                # Build edge_label_index for all (prof, true_job) and (prof, neg_job)
                prof_tensor = torch.full((len(true_jobs) + len(negatives),), prof, dtype=torch.long, device=device)
                jobs_tensor = torch.tensor(np.concatenate([true_jobs, negatives]), dtype=torch.long, device=device)
                edge_label_index = (prof_tensor, jobs_tensor)
                edge_label = np.array([1]*len(true_jobs) + [0]*len(negatives))
                preds = model(data, {('Professional', 'interacted_with', 'Job'): edge_label_index}).cpu().numpy()
                # Compute metrics for this professional
                per_prof_precisions.append(precision_at_k(edge_label, preds, k=k_epoch_eval))
                per_prof_recalls.append(recall_at_k(edge_label, preds, k=k_epoch_eval))
                per_prof_ndcgs.append(ndcg_at_k(edge_label, preds, k=k_epoch_eval))
            # Average per-user metrics
            avg_precision = float(np.mean(per_prof_precisions)) if per_prof_precisions else 0.0
            avg_recall = float(np.mean(per_prof_recalls)) if per_prof_recalls else 0.0
            avg_ndcg = float(np.mean(per_prof_ndcgs)) if per_prof_ndcgs else 0.0
            # Global MAE/RMSE as before
            # For global MAE/RMSE, use all validation edges (as before)
            neg_src, neg_dst = sampler(
                (pos_src, pos_dst),
                data['Job'].num_nodes,
                num_neg_per_pos,
                all_positive_interactions_map
            )
            edge_label_index = (
                torch.cat([pos_src, neg_src]),
                torch.cat([pos_dst, neg_dst])
            )
            edge_label = torch.cat([
                torch.ones(len(pos_src)),
                torch.zeros(len(neg_src))
            ]).to(device)
            preds = model(data, {('Professional', 'interacted_with', 'Job'): edge_label_index})
            val_loss = loss_fn(preds, edge_label)
            history['val_loss'].append(val_loss.item())
            y_true_val = edge_label.cpu().numpy()
            y_score_val = preds.cpu().numpy()
            current_val_mae = mae(y_true_val, y_score_val)
            current_val_rmse = rmse(y_true_val, y_score_val)
            # Store per-user averaged ranking metrics
            history['val_ndcg@10'].append(avg_ndcg)
            history['val_precision@10'].append(avg_precision)
            history['val_recall@10'].append(avg_recall)
            history['val_mae'].append(current_val_mae)
            history['val_rmse'].append(current_val_rmse)
        tqdm.write(
            f"Epoch {epoch+1}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, "
            f"val_NDCG@10={avg_ndcg:.4f}, val_P@10={avg_precision:.4f}, "
            f"val_R@10={avg_recall:.4f}, val_MAE={current_val_mae:.4f}, val_RMSE={current_val_rmse:.4f}"
        )
    return model, history
