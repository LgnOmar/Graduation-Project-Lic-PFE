"""
Script for training the GCN model.
"""
import torch
from torch_geometric.transforms import RandomLinkSplit
import pandas as pd
from typing import Dict, Tuple
import pickle
import os
import logging
import sys
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import numpy as np

from graph_construction import build_graph
from gcn_model import HeteroGCNLinkPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
    y_pred_binary = (y_pred > 0.5).float()
    return {
        'auc': roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy()),
        'precision': precision_score(y_true.detach().cpu().numpy(), y_pred_binary.detach().cpu().numpy(), zero_division=0),
        'recall': recall_score(y_true.detach().cpu().numpy(), y_pred_binary.detach().cpu().numpy(), zero_division=0)
    }

def load_data(data_dir: str = 'data') -> Tuple[pd.DataFrame, Dict]:
    try:
        interactions_df = pd.read_csv(f'{data_dir}/processed_interactions.csv')
        with open(f'{data_dir}/job_embeddings.pkl', 'rb') as f:
            job_embeddings = pickle.load(f)
        return interactions_df, job_embeddings
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def evaluate_model(model: torch.nn.Module, data, criterion) -> Tuple[float, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        pred = model(
            data.x_dict,
            data.edge_index_dict,
            data['user', 'interacts_with', 'job'].edge_label_index
        )
        target = data['user', 'interacts_with', 'job'].edge_label
        loss = criterion(pred, target)
        metrics = calculate_metrics(target, pred)
    return loss.item(), metrics

def train_model(
    data_dir: str = 'data',
    hidden_channels: int = 64,
    num_layers: int = 2,
    num_epochs: int = 100,
    lr: float = 0.01,
    output_dir: str = 'models',
    patience: int = 10
) -> None:
    try:
        os.makedirs(output_dir, exist_ok=True)

        interactions_df, job_embeddings = load_data(data_dir)
        graph = build_graph(interactions_df, job_embeddings)

        transform = RandomLinkSplit(
            num_val=0.1,
            num_test=0.1,
            neg_sampling_ratio=1.0,
            edge_types=[('user', 'interacts_with', 'job')],
            rev_edge_types=[('job', 'rev_interacts_with', 'user')]
        )
        train_data, val_data, test_data = transform(graph)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        model = HeteroGCNLinkPredictor(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            data=train_data
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            pred = model(
                train_data.x_dict,
                train_data.edge_index_dict,
                train_data['user', 'interacts_with', 'job'].edge_label_index
            )
            target = train_data['user', 'interacts_with', 'job'].edge_label
            train_loss = criterion(pred, target)
            train_metrics = calculate_metrics(target, pred)
            train_loss.backward()
            optimizer.step()

            val_loss, val_metrics = evaluate_model(model, val_data, criterion)

            logger.info(f"Epoch {epoch+1}/{num_epochs}:")
            logger.info(f"Train Loss: {train_loss:.4f}, AUC: {train_metrics['auc']:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, AUC: {val_metrics['auc']:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(output_dir, 'best_model.pt'))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Load best model for final evaluation
        import torch.serialization
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])

        best_model_path = os.path.join(output_dir, 'best_model.pt')
        checkpoint = torch.load(best_model_path, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_loss, test_metrics = evaluate_model(model, test_data, criterion)
        logger.info("\nFinal Test Results:")
        logger.info(f"Loss: {test_loss:.4f}")
        logger.info(f"AUC: {test_metrics['auc']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")

        results = {
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss
        }
        with open(os.path.join(output_dir, 'test_results.pkl'), 'wb') as f:
            pickle.dump(results, f)

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)
