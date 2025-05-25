"""
Trainer module for training GCN recommender models.
"""

import os
import logging
import time
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.optim import Adam, AdamW
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch_geometric.data

logger = logging.getLogger(__name__)

class Trainer:
    """
    Class responsible for training GCN recommender models.
    """
    
    def __init__(self, config: Dict[str, Any], model: torch.nn.Module):
        """
        Initialize the Trainer.
        
        Args:
            config: Configuration dictionary.
            model: Model to train.
        """
        self.config = config
        self.training_config = config.get('training', {})
        
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training hyperparameters
        self.lr = self.config.get('model', {}).get('lr', 0.001)
        self.weight_decay = self.config.get('model', {}).get('weight_decay', 0.0001)
        self.num_epochs = self.training_config.get('num_epochs', 20)
        self.batch_size = self.training_config.get('batch_size', 64)
        self.patience = self.training_config.get('patience', 5)
        self.val_split = self.training_config.get('val_split', 0.1)
        self.test_split = self.training_config.get('test_split', 0.1)
        self.seed = self.training_config.get('seed', 42)
        self.neg_samples = self.training_config.get('neg_samples', 5)
        
        # Set random seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        # Initialize optimizer
        self.optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
          def train_homogeneous_model(self, 
                               data: torch_geometric.data.Data,
                               professional_job_pairs: torch.Tensor, 
                               labels: torch.Tensor) -> Dict[str, Any]:
        """
        Train a homogeneous GCN model.
        
        Args:
            data: Graph data.
            professional_job_pairs: Tensor of shape [2, N] containing (professional_idx, job_idx) pairs.
            labels: Tensor of shape [N] containing binary labels (1 for positive pairs, 0 for negative pairs).
            
        Returns:
            Dict[str, Any]: Training statistics.
        """
        logger.info(f"Training homogeneous GCN model on {self.device}")
        logger.info(f"Training set: {professional_job_pairs.shape[1]} pairs, "
                   f"{torch.sum(labels).item()} positive, {(labels == 0).sum().item()} negative")
                   
        # Split data into train/val/test
        train_pairs, test_pairs, train_labels, test_labels = train_test_split(
            professional_job_pairs.t().cpu().numpy(),
            labels.cpu().numpy(),
            test_size=self.test_split,
            random_state=self.seed,
            stratify=labels.cpu().numpy()
        )
        
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            train_pairs,
            train_labels,
            test_size=self.val_split / (1 - self.test_split),
            random_state=self.seed,
            stratify=train_labels
        )
        
        # Convert back to torch tensors
        train_pairs = torch.tensor(train_pairs, dtype=torch.long).to(self.device)
        val_pairs = torch.tensor(val_pairs, dtype=torch.long).to(self.device)
        test_pairs = torch.tensor(test_pairs, dtype=torch.long).to(self.device)
        
        train_labels = torch.tensor(train_labels, dtype=torch.float).to(self.device)
        val_labels = torch.tensor(val_labels, dtype=torch.float).to(self.device)
        test_labels = torch.tensor(test_labels, dtype=torch.float).to(self.device)
        
        logger.info(f"Train: {train_pairs.shape[0]} pairs, Val: {val_pairs.shape[0]} pairs, "
                   f"Test: {test_pairs.shape[0]} pairs")
                   
        # Training statistics
        stats = {
            'train_losses': [],
            'val_losses': [],
            'train_aucs': [],
            'val_aucs': [],
            'best_val_loss': float('inf'),
            'best_epoch': -1,
            'training_time': 0
        }
        
        # Move data to device
        data = data.to(self.device)
        
        # Early stopping variables
        patience_counter = 0
        best_val_loss = float('inf')
        
        # Start timer
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass to get node embeddings
            embeddings = self.model(data.x, data.edge_index)
            
            # Get professional and job embeddings for training pairs
            professional_embeddings = embeddings[train_pairs[:, 0]]
            job_embeddings = embeddings[train_pairs[:, 1]]
            
            # Predict links
            preds = self.model.predict_link(professional_embeddings, job_embeddings).squeeze()
            
            # Compute loss
            loss = F.binary_cross_entropy(preds, train_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Validation
            val_loss, val_auc = self._evaluate_homogeneous_model(
                val_pairs, val_labels, data, embeddings)
                
            # Calculate training AUC
            with torch.no_grad():
                train_auc = self._calculate_auc(preds.cpu().numpy(), train_labels.cpu().numpy())
            
            # Track statistics
            stats['train_losses'].append(loss.item())
            stats['val_losses'].append(val_loss)
            stats['train_aucs'].append(train_auc)
            stats['val_aucs'].append(val_auc)
            
            logger.info(f"Epoch {epoch}/{self.num_epochs}: "
                       f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, "
                       f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
                       
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stats['best_val_loss'] = val_loss
                stats['best_epoch'] = epoch
                patience_counter = 0
                
                # Save best model
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(self.model.state_dict(), 'models/best_model.pt')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
                
        # End timer
        stats['training_time'] = time.time() - start_time
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load('models/best_model.pt', map_location=self.device))
        self.model.eval()
        
        # Final evaluation on test set
        with torch.no_grad():
            # Get node embeddings
            embeddings = self.model(data.x, data.edge_index)
            
            # Test loss and AUC
            test_loss, test_auc = self._evaluate_homogeneous_model(
                test_pairs, test_labels, data, embeddings)
                
            stats['test_loss'] = test_loss
            stats['test_auc'] = test_auc
            
            logger.info(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
            
        return stats
          def train_heterogeneous_model(self, 
                                data: torch_geometric.data.HeteroData,
                                professional_job_pairs: torch.Tensor, 
                                labels: torch.Tensor) -> Dict[str, Any]:
        """
        Train a heterogeneous GCN model.
        
        Args:
            data: Heterogeneous graph data.
            professional_job_pairs: Tensor of shape [2, N] containing (professional_idx, job_idx) pairs.
            labels: Tensor of shape [N] containing binary labels (1 for positive pairs, 0 for negative pairs).
            
        Returns:
            Dict[str, Any]: Training statistics.
        """
        logger.info(f"Training heterogeneous GCN model on {self.device}")
        logger.info(f"Training set: {professional_job_pairs.shape[1]} pairs, "
                   f"{torch.sum(labels).item()} positive, {(labels == 0).sum().item()} negative")
                   
        # Split data into train/val/test
        train_pairs, test_pairs, train_labels, test_labels = train_test_split(
            professional_job_pairs.t().cpu().numpy(),
            labels.cpu().numpy(),
            test_size=self.test_split,
            random_state=self.seed,
            stratify=labels.cpu().numpy()
        )
        
        train_pairs, val_pairs, train_labels, val_labels = train_test_split(
            train_pairs,
            train_labels,
            test_size=self.val_split / (1 - self.test_split),
            random_state=self.seed,
            stratify=train_labels
        )
        
        # Convert back to torch tensors
        train_pairs = torch.tensor(train_pairs, dtype=torch.long).to(self.device)
        val_pairs = torch.tensor(val_pairs, dtype=torch.long).to(self.device)
        test_pairs = torch.tensor(test_pairs, dtype=torch.long).to(self.device)
        
        train_labels = torch.tensor(train_labels, dtype=torch.float).to(self.device)
        val_labels = torch.tensor(val_labels, dtype=torch.float).to(self.device)
        test_labels = torch.tensor(test_labels, dtype=torch.float).to(self.device)
        
        logger.info(f"Train: {train_pairs.shape[0]} pairs, Val: {val_pairs.shape[0]} pairs, "
                   f"Test: {test_pairs.shape[0]} pairs")
                   
        # Training statistics
        stats = {
            'train_losses': [],
            'val_losses': [],
            'train_aucs': [],
            'val_aucs': [],
            'best_val_loss': float('inf'),
            'best_epoch': -1,
            'training_time': 0
        }
        
        # Move data to device
        data = data.to(self.device)
        
        # Early stopping variables
        patience_counter = 0
        best_val_loss = float('inf')
        
        # Start timer
        start_time = time.time()
        
        # Training loop
        for epoch in range(1, self.num_epochs + 1):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass to get node embeddings
            embeddings_dict = self.model(data.x_dict, data.edge_index_dict)
            
            # Get professional and job embeddings for training pairs
            professional_embeddings = embeddings_dict['professional'][train_pairs[:, 0]]
            job_embeddings = embeddings_dict['job'][train_pairs[:, 1]]
            
            # Predict links
            preds = self.model.predict_link(professional_embeddings, job_embeddings).squeeze()
            
            # Compute loss
            loss = F.binary_cross_entropy(preds, train_labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Validation
            val_loss, val_auc = self._evaluate_heterogeneous_model(
                val_pairs, val_labels, data, embeddings_dict)
                
            # Calculate training AUC
            with torch.no_grad():
                train_auc = self._calculate_auc(preds.cpu().numpy(), train_labels.cpu().numpy())
            
            # Track statistics
            stats['train_losses'].append(loss.item())
            stats['val_losses'].append(val_loss)
            stats['train_aucs'].append(train_auc)
            stats['val_aucs'].append(val_auc)
            
            logger.info(f"Epoch {epoch}/{self.num_epochs}: "
                       f"Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, "
                       f"Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
                       
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                stats['best_val_loss'] = val_loss
                stats['best_epoch'] = epoch
                patience_counter = 0
                
                # Save best model
                if not os.path.exists('models'):
                    os.makedirs('models')
                torch.save(self.model.state_dict(), 'models/best_hetero_model.pt')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
                
        # End timer
        stats['training_time'] = time.time() - start_time
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load('models/best_hetero_model.pt', map_location=self.device))
        self.model.eval()
        
        # Final evaluation on test set
        with torch.no_grad():
            # Get node embeddings
            embeddings_dict = self.model(data.x_dict, data.edge_index_dict)
            
            # Test loss and AUC
            test_loss, test_auc = self._evaluate_heterogeneous_model(
                test_pairs, test_labels, data, embeddings_dict)
                
            stats['test_loss'] = test_loss
            stats['test_auc'] = test_auc
            
            logger.info(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
            
        return stats
        
    def _evaluate_homogeneous_model(self, 
                                  pairs: torch.Tensor, 
                                  labels: torch.Tensor, 
                                  data: torch.geometric.data.Data,
                                  embeddings: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """
        Evaluate a homogeneous GCN model.
        
        Args:
            pairs: Tensor of shape [N, 2] containing (professional_idx, job_idx) pairs.
            labels: Tensor of shape [N] containing binary labels.
            data: Graph data.
            embeddings: Pre-computed node embeddings (optional).
            
        Returns:
            Tuple[float, float]: (Loss, AUC)
        """
        self.model.eval()
        with torch.no_grad():
            # Get node embeddings if not provided
            if embeddings is None:
                embeddings = self.model(data.x, data.edge_index)
                
            # Get professional and job embeddings for evaluation pairs
            professional_embeddings = embeddings[pairs[:, 0]]
            job_embeddings = embeddings[pairs[:, 1]]
            
            # Predict links
            preds = self.model.predict_link(professional_embeddings, job_embeddings).squeeze()
            
            # Compute loss
            loss = F.binary_cross_entropy(preds, labels).item()
            
            # Compute AUC
            auc = self._calculate_auc(preds.cpu().numpy(), labels.cpu().numpy())
            
            return loss, auc
            
    def _evaluate_heterogeneous_model(self, 
                                    pairs: torch.Tensor, 
                                    labels: torch.Tensor, 
                                    data: torch.geometric.data.HeteroData,
                                    embeddings_dict: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[float, float]:
        """
        Evaluate a heterogeneous GCN model.
        
        Args:
            pairs: Tensor of shape [N, 2] containing (professional_idx, job_idx) pairs.
            labels: Tensor of shape [N] containing binary labels.
            data: Heterogeneous graph data.
            embeddings_dict: Pre-computed node embeddings by type (optional).
            
        Returns:
            Tuple[float, float]: (Loss, AUC)
        """
        self.model.eval()
        with torch.no_grad():
            # Get node embeddings if not provided
            if embeddings_dict is None:
                embeddings_dict = self.model(data.x_dict, data.edge_index_dict)
                
            # Get professional and job embeddings for evaluation pairs
            professional_embeddings = embeddings_dict['professional'][pairs[:, 0]]
            job_embeddings = embeddings_dict['job'][pairs[:, 1]]
            
            # Predict links
            preds = self.model.predict_link(professional_embeddings, job_embeddings).squeeze()
            
            # Compute loss
            loss = F.binary_cross_entropy(preds, labels).item()
            
            # Compute AUC
            auc = self._calculate_auc(preds.cpu().numpy(), labels.cpu().numpy())
            
            return loss, auc
            
    def _calculate_auc(self, preds: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate AUC score.
        
        Args:
            preds: Predicted probabilities.
            labels: Ground truth labels.
            
        Returns:
            float: AUC score.
        """
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(labels, preds)
        except:
            # Fallback if sklearn is not available
            # Simple AUC calculation
            n_pos = int(np.sum(labels))
            n_neg = len(labels) - n_pos
            
            if n_pos == 0 or n_neg == 0:
                return 0.5  # Undefined AUC
                
            pos_preds = preds[labels == 1]
            neg_preds = preds[labels == 0]
            
            # Count number of positive examples ranked above negative examples
            count = 0
            for pos_pred in pos_preds:
                count += np.sum(pos_pred > neg_preds)
                
            return count / (n_pos * n_neg)
            
    def generate_negative_samples(self, 
                                positive_pairs: torch.Tensor, 
                                num_professionals: int, 
                                num_jobs: int, 
                                n_neg_per_pos: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate negative samples for training.
        
        Args:
            positive_pairs: Tensor of shape [2, N] containing (professional_idx, job_idx) positive pairs.
            num_professionals: Total number of professionals.
            num_jobs: Total number of jobs.
            n_neg_per_pos: Number of negative samples per positive sample.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (Pairs, Labels)
            - Pairs: Tensor of shape [2, N*(n_neg_per_pos+1)] containing both positive and negative pairs.
            - Labels: Tensor of shape [N*(n_neg_per_pos+1)] containing binary labels.
        """
        # Convert positive pairs to set for O(1) lookup
        positive_set = set((p.item(), j.item()) for p, j in zip(positive_pairs[0], positive_pairs[1]))
        
        # Generate negative pairs
        negative_pairs = []
        for _ in range(n_neg_per_pos * positive_pairs.shape[1]):
            # Keep sampling until we find a valid negative pair
            while True:
                # Random sampling strategy
                p_idx = np.random.randint(0, num_professionals)
                j_idx = np.random.randint(0, num_jobs)
                
                # Check if this is not a positive pair
                if (p_idx, j_idx) not in positive_set:
                    negative_pairs.append([p_idx, j_idx])
                    break
                    
        # Convert to tensor
        negative_pairs = torch.tensor(negative_pairs, dtype=torch.long).t()
        
        # Combine positive and negative pairs
        all_pairs = torch.cat([positive_pairs, negative_pairs], dim=1)
        
        # Create labels (1 for positive, 0 for negative)
        labels = torch.zeros(all_pairs.shape[1], dtype=torch.float)
        labels[:positive_pairs.shape[1]] = 1.0
        
        # Shuffle
        indices = torch.randperm(all_pairs.shape[1])
        all_pairs = all_pairs[:, indices]
        labels = labels[indices]
        
        return all_pairs, labels
