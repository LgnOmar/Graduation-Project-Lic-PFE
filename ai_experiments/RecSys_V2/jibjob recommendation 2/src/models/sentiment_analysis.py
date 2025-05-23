"""
BERT-based sentiment analysis model for JibJob recommendation system.
This module handles sentiment extraction from user comments and reviews.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Union, Optional, Tuple
import logging
import os
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__) 

class SentimentAnalysis:
    """
    A class for analyzing sentiment in user comments and reviews.
    
    This class provides methods to extract sentiment scores from text data 
    using pre-trained BERT models fine-tuned for sentiment analysis.
    """
    
    def __init__(
        self, 
        model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        num_labels: int = 5  # 1-5 star rating for the default model
    ):
        """
        Initialize the sentiment analysis model.
        
        Args:
            model_name: Name of the pre-trained model to use.
                Default is a multilingual sentiment model (outputs 1-5 stars).
            device: Device to run the model on ('cpu' or 'cuda'). If None, automatically detect.
            cache_dir: Directory to cache the downloaded models.
            num_labels: Number of sentiment classes/labels in the model.
        """
        # Determine device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading sentiment analysis model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            cache_dir=cache_dir
        ).to(self.device)
        
        self.num_labels = num_labels
        
        # Set model to evaluation mode
        self.model.eval()
        
    def analyze_sentiment(
        self, 
        texts: Union[str, List[str]], 
        max_length: int = 128,
        return_scores: bool = False
    ) -> Union[Union[float, List[float]], Tuple[Union[float, List[float]], np.ndarray]]:
        """
        Analyze sentiment in the provided texts.
        
        Args:
            texts: A single text string or a list of text strings.
            max_length: Maximum sequence length for tokenization.
            return_scores: If True, return the raw scores for each sentiment class.
            
        Returns:
            If return_scores is False:
                For a single text: A float representing normalized sentiment (0 to 1)
                For multiple texts: A list of floats representing normalized sentiments
            If return_scores is True:
                A tuple containing the above plus a numpy array of raw scores
        """
        # Convert single text to list and track if input was a single string
        single_text = False
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate sentiment scores
        with torch.no_grad():
            output = self.model(**encoded_input)
        
        # Get raw scores and convert to probabilities
        scores = output.logits.cpu().numpy()
        probabilities = torch.nn.functional.softmax(output.logits, dim=1).cpu().numpy()
        
        # Calculate weighted average for sentiment score (normalized to 0-1)
        # For the default 5-star model, we weight by the star rating (1-5) then normalize
        weights = np.array(range(1, self.num_labels + 1))
        weighted_sum = np.sum(probabilities * weights.reshape(1, -1), axis=1)
        normalized_sentiment = (weighted_sum - 1) / (self.num_labels - 1)  # Normalize to 0-1
        
        # Return based on input format and return_scores flag
        if single_text:
            if return_scores:
                return float(normalized_sentiment[0]), scores[0]
            else:
                return float(normalized_sentiment[0])
        else:
            if return_scores:
                return normalized_sentiment.tolist(), scores
            else:
                return normalized_sentiment.tolist()
    
    def batch_analyze_sentiment(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False,
        **kwargs
    ) -> List[float]:
        """
        Analyze sentiment for a large list of texts using batching.
        
        Args:
            texts: List of text strings.
            batch_size: Number of texts to process in each batch.
            show_progress: Whether to show a progress bar.
            **kwargs: Additional arguments to pass to analyze_sentiment.
            
        Returns:
            List[float]: List of normalized sentiment scores.
        """
        all_sentiments = []
        
        # Create iterator with or without progress bar
        iterator = tqdm(range(0, len(texts), batch_size)) if show_progress else range(0, len(texts), batch_size)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            batch_sentiments = self.analyze_sentiment(batch_texts, **kwargs)
            all_sentiments.extend(batch_sentiments)
            
        return all_sentiments
    
    def fine_tune(
        self,
        texts: List[str],
        labels: List[int],
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[int]] = None,
        learning_rate: float = 5e-5,
        batch_size: int = 16,
        epochs: int = 3,
        max_length: int = 128,
        save_path: Optional[str] = None,
        class_weights: Optional[List[float]] = None
    ):
        """
        Fine-tune the sentiment analysis model on custom data.
        
        Args:
            texts: Training text data.
            labels: Training labels (should be integer classes, e.g., 0-4 for 5-class sentiment).
            val_texts: Validation text data.
            val_labels: Validation labels.
            learning_rate: Learning rate for optimizer.
            batch_size: Training batch size.
            epochs: Number of training epochs.
            max_length: Maximum sequence length.
            save_path: Path to save the fine-tuned model.
            class_weights: Optional weights for each class to handle class imbalance.
        
        Returns:
            Dict: Training history with loss and accuracy metrics.
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.optim import AdamW
        
        # Set model to training mode
        self.model.train()
        
        # Prepare training data
        train_encodings = self.tokenizer(
            texts, 
            truncation=True, 
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        train_dataset = TensorDataset(
            train_encodings['input_ids'].to(self.device),
            train_encodings['attention_mask'].to(self.device),
            torch.tensor(labels).long().to(self.device)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        # Prepare validation data if provided
        val_loader = None
        if val_texts and val_labels:
            val_encodings = self.tokenizer(
                val_texts,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            val_dataset = TensorDataset(
                val_encodings['input_ids'].to(self.device),
                val_encodings['attention_mask'].to(self.device),
                torch.tensor(val_labels).long().to(self.device)
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size
            )
        
        # Setup optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Setup loss function with class weights if provided
        if class_weights:
            weight = torch.tensor(class_weights).float().to(self.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        
        # Training loop
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                input_ids, attention_mask, labels_batch = batch
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                loss = loss_fn(outputs.logits, labels_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track metrics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.logits, dim=1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
            
            # Calculate epoch metrics
            epoch_loss = train_loss / len(train_loader)
            epoch_acc = correct / total
            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                        f"Train Loss: {epoch_loss:.4f}, "
                        f"Train Acc: {epoch_acc:.4f}")
            
            # Validation if data provided
            if val_loader:
                self.model.eval()
                val_loss = 0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                        input_ids, attention_mask, labels_batch = batch
                        
                        # Forward pass
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        
                        # Calculate loss
                        loss = loss_fn(outputs.logits, labels_batch)
                        
                        # Track metrics
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.logits, dim=1)
                        total += labels_batch.size(0)
                        correct += (predicted == labels_batch).sum().item()
                
                # Calculate epoch metrics
                epoch_val_loss = val_loss / len(val_loader)
                epoch_val_acc = correct / total
                history['val_loss'].append(epoch_val_loss)
                history['val_acc'].append(epoch_val_acc)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                            f"Val Loss: {epoch_val_loss:.4f}, "
                            f"Val Acc: {epoch_val_acc:.4f}")
        
        # Save fine-tuned model if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            logger.info(f"Model saved to {save_path}")
        
        # Set model back to evaluation mode
        self.model.eval()
        
        return history
        
    def save_model(self, path: str):
        """Save tokenizer and model to disk"""
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)

    @staticmethod
    def load_model(path: str, device: Optional[str] = None):
        """Load a saved sentiment analysis model"""
        # Safety check for CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
            
        device = device if device else ('cpu')
        
        # Determine number of labels based on the model configuration
        import json
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
        num_labels = config.get('num_labels', 5)
        
        # Create instance and load model
        model = SentimentAnalysis(device=device, num_labels=num_labels)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
        model.model.eval()
        
        return model
