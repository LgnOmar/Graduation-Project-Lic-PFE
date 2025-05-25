"""
BERT-based text embedding model for JibJob recommendation system.
This module handles the extraction of semantic features from job descriptions.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Union, Optional
import logging

# Setup logging
logger = logging.getLogger(__name__) 

class BERTEmbeddings:
    """
    A class to generate embeddings from text using BERT models.
    
    This class provides methods to extract semantic embeddings from job descriptions
    or any other text data using pre-trained BERT models.
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-multilingual-cased",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the BERT embeddings model.
        
        Args:
            model_name: Name of the pre-trained BERT model to use. Default is multilingual BERT.
            device: Device to run the model on ('cpu' or 'cuda'). If None, automatically detect.
            cache_dir: Directory to cache the downloaded models.
        """        # Determine device safely
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
        
        self.device = device if device else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading BERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
    def get_embeddings(
        self, 
        texts: Union[str, List[str]], 
        pooling_strategy: str = 'mean',
        max_length: int = 128
    ) -> np.ndarray:
        """
        Generate embeddings for the provided texts.
        
        Args:
            texts: A single text string or a list of text strings.
            pooling_strategy: Strategy to pool token embeddings ('mean', 'cls', or 'max').
            max_length: Maximum sequence length for tokenization.
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize texts
        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        # Pool token embeddings based on strategy
        if pooling_strategy == 'cls':
            # Use [CLS] token embedding
            embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        elif pooling_strategy == 'mean':
            # Mean pooling - take average of all tokens
            attention_mask = encoded_input['attention_mask']
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = embeddings.cpu().numpy()
        elif pooling_strategy == 'max':
            # Max pooling - take maximum of all tokens
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            embeddings = torch.max(token_embeddings, 1)[0].cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")
        
        return embeddings
    
    def batch_get_embeddings(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        **kwargs
    ) -> np.ndarray:
        """
        Generate embeddings for a large list of texts using batching.
        
        Args:
            texts: List of text strings.
            batch_size: Number of texts to process in each batch.
            **kwargs: Additional arguments to pass to get_embeddings.
            
        Returns:
            numpy.ndarray: Array of embeddings with shape (n_texts, embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.get_embeddings(batch_texts, **kwargs)
            all_embeddings.append(batch_embeddings)
            
        return np.vstack(all_embeddings)
    
    def calculate_text_similarity(
        self, 
        text1: str, 
        text2: str,
        pooling_strategy: str = 'mean',
        similarity_metric: str = 'cosine'
    ) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text.
            text2: Second text.
            pooling_strategy: Strategy to pool token embeddings.
            similarity_metric: Metric to use for similarity calculation ('cosine' or 'dot').
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        # Get embeddings for both texts
        emb1 = self.get_embeddings(text1, pooling_strategy=pooling_strategy)
        emb2 = self.get_embeddings(text2, pooling_strategy=pooling_strategy)
        
        # Calculate similarity
        if similarity_metric == 'cosine':
            # Cosine similarity
            sim = np.dot(emb1, emb2.T) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif similarity_metric == 'dot':
            # Dot product
            sim = np.dot(emb1, emb2.T)
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        return float(sim)
    
    def save_cache(self, path: str):
        """Save tokenizer and model cache to disk"""
        self.tokenizer.save_pretrained(path)
        self.model.save_pretrained(path)
        
    @staticmethod
    def load_from_cache(path: str, device: Optional[str] = None):
        """Load tokenizer and model from cache"""
        # Safety check for CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'
            
        model = BERTEmbeddings(device=device)
        model.tokenizer = AutoTokenizer.from_pretrained(path)
        model.model = AutoModel.from_pretrained(path).to(model.device)
        model.model.eval()
        return model
