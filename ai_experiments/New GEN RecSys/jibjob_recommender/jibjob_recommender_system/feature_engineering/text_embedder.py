"""
Text embedder module for generating embeddings from text data using BERT or similar models.
"""

import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Class responsible for generating text embeddings using transformer models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TextEmbedder with the given configuration.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.features_config = config['features']['text']
        
        self.model_name = self.features_config.get('model_name', 'distilbert-base-uncased')
        self.max_length = self.features_config.get('max_length', 128)
        self.batch_size = self.features_config.get('batch_size', 32)
        
        logger.info(f"Initializing TextEmbedder with model: {self.model_name}")
        
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self) -> None:
        """
        Load the transformer model and tokenizer.
        """
        try:
            logger.info(f"Loading tokenizer for model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise
            
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed.
            
        Returns:
            np.ndarray: Matrix of embeddings, shape (len(texts), embedding_dim)
        """
        if self.tokenizer is None or self.model is None:
            self.load_model()
            
        # Ensure texts are not None and handle empty texts
        texts = [text if isinstance(text, str) and text.strip() else "" for text in texts]
        
        # Process in batches to avoid memory issues
        all_embeddings = []
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Process in batches
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            encoded_input = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
                
            # Use CLS token representation as the sentence embedding
            # For models like BERT, the first token ([CLS]) is used as the aggregate sequence representation
            batch_embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)
            
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
        
    def generate_job_embeddings(self, jobs_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for job data.
        
        Args:
            jobs_df: DataFrame containing job data.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping job_id to embedding vector.
        """
        if 'text_content' not in jobs_df.columns:
            logger.warning("text_content column not found in jobs_df, using title + description instead")
            jobs_df['text_content'] = jobs_df['title'] + ' ' + jobs_df['description']
        
        job_texts = jobs_df['text_content'].tolist()
        job_ids = jobs_df['job_id'].tolist()
        
        # Generate embeddings for all jobs
        embeddings = self.generate_embeddings(job_texts)
        
        # Create a dictionary mapping job_id to embedding
        job_embeddings = {job_id: embedding for job_id, embedding in zip(job_ids, embeddings)}
        
        logger.info(f"Generated embeddings for {len(job_embeddings)} jobs")
        return job_embeddings
        
    def generate_professional_embeddings(self, users_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for professional user data.
        
        Args:
            users_df: DataFrame containing user data.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping user_id to embedding vector.
        """
        # Filter for professional users only
        professionals = users_df[users_df['user_type'] == 'professional']
        
        if professionals.empty:
            logger.warning("No professional users found in users_df")
            return {}
        
        if 'profile_bio' not in professionals.columns:
            logger.error("profile_bio column not found in users_df")
            return {}
            
        professional_texts = professionals['profile_bio'].tolist()
        professional_ids = professionals['user_id'].tolist()
        
        # Generate embeddings for all professionals
        embeddings = self.generate_embeddings(professional_texts)
        
        # Create a dictionary mapping user_id to embedding
        professional_embeddings = {user_id: embedding for user_id, embedding in zip(professional_ids, embeddings)}
        
        logger.info(f"Generated embeddings for {len(professional_embeddings)} professionals")
        return professional_embeddings
        
    def generate_category_embeddings(self, categories_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for category names.
        
        Args:
            categories_df: DataFrame containing category data.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping category_id to embedding vector.
        """
        category_texts = categories_df['category_name'].tolist()
        category_ids = categories_df['category_id'].tolist()
        
        # Generate embeddings for all categories
        embeddings = self.generate_embeddings(category_texts)
        
        # Create a dictionary mapping category_id to embedding
        category_embeddings = {cat_id: embedding for cat_id, embedding in zip(category_ids, embeddings)}
        
        logger.info(f"Generated embeddings for {len(category_embeddings)} categories")
        return category_embeddings
