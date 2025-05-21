"""
Data preprocessing utilities for JibJob recommendation system.
This module handles data preparation, cleaning, and feature extraction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
import re
import logging
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_nltk_resources():
    """Ensure that required NLTK resources are downloaded."""
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource}")
            nltk.download(resource)

def clean_text(
    text: str,
    remove_numbers: bool = False,
    remove_punctuation: bool = True,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    stemming: bool = False,
    language: str = 'english'
) -> str:
    """
    Clean and preprocess text data.
    
    Args:
        text: Input text to clean.
        remove_numbers: Whether to remove numeric characters.
        remove_punctuation: Whether to remove punctuation.
        lowercase: Whether to convert to lowercase.
        remove_stopwords: Whether to remove stop words.
        stemming: Whether to apply stemming.
        language: Language for stopwords (if remove_stopwords is True).
        
    Returns:
        str: Cleaned text.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    if lowercase:
        text = text.lower()
    
    # Remove punctuation
    if remove_punctuation:
        text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        ensure_nltk_resources()  # Make sure stopwords are available
        stop_words = set(stopwords.words(language))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a string
    clean_text = ' '.join(tokens)
    
    return clean_text

def clean_dataframe_text(
    df: pd.DataFrame,
    text_columns: List[str],
    **kwargs
) -> pd.DataFrame:
    """
    Clean text columns in a DataFrame.
    
    Args:
        df: Input DataFrame.
        text_columns: List of column names containing text to clean.
        **kwargs: Arguments to pass to clean_text function.
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text columns.
    """
    df_cleaned = df.copy()
    
    for col in text_columns:
        if col in df.columns:
            logger.info(f"Cleaning text column: {col}")
            df_cleaned[f"cleaned_{col}"] = df[col].astype(str).apply(
                lambda x: clean_text(x, **kwargs)
            )
    
    return df_cleaned

def extract_tfidf_features(
    texts: List[str],
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract TF-IDF features from text.
    
    Args:
        texts: List of text documents.
        max_features: Maximum number of features (vocabulary size).
        ngram_range: Range of n-grams to consider.
        min_df: Minimum document frequency for a term to be included.
        
    Returns:
        Tuple[np.ndarray, List[str]]: 
            TF-IDF features array and list of feature names.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )
    
    features = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    return features, feature_names

def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    encoding_type: str = 'one-hot'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame.
        categorical_columns: List of categorical column names.
        encoding_type: Type of encoding ('one-hot' or 'label').
        
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            Encoded features and encoding information for later use.
    """
    df_encoded = df.copy()
    encoding_info = {}
    
    if encoding_type == 'one-hot':
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = encoder.fit_transform(df[categorical_columns])
        
        # Store encoding information
        encoding_info = {
            'encoder': encoder,
            'categorical_columns': categorical_columns,
            'encoded_features': encoder.get_feature_names_out(categorical_columns)
        }
        
        return encoded_data, encoding_info
    
    elif encoding_type == 'label':
        encoded_data = pd.DataFrame()
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[f"{col}_encoded"] = le.fit_transform(df[col])
            encoding_info[col] = {
                'encoder': le,
                'classes': le.classes_.tolist()
            }
        
        return df_encoded, encoding_info
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

def scale_numerical_features(
    df: pd.DataFrame,
    numerical_columns: List[str],
    scaler_type: str = 'standard'
) -> Tuple[np.ndarray, Any]:
    """
    Scale numerical features.
    
    Args:
        df: Input DataFrame.
        numerical_columns: List of numerical column names.
        scaler_type: Type of scaling ('standard' or 'minmax').
        
    Returns:
        Tuple[np.ndarray, Any]: Scaled features and scaler object.
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Handle missing values
    df_clean = df[numerical_columns].fillna(df[numerical_columns].mean())
    
    # Fit and transform
    scaled_data = scaler.fit_transform(df_clean)
    
    return scaled_data, scaler

def prepare_user_job_data(
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    user_id_col: str = 'user_id',
    job_id_col: str = 'job_id',
    user_text_cols: List[str] = None,
    job_text_cols: List[str] = None,
    user_categorical_cols: List[str] = None,
    job_categorical_cols: List[str] = None,
    user_numerical_cols: List[str] = None,
    job_numerical_cols: List[str] = None,
    rating_col: str = 'rating',
    comment_col: str = 'comment',
    clean_text_params: Dict[str, Any] = None,
    use_tfidf: bool = True,
    tfidf_params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Prepare user, job, and interaction data for the recommendation system.
    
    Args:
        users_df: DataFrame with user data.
        jobs_df: DataFrame with job data.
        interactions_df: DataFrame with user-job interactions.
        user_id_col: Column name for user IDs.
        job_id_col: Column name for job IDs.
        user_text_cols: Text columns in user data to process.
        job_text_cols: Text columns in job data to process.
        user_categorical_cols: Categorical columns in user data.
        job_categorical_cols: Categorical columns in job data.
        user_numerical_cols: Numerical columns in user data.
        job_numerical_cols: Numerical columns in job data.
        rating_col: Column containing ratings in interactions.
        comment_col: Column containing comments in interactions.
        clean_text_params: Parameters for text cleaning.
        use_tfidf: Whether to extract TF-IDF features from text.
        tfidf_params: Parameters for TF-IDF feature extraction.
        
    Returns:
        Dict[str, Any]: Dictionary with prepared data and feature information.
    """
    result = {}
    
    # Default parameters
    if clean_text_params is None:
        clean_text_params = {
            'remove_numbers': False,
            'remove_punctuation': True,
            'lowercase': True,
            'remove_stopwords': True,
            'stemming': False
        }
    
    if tfidf_params is None:
        tfidf_params = {
            'max_features': 1000,
            'ngram_range': (1, 2),
            'min_df': 2
        }
    
    # Process user text features
    user_text_features = None
    user_text_feature_names = None
    
    if user_text_cols:
        logger.info("Processing user text features")
        users_df_clean = clean_dataframe_text(
            users_df, user_text_cols, **clean_text_params
        )
        
        if use_tfidf:
            # Combine cleaned text columns for TF-IDF
            combined_texts = []
            for _, row in users_df_clean.iterrows():
                text_parts = [row[f"cleaned_{col}"] for col in user_text_cols if f"cleaned_{col}" in row]
                combined_texts.append(" ".join(text_parts))
                
            user_text_features, user_text_feature_names = extract_tfidf_features(
                combined_texts, **tfidf_params
            )
    
    # Process job text features
    job_text_features = None
    job_text_feature_names = None
    
    if job_text_cols:
        logger.info("Processing job text features")
        jobs_df_clean = clean_dataframe_text(
            jobs_df, job_text_cols, **clean_text_params
        )
        
        if use_tfidf:
            # Combine cleaned text columns for TF-IDF
            combined_texts = []
            for _, row in jobs_df_clean.iterrows():
                text_parts = [row[f"cleaned_{col}"] for col in job_text_cols if f"cleaned_{col}" in row]
                combined_texts.append(" ".join(text_parts))
                
            job_text_features, job_text_feature_names = extract_tfidf_features(
                combined_texts, **tfidf_params
            )
    
    # Process user categorical features
    user_cat_features = None
    user_cat_info = None
    
    if user_categorical_cols:
        logger.info("Processing user categorical features")
        user_cat_features, user_cat_info = encode_categorical_features(
            users_df, user_categorical_cols, encoding_type='one-hot'
        )
    
    # Process job categorical features
    job_cat_features = None
    job_cat_info = None
    
    if job_categorical_cols:
        logger.info("Processing job categorical features")
        job_cat_features, job_cat_info = encode_categorical_features(
            jobs_df, job_categorical_cols, encoding_type='one-hot'
        )
    
    # Process user numerical features
    user_num_features = None
    user_num_scaler = None
    
    if user_numerical_cols:
        logger.info("Processing user numerical features")
        user_num_features, user_num_scaler = scale_numerical_features(
            users_df, user_numerical_cols, scaler_type='standard'
        )
    
    # Process job numerical features
    job_num_features = None
    job_num_scaler = None
    
    if job_numerical_cols:
        logger.info("Processing job numerical features")
        job_num_features, job_num_scaler = scale_numerical_features(
            jobs_df, job_numerical_cols, scaler_type='standard'
        )
    
    # Combine features for users
    user_features = []
    
    if user_text_features is not None:
        user_features.append(user_text_features)
    
    if user_cat_features is not None:
        user_features.append(user_cat_features)
    
    if user_num_features is not None:
        user_features.append(user_num_features)
    
    if user_features:
        # Handle different formats (sparse/dense) and concatenate
        dense_features = []
        for feat in user_features:
            if isinstance(feat, np.ndarray):
                dense_features.append(feat)
            else:  # Assuming sparse matrix
                dense_features.append(feat.toarray())
                
        final_user_features = np.hstack(dense_features)
        result['user_features'] = torch.tensor(final_user_features, dtype=torch.float)
    
    # Combine features for jobs
    job_features = []
    
    if job_text_features is not None:
        job_features.append(job_text_features)
    
    if job_cat_features is not None:
        job_features.append(job_cat_features)
    
    if job_num_features is not None:
        job_features.append(job_num_features)
    
    if job_features:
        # Handle different formats (sparse/dense) and concatenate
        dense_features = []
        for feat in job_features:
            if isinstance(feat, np.ndarray):
                dense_features.append(feat)
            else:  # Assuming sparse matrix
                dense_features.append(feat.toarray())
                
        final_job_features = np.hstack(dense_features)
        result['job_features'] = torch.tensor(final_job_features, dtype=torch.float)
    
    # Process interaction data
    logger.info("Processing interaction data")
    
    # Create user and job ID mappings
    unique_users = users_df[user_id_col].unique()
    unique_jobs = jobs_df[job_id_col].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    job_to_idx = {job_id: idx for idx, job_id in enumerate(unique_jobs)}
    
    # Filter interactions to include only known users and jobs
    valid_interactions = interactions_df[
        (interactions_df[user_id_col].isin(unique_users)) & 
        (interactions_df[job_id_col].isin(unique_jobs))
    ]
    
    # Map IDs to indices
    user_indices = valid_interactions[user_id_col].map(user_to_idx).values
    job_indices = valid_interactions[job_id_col].map(job_to_idx).values
    
    # Get ratings if available
    if rating_col in valid_interactions.columns:
        ratings = valid_interactions[rating_col].values
    else:
        # Default to neutral ratings
        ratings = np.ones(len(valid_interactions)) * 0.5
    
    # Convert to tensor format
    result['user_indices'] = torch.tensor(user_indices, dtype=torch.long)
    result['job_indices'] = torch.tensor(job_indices, dtype=torch.long)
    result['ratings'] = torch.tensor(ratings, dtype=torch.float)
    
    # Store mappings for later use
    result['user_to_idx'] = user_to_idx
    result['job_to_idx'] = job_to_idx
    result['idx_to_user'] = {idx: user_id for user_id, idx in user_to_idx.items()}
    result['idx_to_job'] = {idx: job_id for job_id, idx in job_to_idx.items()}
    
    # Store raw data for reference
    result['users_df'] = users_df
    result['jobs_df'] = jobs_df
    result['interactions_df'] = valid_interactions
    
    return result

def save_processed_data(data_dict: Dict[str, Any], output_dir: str):
    """
    Save processed data to disk.
    
    Args:
        data_dict: Dictionary with processed data.
        output_dir: Directory to save the data.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tensors
    for key in ['user_indices', 'job_indices', 'ratings', 'user_features', 'job_features']:
        if key in data_dict and data_dict[key] is not None:
            torch.save(data_dict[key], os.path.join(output_dir, f"{key}.pt"))
    
    # Save mappings
    for key in ['user_to_idx', 'job_to_idx', 'idx_to_user', 'idx_to_job']:
        if key in data_dict:
            # Convert keys to strings for JSON serialization
            mapping = {str(k): v if not isinstance(v, (np.int64, np.int32)) else int(v) 
                      for k, v in data_dict[key].items()}
            
            with open(os.path.join(output_dir, f"{key}.json"), 'w') as f:
                json.dump(mapping, f)
    
    # Save DataFrames
    for key in ['users_df', 'jobs_df', 'interactions_df']:
        if key in data_dict and data_dict[key] is not None:
            data_dict[key].to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)

def load_processed_data(input_dir: str) -> Dict[str, Any]:
    """
    Load processed data from disk.
    
    Args:
        input_dir: Directory with saved data.
        
    Returns:
        Dict[str, Any]: Dictionary with loaded data.
    """
    result = {}
    
    # Load tensors
    for key in ['user_indices', 'job_indices', 'ratings', 'user_features', 'job_features']:
        file_path = os.path.join(input_dir, f"{key}.pt")
        if os.path.exists(file_path):
            result[key] = torch.load(file_path)
    
    # Load mappings
    for key in ['user_to_idx', 'job_to_idx', 'idx_to_user', 'idx_to_job']:
        file_path = os.path.join(input_dir, f"{key}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                mapping = json.load(f)
                # Convert string keys back to integers where appropriate
                if key in ['idx_to_user', 'idx_to_job']:
                    mapping = {int(k): v for k, v in mapping.items()}
                result[key] = mapping
    
    # Load DataFrames
    for key in ['users_df', 'jobs_df', 'interactions_df']:
        file_path = os.path.join(input_dir, f"{key}.csv")
        if os.path.exists(file_path):
            result[key] = pd.read_csv(file_path)
    
    return result

def create_train_test_split(
    interactions: pd.DataFrame,
    user_col: str = 'user_id',
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interaction data into training, validation, and test sets.
    
    Args:
        interactions: DataFrame with user-job interactions.
        user_col: Column name for user IDs.
        test_ratio: Fraction of data to use for testing.
        val_ratio: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Training, validation, and test DataFrames.
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    train_data, test_data, val_data = [], [], []
    
    # Group by user
    user_groups = interactions.groupby(user_col)
    
    for user_id, group in user_groups:
        n_items = len(group)
        
        if n_items >= 3:  # Ensure we have enough interactions
            # Shuffle the user's interactions
            shuffled_idx = np.random.permutation(n_items)
            group_array = group.values
            
            # Determine split sizes
            test_size = max(1, int(n_items * test_ratio))
            val_size = max(1, int(n_items * val_ratio))
            train_size = n_items - test_size - val_size
            
            # Split the data
            train_idx = shuffled_idx[:train_size]
            val_idx = shuffled_idx[train_size:train_size + val_size]
            test_idx = shuffled_idx[train_size + val_size:]
            
            # Allocate data to respective sets
            train_data.append(group.iloc[train_idx])
            val_data.append(group.iloc[val_idx])
            test_data.append(group.iloc[test_idx])
        else:
            # Not enough data, put everything in training
            train_data.append(group)
    
    # Combine into DataFrames
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    return train_df, val_df, test_df

def process_job_descriptions(jobs_df: pd.DataFrame, 
                            text_columns: List[str] = ['title', 'description'],
                            remove_stopwords: bool = True) -> pd.DataFrame:
    """
    Process job descriptions by cleaning text fields.
    
    Args:
        jobs_df: DataFrame containing job information
        text_columns: List of column names containing text to process
        remove_stopwords: Whether to remove stopwords during text cleaning
        
    Returns:
        Processed DataFrame with cleaned text columns
    """
    ensure_nltk_resources()
    result_df = jobs_df.copy()
    
    # Process all text columns at once
    result_df = clean_dataframe_text(
        result_df,
        text_columns=text_columns,
        remove_stopwords=remove_stopwords
    )
    logger.info(f"Processed text columns: {text_columns}")
    
    return result_df

def normalize_ratings(interactions_df: pd.DataFrame, 
                     rating_col: str = 'rating',
                     min_val: float = 0.0,
                     max_val: float = 1.0) -> pd.DataFrame:
    """
    Normalize ratings to a specified range.
    
    Args:
        interactions_df: DataFrame containing user-job interactions
        rating_col: Name of the column containing ratings
        min_val: Minimum value for normalized ratings
        max_val: Maximum value for normalized ratings
        
    Returns:
        DataFrame with normalized ratings
    """
    result_df = interactions_df.copy()
    
    if rating_col in result_df.columns:
        # Get current min and max
        current_min = result_df[rating_col].min()
        current_max = result_df[rating_col].max()
        
        # Avoid division by zero
        if current_max > current_min:
            # Apply min-max normalization
            result_df[rating_col] = min_val + (result_df[rating_col] - current_min) * \
                                   (max_val - min_val) / (current_max - current_min)
            logger.info(f"Normalized ratings from [{current_min}, {current_max}] to [{min_val}, {max_val}]")
        else:
            logger.warning(f"Cannot normalize ratings: min={current_min}, max={current_max}")
    else:
        logger.warning(f"Rating column '{rating_col}' not found in DataFrame")
    
    return result_df
