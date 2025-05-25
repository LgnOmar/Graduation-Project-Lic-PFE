"""
Data preprocessing utilities for JibJob recommendation system.
This module handles data preparation, cleaning, and feature extraction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
# Corrected import for newer scikit-learn versions if LabelEncoder is used for features,
# though it's primarily for target encoding or simple ordinal.
# For features, OneHotEncoder is generally preferred as used.
from sklearn.preprocessing import LabelEncoder # Kept as in original for LabelEncoder use case
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
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords'
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            logger.info(f"Downloading NLTK resource: {resource_name}")
            nltk.download(resource_name)

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
        text = re.sub(r'[^\w\s]', ' ', text) # Replaces punctuation with space
        text = re.sub(r'\s+', ' ', text).strip() # Consolidate multiple spaces
    
    # Remove numbers
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # Tokenize
    # ensure_nltk_resources() was called here for 'punkt' implicitly by word_tokenize
    # It's better called once globally or at the start of a batch process.
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        # ensure_nltk_resources() call REMOVED FROM HERE
        stop_words = set(stopwords.words(language))
        tokens = [token for token in tokens if token.lower() not in stop_words and token.isalpha()] # Keep only alpha tokens
    else:
        tokens = [token for token in tokens if token.isalpha()] # Keep only alpha tokens even if not removing stopwords

    # Apply stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Join tokens back into a string
    cleaned_text = ' '.join(tokens) # Renamed from clean_text to avoid confusion
    
    return cleaned_text

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
    
    # Ensure NLTK resources once before processing all columns/rows in this call
    # This covers 'punkt' for word_tokenize and 'stopwords' if used.
    ensure_nltk_resources() 
    
    for col in text_columns:
        if col in df.columns:
            logger.info(f"Cleaning text column: {col}")
            df_cleaned[f"cleaned_{col}"] = df[col].astype(str).apply(
                lambda x: clean_text(x, **kwargs)
            )
        else:
            logger.warning(f"Text column '{col}' not found in DataFrame. Skipping.")

    return df_cleaned

def extract_tfidf_features(
    texts: List[str],
    max_features: int = 1000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2
) -> Tuple[Union[np.ndarray, Any], List[str]]: # Adjusted return type for sparse matrix
    """
    Extract TF-IDF features from text.
    
    Args:
        texts: List of text documents.
        max_features: Maximum number of features (vocabulary size).
        ngram_range: Range of n-grams to consider.
        min_df: Minimum document frequency for a term to be included.
        
    Returns:
        Tuple[Union[np.ndarray, Any], List[str]]: 
            TF-IDF features array (can be sparse) and list of feature names.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        stop_words='english' # TF-IDF can also handle stopwords
    )
    
    features = vectorizer.fit_transform(texts) # This returns a sparse matrix
    feature_names = vectorizer.get_feature_names_out().tolist()
    
    return features, feature_names # Return sparse matrix directly

def encode_categorical_features(
    df: pd.DataFrame,
    categorical_columns: List[str],
    encoding_type: str = 'one-hot'
) -> Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]: # Adjusted return type for LabelEncoder
    """
    Encode categorical features.
    
    Args:
        df: Input DataFrame.
        categorical_columns: List of categorical column names.
        encoding_type: Type of encoding ('one-hot' or 'label').
        
    Returns:
        Tuple[Union[np.ndarray, pd.DataFrame], Dict[str, Any]]:
            Encoded features and encoding information for later use.
    """
    df_to_encode = df.copy()
    encoding_info = {}
    
    if encoding_type == 'one-hot':
        # Ensure categorical columns exist
        valid_cols = [col for col in categorical_columns if col in df_to_encode.columns]
        if not valid_cols:
            logger.warning("No valid categorical columns found for one-hot encoding.")
            return np.array([]), encoding_info # Return empty array if no columns

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore') # sparse_output for dense array
        encoded_data = encoder.fit_transform(df_to_encode[valid_cols])
        
        encoding_info = {
            'encoder': encoder,
            'categorical_columns': valid_cols, # Use valid_cols
            'encoded_features': encoder.get_feature_names_out(valid_cols).tolist()
        }
        
        return encoded_data, encoding_info
    
    elif encoding_type == 'label':
        # This path modifies df_encoded and returns it, which is a bit inconsistent
        # with one-hot. Usually, you'd return just the encoded columns as an array.
        # For now, keeping original logic but adding safety.
        df_result = df_to_encode.copy() 
        encoded_columns_data = []

        for col in categorical_columns:
            if col in df_result.columns:
                le = LabelEncoder()
                # Fit transform and store the column
                encoded_col_data = le.fit_transform(df_result[col].astype(str)) # Ensure string type
                df_result[f"{col}_encoded"] = encoded_col_data
                encoded_columns_data.append(encoded_col_data.reshape(-1,1))

                encoding_info[col] = {
                    'encoder': le,
                    'classes': le.classes_.tolist()
                }
            else:
                logger.warning(f"Categorical column '{col}' not found for label encoding.")
        
        if not encoded_columns_data:
             return pd.DataFrame(), encoding_info # Return empty DataFrame

        # To be consistent, maybe return np.hstack(encoded_columns_data)
        # But the original returns the modified DataFrame
        return df_result, encoding_info # Original behavior was returning df_encoded
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

def scale_numerical_features(
    df: pd.DataFrame,
    numerical_columns: List[str],
    scaler_type: str = 'standard'
) -> Tuple[Union[np.ndarray, None], Optional[Any]]:
    """
    Scale numerical features.
    
    Args:
        df: Input DataFrame.
        numerical_columns: List of numerical column names.
        scaler_type: Type of scaling ('standard' or 'minmax').
        
    Returns:
        Tuple[Union[np.ndarray, None], Optional[Any]]: Scaled features and scaler object. Returns None if no valid columns.
    """
    valid_numerical_cols = [col for col in numerical_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    if not valid_numerical_cols:
        logger.warning("No valid numerical columns found for scaling.")
        return None, None

    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Handle missing values only on the valid subset
    df_clean = df[valid_numerical_cols].fillna(df[valid_numerical_cols].mean())
    
    # Fit and transform
    scaled_data = scaler.fit_transform(df_clean)
    
    return scaled_data, scaler

def prepare_user_job_data(
    users_df: pd.DataFrame,
    jobs_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    categories_master: Optional[pd.DataFrame] = None,
    user_id_col: str = 'user_id',
    job_id_col: str = 'job_id',
    user_text_cols: Optional[List[str]] = None,
    job_text_cols: Optional[List[str]] = None,
    user_categorical_cols: Optional[List[str]] = None,
    job_categorical_cols: Optional[List[str]] = None,
    user_numerical_cols: Optional[List[str]] = None,
    job_numerical_cols: Optional[List[str]] = None,
    rating_col: str = 'rating',
    comment_col: Optional[str] = None,
    clean_text_params: Optional[Dict[str, Any]] = None,
    use_tfidf: bool = True,
    tfidf_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare user, job, and interaction data for the recommendation system.
    This version supports explicit multi-hot encoding of professional selected_categories.
    """
    ensure_nltk_resources()
    result = {}
    
    # Default parameters
    if clean_text_params is None:
        clean_text_params = {
            'remove_numbers': False,
            'remove_punctuation': True,
            'lowercase': True,
            'remove_stopwords': True, # Note: BERT models often work better with stopwords
            'stemming': False # BERT doesn't need stemming
        }
    
    if tfidf_params is None:
        tfidf_params = {
            'max_features': 1000, # This might be too small for good representation
            'ngram_range': (1, 2),
            'min_df': 2
        }
    
    # Process user text features
    final_user_text_features_for_GCN = None
    if user_text_cols:
        logger.info("Processing user text features")
        # Assuming users_df is available and has user_text_cols
        users_df_clean = clean_dataframe_text( 
            users_df, user_text_cols, **clean_text_params
        )
        
        if use_tfidf:
            combined_texts = []
            for _, row in users_df_clean.iterrows():
                text_parts = [str(row[f"cleaned_{col}"]) for col in user_text_cols if f"cleaned_{col}" in row and pd.notna(row[f"cleaned_{col}"])]
                combined_texts.append(" ".join(text_parts))
                
            if combined_texts:
                user_text_features_sparse, _ = extract_tfidf_features(
                    combined_texts, **tfidf_params
                )
                if user_text_features_sparse.shape[0] > 0: # Check if any features were extracted
                    final_user_text_features_for_GCN = user_text_features_sparse
    
    # Process job text features (primarily for TF-IDF path, BERT embeddings handled separately)
    final_job_text_features_for_GCN = None
    if job_text_cols: # e.g. ['title', 'description']
        logger.info("Processing job text features for potential GCN input (e.g., TF-IDF)")
        # jobs_df comes from system's data generation.
        # It's assumed process_job_descriptions in demo.py already cleaned 'title', 'description'
        # and generated 'cleaned_title', 'cleaned_description' for BERT.
        # If TF-IDF is used for GCN node features, we use these.
        
        # This part uses jobs_df directly which might not have "cleaned_" columns yet if
        # process_job_descriptions was not called before. For safety, clean them if TF-IDF.
        if use_tfidf:
            jobs_df_for_tfidf = clean_dataframe_text(
                jobs_df, job_text_cols, **clean_text_params
            )
            combined_texts = []
            for _, row in jobs_df_for_tfidf.iterrows():
                # Use cleaned_{col} which clean_dataframe_text creates
                text_parts = [str(row[f"cleaned_{col}"]) for col in job_text_cols if f"cleaned_{col}" in row and pd.notna(row[f"cleaned_{col}"])]
                combined_texts.append(" ".join(text_parts))
            
            if combined_texts:
                job_text_features_sparse, _ = extract_tfidf_features(
                    combined_texts, **tfidf_params
                )
                if job_text_features_sparse.shape[0] > 0:
                    final_job_text_features_for_GCN = job_text_features_sparse
    
    # Process user categorical features
    user_cat_features = None
    if user_categorical_cols:
        logger.info("Processing user categorical features")
        user_cat_features, _ = encode_categorical_features(
            users_df, user_categorical_cols, encoding_type='one-hot'
        )
    
    # Process job categorical features
    job_cat_features = None
    if job_categorical_cols: # e.g. ['category']
        logger.info("Processing job categorical features")
        job_cat_features, _ = encode_categorical_features(
            jobs_df, job_categorical_cols, encoding_type='one-hot'
        )
        if job_cat_features is not None and job_cat_features.shape[1] == 0: # If no features produced
             job_cat_features = None

    # Process user numerical features
    user_num_features = None
    if user_numerical_cols:
        logger.info("Processing user numerical features")
        user_num_features, _ = scale_numerical_features(
            users_df, user_numerical_cols, scaler_type='standard'
        )
    
    # Process job numerical features
    job_num_features = None
    if job_numerical_cols: # e.g. ['avg_rating_generated_by_demo']
        logger.info("Processing job numerical features")
        job_num_features, _ = scale_numerical_features(
            jobs_df, job_numerical_cols, scaler_type='standard'
        )

    # Combine features for users for GCN
    # IMPORTANT: This part constructs GCN node features. BERT embeddings are usually separate.
    # If BERT embeddings are the primary job features, job_features list might only contain those.
    # For this function, we prepare TF-IDF/categorical/numerical if specified.
    
    user_gcn_features_list = []
    if final_user_text_features_for_GCN is not None:
        user_gcn_features_list.append(final_user_text_features_for_GCN)
    if user_cat_features is not None and user_cat_features.size > 0 :
        user_gcn_features_list.append(user_cat_features)
    if user_num_features is not None and user_num_features.size > 0:
        user_gcn_features_list.append(user_num_features)
    
    if user_gcn_features_list:
        dense_features = []
        for feat in user_gcn_features_list:
            if hasattr(feat, 'toarray'): # Check if it's a sparse matrix
                dense_features.append(feat.toarray())
            else:
                dense_features.append(feat)
        
        if dense_features: # Check if list is not empty
            final_user_features = np.hstack(dense_features)
            result['user_features'] = torch.tensor(final_user_features, dtype=torch.float)
        else:
            result['user_features'] = None # Or torch.empty(len(users_df), 0)
    else: # No user features provided or extracted
         result['user_features'] = torch.empty(len(users_df), 0) # Placeholder for GCN if no features
         logger.info("No GCN features generated for users.")

    # Combine features for jobs for GCN
    job_gcn_features_list = []
    if final_job_text_features_for_GCN is not None: # TF-IDF primarily
        job_gcn_features_list.append(final_job_text_features_for_GCN)
    # Note: BERT embeddings would typically be loaded/generated elsewhere and passed to the GCN model,
    # or if this function is meant to generate *all* features, it needs a path for BERT.
    # Assuming `job_features` here are non-BERT GCN node features for now.
    if job_cat_features is not None and job_cat_features.size > 0: # e.g. one-hot encoded 'category'
        job_gcn_features_list.append(job_cat_features)
    if job_num_features is not None and job_num_features.size > 0:
        job_gcn_features_list.append(job_num_features)
        
    if job_gcn_features_list:
        dense_features = []
        for feat in job_gcn_features_list:
            if hasattr(feat, 'toarray'): # Check if it's a sparse matrix
                dense_features.append(feat.toarray())
            else:
                dense_features.append(feat)
        if dense_features:
            final_job_features = np.hstack(dense_features)
            result['job_features'] = torch.tensor(final_job_features, dtype=torch.float)
        else:
            result['job_features'] = None # Or torch.empty(len(jobs_df), 0)

    else: # No job features to be used by GCN (e.g. relying purely on IDs or external BERT embeddings)
         # If BERT embeddings are prepared by bert_embeddings.py and used directly by GCN, this part is fine.
         # This assumes the recommender.py or gcn.py will handle that.
         # The recommender might get job_embeddings from bert_embeddings.py
         # and user_features/job_features from here IF they exist (e.g. categorical).
         result['job_features'] = torch.empty(len(jobs_df), 0) # Placeholder if no TFIDF/Cat/Num
         logger.info("No GCN features (TFIDF/Categorical/Numerical) generated for jobs by this function.")
         logger.info("It is assumed BERT embeddings are handled separately if they are the primary job features for GCN.")


    # Process interaction data
    logger.info("Processing interaction data")
    
    unique_users = users_df[user_id_col].unique()
    unique_jobs = jobs_df[job_id_col].unique()
    
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    job_to_idx = {job_id: idx for idx, job_id in enumerate(unique_jobs)}
    
    # Add IDs not in unique lists to ensure all users/jobs in interactions get an index
    # This is important if interactions_df contains users/jobs not in users_df/jobs_df
    current_max_user_idx = len(user_to_idx)
    for uid in interactions_df[user_id_col].unique():
        if uid not in user_to_idx:
            user_to_idx[uid] = current_max_user_idx
            current_max_user_idx +=1
            # Potentially add this user to a 'master' unique user list if needed downstream

    current_max_job_idx = len(job_to_idx)
    for jid in interactions_df[job_id_col].unique():
        if jid not in job_to_idx:
            job_to_idx[jid] = current_max_job_idx
            current_max_job_idx += 1


    valid_interactions = interactions_df.copy() # Assume all interactions are valid now due to mapping extension
    
    user_indices = valid_interactions[user_id_col].map(user_to_idx).values
    job_indices = valid_interactions[job_id_col].map(job_to_idx).values
    
    if pd.isna(user_indices).any() or pd.isna(job_indices).any():
        logger.error("NaN found in user or job indices after mapping. This should not happen if mapping is exhaustive.")
        # Handle error: e.g. filter out rows with NaN indices, or raise exception
        raise ValueError("NaN indices created. Check user/job ID mapping logic with interactions.")

    if rating_col in valid_interactions.columns:
        ratings = valid_interactions[rating_col].values
        # Normalize ratings if GCN expects a certain range (e.g. 0-1)
        # The demo uses raw ratings. If normalized ratings are desired:
        # ratings_df_temp = pd.DataFrame({'rating': ratings})
        # ratings_normalized = normalize_ratings(ratings_df_temp, rating_col='rating')['rating'].values
        # result['ratings'] = torch.tensor(ratings_normalized, dtype=torch.float)
        result['ratings'] = torch.tensor(ratings, dtype=torch.float)
    else:
        logger.warning(f"Rating column '{rating_col}' not found. Defaulting to 0.5 for interactions.")
        ratings = np.ones(len(valid_interactions)) * 0.5 
        result['ratings'] = torch.tensor(ratings, dtype=torch.float)

    result['user_indices'] = torch.tensor(user_indices, dtype=torch.long)
    result['job_indices'] = torch.tensor(job_indices, dtype=torch.long)
    
    result['user_to_idx'] = user_to_idx
    result['job_to_idx'] = job_to_idx
    result['idx_to_user'] = {idx: user_id for user_id, idx in user_to_idx.items()}
    result['idx_to_job'] = {idx: job_id for job_id, idx in job_to_idx.items()}
    
    result['users_df'] = users_df
    result['jobs_df'] = jobs_df
    result['interactions_df'] = valid_interactions
    
    # Log shapes for GCN inputs
    if 'user_features' in result and result['user_features'] is not None:
        logger.info(f"Shape of final user_features for GCN: {result['user_features'].shape}")
    if 'job_features' in result and result['job_features'] is not None:
        logger.info(f"Shape of final job_features for GCN (e.g. TFIDF/Cat/Num): {result['job_features'].shape}")
    logger.info(f"Number of unique users for GCN: {len(user_to_idx)}")
    logger.info(f"Number of unique jobs for GCN: {len(job_to_idx)}")


    return result

def save_processed_data(data_dict: Dict[str, Any], output_dir: str):
    """
    Save processed data to disk.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for key in ['user_indices', 'job_indices', 'ratings', 'user_features', 'job_features']:
        if key in data_dict and data_dict[key] is not None and data_dict[key].numel() > 0: # Check if tensor is not empty
            torch.save(data_dict[key], os.path.join(output_dir, f"{key}.pt"))
    
    for key in ['user_to_idx', 'job_to_idx', 'idx_to_user', 'idx_to_job']:
        if key in data_dict:
            mapping = {str(k): (int(v) if isinstance(v, (np.int64, np.int32)) else v)
                       for k, v in data_dict[key].items()}
            with open(os.path.join(output_dir, f"{key}.json"), 'w') as f:
                json.dump(mapping, f, indent=4)
    
    for key in ['users_df', 'jobs_df', 'interactions_df']:
        if key in data_dict and data_dict[key] is not None and not data_dict[key].empty:
            data_dict[key].to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)

def load_processed_data(input_dir: str) -> Dict[str, Any]:
    """
    Load processed data from disk.
    """
    result = {}
    
    for key in ['user_indices', 'job_indices', 'ratings', 'user_features', 'job_features']:
        file_path = os.path.join(input_dir, f"{key}.pt")
        if os.path.exists(file_path):
            result[key] = torch.load(file_path)
            if result[key].numel() == 0 and key.endswith('_features'): # Handle empty features if saved
                 logger.info(f"Loaded empty tensor for {key}. Setting to None or expected empty tensor shape.")
                 # result[key] = None # Or a correctly shaped empty tensor
    
    for key in ['user_to_idx', 'job_to_idx', 'idx_to_user', 'idx_to_job']:
        file_path = os.path.join(input_dir, f"{key}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                mapping_str_keys = json.load(f)
                # Convert keys based on mapping type
                if key.startswith('idx_to_'): # Keys are integer indices
                     result[key] = {int(k_str): val for k_str, val in mapping_str_keys.items()}
                elif key.endswith('_to_idx'): # Values are integer indices
                     result[key] = {k_str: int(val) for k_str, val in mapping_str_keys.items()}
                else: # Default (should not happen with current keys)
                     result[key] = mapping_str_keys

    for key in ['users_df', 'jobs_df', 'interactions_df']:
        file_path = os.path.join(input_dir, f"{key}.csv")
        if os.path.exists(file_path):
            result[key] = pd.read_csv(file_path)
    
    return result

def create_train_test_split(
    interactions: pd.DataFrame,
    user_col: str = 'user_id',
    test_ratio: float = 0.2, # test_ratio applied to items per user
    val_ratio: float = 0.1,  # val_ratio applied to items per user from remaining
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interaction data into training, validation, and test sets per user.
    Ensures each user has at least one interaction in train if possible.
    """
    if interactions.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    np.random.seed(seed)
    
    train_list, val_list, test_list = [], [], []
    
    user_groups = interactions.groupby(user_col)
    
    for _, group in user_groups:
        n_items = len(group)
        group_shuffled = group.sample(frac=1, random_state=seed) # Shuffle items for this user
        
        if n_items < 3: # If 1 or 2 items, all go to train
            train_list.append(group_shuffled)
            continue

        # Calculate number of test and val items
        test_size = max(1, int(n_items * test_ratio))
        val_size = max(1, int(n_items * val_ratio)) # val_ratio on original count
        
        # Ensure train_size is at least 1 if n_items - test_size - val_size would be < 1
        train_size = n_items - test_size - val_size
        if train_size < 1 : # if e.g. 3 items, test=1, val=1 => train=1
            # Adjust: prioritize train, then test, then val
            if n_items == 3: test_size = 1; val_size=1; train_size=1;
            elif n_items == 2: test_size =1; val_size=0; train_size=1; # Test set is important
            elif n_items == 1: test_size =0; val_size=0; train_size=1;

        # Correct indexing for split based on sizes
        current_idx = 0
        
        # Test items
        test_list.append(group_shuffled.iloc[current_idx : current_idx + test_size])
        current_idx += test_size
        
        # Validation items
        val_list.append(group_shuffled.iloc[current_idx : current_idx + val_size])
        current_idx += val_size
        
        # Training items (rest)
        train_list.append(group_shuffled.iloc[current_idx:])

    train_df = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    val_df = pd.concat(val_list, ignore_index=True) if val_list else pd.DataFrame()
    test_df = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    
    return train_df, val_df, test_df


def process_job_descriptions(jobs_df: pd.DataFrame, 
                            text_columns: List[str] = ['title', 'description'],
                            remove_stopwords: bool = True, # Passed to clean_dataframe_text via kwargs
                            **clean_kwargs) -> pd.DataFrame:
    """
    Process job descriptions by cleaning text fields.
    """
    # ensure_nltk_resources() # Moved into clean_dataframe_text or called before this
    result_df = jobs_df.copy()
    
    # clean_kwargs can include remove_stopwords, stemming, etc.
    # Combine default remove_stopwords with any provided in clean_kwargs
    final_clean_kwargs = {'remove_stopwords': remove_stopwords, **clean_kwargs}

    result_df = clean_dataframe_text(
        result_df,
        text_columns=text_columns,
        **final_clean_kwargs
    )
    logger.info(f"Processed text columns for job descriptions: {text_columns}")
    
    return result_df

def normalize_ratings(interactions_df: pd.DataFrame, 
                     rating_col: str = 'rating',
                     min_val: float = 0.0, # For 0-1 normalization
                     max_val: float = 1.0) -> pd.DataFrame:
    """
    Normalize ratings to a specified range.
    """
    result_df = interactions_df.copy()
    
    if rating_col in result_df.columns:
        current_min = result_df[rating_col].min()
        current_max = result_df[rating_col].max()
        
        if pd.isna(current_min) or pd.isna(current_max):
            logger.warning(f"Ratings in '{rating_col}' contain NaNs. Cannot normalize before handling NaNs.")
            return result_df # Or fill NaNs first

        if current_max > current_min:
            result_df[rating_col] = min_val + (result_df[rating_col] - current_min) * \
                                   (max_val - min_val) / (current_max - current_min)
            logger.info(f"Normalized ratings from '{rating_col}' column from range [{current_min}, {current_max}] to [{min_val}, {max_val}]")
        elif current_max == current_min : # All ratings are the same
             # Set all to mid-point of target range, or min_val, or avg(min_val, max_val)
             result_df[rating_col] = (min_val + max_val) / 2.0 
             logger.info(f"All ratings in '{rating_col}' are '{current_min}'. Normalized to '{(min_val + max_val) / 2.0}'.")
        # No else needed as case where current_max < current_min is impossible with .min()/.max()
    else:
        logger.warning(f"Rating column '{rating_col}' not found. Cannot normalize ratings.")
    
    return result_df

def load_jibjob_csv_data(
    data_dir: str = "sample_data"
) -> Dict[str, pd.DataFrame]:
    """
    Load users, jobs, interactions, and job_categories_master_list from CSV files.
    Returns a dict with DataFrames.
    """
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    jobs = pd.read_csv(os.path.join(data_dir, "jobs.csv"))
    interactions = pd.read_csv(os.path.join(data_dir, "interactions.csv"))
    categories = pd.read_csv(os.path.join(data_dir, "job_categories_master_list.csv"))
    return {
        "users": users,
        "jobs": jobs,
        "interactions": interactions,
        "categories": categories
    }


def encode_professional_selected_categories(
    users_df: pd.DataFrame,
    categories_master: pd.DataFrame,
    user_id_col: str = "user_id",
    selected_categories_col: str = "selected_categories",
    user_type_col: str = "user_type"
) -> np.ndarray:
    """
    For each user, encode their selected_categories as a multi-hot vector (if professional), else zeros.
    Returns a numpy array of shape (num_users, num_categories).
    """
    cat_list = categories_master["category_name"].tolist()
    cat_idx = {cat: i for i, cat in enumerate(cat_list)}
    num_users = len(users_df)
    num_cats = len(cat_list)
    features = np.zeros((num_users, num_cats), dtype=np.float32)
    for idx, row in users_df.iterrows():
        if row.get(user_type_col, "") == "professional":
            selected = str(row.get(selected_categories_col, "")).split(";")
            for cat in selected:
                cat = cat.strip()
                if cat in cat_idx:
                    features[idx, cat_idx[cat]] = 1.0
    return features