"""
text_embedders.py

Purpose:
    Generate BERT embeddings for text fields (job titles/descriptions, category descriptions) for use as node features.

Key Functions:
    - get_bert_embeddings(texts: List[str], model_name: str, batch_size: int, pooling: str = 'cls') -> np.ndarray
        - Tokenizes and encodes texts using Hugging Face Transformers.
        - Supports [CLS] or mean pooling.
        - Returns numpy array or torch tensor of embeddings.

Inputs:
    - texts: List of strings to embed.
    - model_name: Name of pre-trained BERT model (e.g., 'bert-base-uncased').
    - batch_size: Batch size for embedding.
    - pooling: Pooling strategy ('cls' or 'mean').

Outputs:
    - Embeddings as np.ndarray or torch.Tensor.

High-Level Logic:
    1. Load pre-trained BERT model and tokenizer.
    2. Tokenize texts, pad, and batch.
    3. Pass through BERT, extract embeddings.
    4. Pool as specified.
    5. Return embeddings.
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import List

def get_bert_embeddings(texts: List[str], model_name: str = 'bert-base-uncased', batch_size: int = 16, pooling: str = 'cls', max_length: int = 128):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if pooling == 'cls':
                batch_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling == 'mean':
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                masked = outputs.last_hidden_state * mask
                summed = masked.sum(1)
                counts = mask.sum(1)
                batch_emb = (summed / counts).cpu().numpy()
            else:
                raise ValueError('Invalid pooling type')
            embeddings.append(batch_emb)
    return np.vstack(embeddings)
