"""
main.py

Purpose:
    Main script to run the full pipeline: dataset generation, feature engineering, graph construction, model training, and evaluation.

Key Functions:
    - main(): Entry point for the pipeline.
    - Calls each phase in order, using config for parameters.

High-Level Logic:
    1. Load config and set random seed.
    2. Generate dataset (if not already present).
    3. Generate BERT embeddings.
    4. Build graph.
    5. Train model.
    6. Evaluate and print metrics.
"""

import os
import pandas as pd
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.transforms import RandomLinkSplit

from utils.common import load_config, set_seed
from dataset_generator.categories_generator import generate_categories
from dataset_generator.users_generator import generate_professionals, generate_clients
from dataset_generator.jobs_generator import generate_jobs
from dataset_generator.interactions_generator import generate_interactions
from features.text_embedders import get_bert_embeddings
from graph.graph_builder import build_hetero_graph
from training.trainer import train_model
from training.negative_sampler import sample_negative_edges
from models.hetgcn_recommender import HetGCNRecommender
from evaluation.metrics import precision_at_k, recall_at_k, ndcg_at_k, auc_roc, mae, rmse

def generate_top_n_recommendations(trained_model, graph_data_for_inference, professional_node_idx, all_job_node_indices, N, all_positive_interactions_for_professional, job_idx_to_original_id_map):
    trained_model.eval()
    device = next(trained_model.parameters()).device
    graph_data_for_inference = graph_data_for_inference.to(device)
    with torch.no_grad():
        final_node_embeddings = trained_model(graph_data_for_inference)
    prof_emb = final_node_embeddings['Professional'][professional_node_idx]
    job_embs = final_node_embeddings['Job'][all_job_node_indices]
    scores = torch.matmul(prof_emb.unsqueeze(0), job_embs.T).squeeze(0)
    # Filter out jobs already interacted with
    recs = []
    for idx, score in zip(all_job_node_indices.tolist(), scores.tolist()):
        if idx not in all_positive_interactions_for_professional:
            recs.append((idx, score))
    recs.sort(key=lambda x: x[1], reverse=True)
    top_n = recs[:N]
    # Map back to original job IDs
    return [(job_idx_to_original_id_map[idx], score) for idx, score in top_n if idx in job_idx_to_original_id_map]

def main():
    # 1. Load config
    config = load_config(os.path.join(os.path.dirname(__file__), '../config.yaml'))
    set_seed(config.get('random_seed', 42))
    data_dir = os.path.join(os.path.dirname(__file__), '../data/generated')
    os.makedirs(data_dir, exist_ok=True)
    # 2. Generate dataset
    generate_categories(
        num_categories=config['dataset']['num_categories'],
        output_path=os.path.join(data_dir, 'categories.csv'),
        base_category_definitions=config['dataset']['base_category_definitions'],
        random_seed=42
    )
    generate_professionals(
        num_professionals=config['dataset']['num_professionals'],
        avg_categories=config['dataset']['avg_categories_per_professional'],
        categories_path=os.path.join(data_dir, 'categories.csv'),
        output_dir=data_dir,
        random_seed=42,
        min_categories=config['dataset']['min_categories_per_professional'],
        max_categories=config['dataset']['max_categories_per_professional']
    )
    generate_clients(
        num_clients=config['dataset']['num_clients'],
        output_dir=data_dir,
        random_seed=42
    )
    generate_jobs(
        num_jobs=config['dataset']['num_jobs'],
        avg_categories_per_job=config['dataset']['avg_categories_per_job'],
        clients_path=os.path.join(data_dir, 'clients.csv'),
        categories_path=os.path.join(data_dir, 'categories.csv'),
        output_dir=data_dir,
        base_category_definitions=config['dataset']['base_category_definitions'],
        random_seed=config.get('random_seed', 42),
        min_categories=config['dataset']['min_categories_per_job'],
        max_categories=config['dataset']['max_categories_per_job']
    )
    generate_interactions(
        professionals_path=os.path.join(data_dir, 'professionals.csv'),
        jobs_path=os.path.join(data_dir, 'jobs.csv'),
        professional_categories_path=os.path.join(data_dir, 'professional_selected_categories.csv'),
        job_categories_path=os.path.join(data_dir, 'job_required_categories.csv'),
        output_path=os.path.join(data_dir, 'interactions.csv'),
        num_interactions_per_professional=config['dataset']['num_interactions_per_professional'],
        random_seed=42,
        jaccard_prob_offset=config['dataset']['jaccard_prob_offset'],
        unrelated_prob=config['dataset']['unrelated_prob']
    )
    # 3. Load DataFrames
    categories = pd.read_csv(os.path.join(data_dir, 'categories.csv'))
    jobs = pd.read_csv(os.path.join(data_dir, 'jobs.csv'))
    professionals = pd.read_csv(os.path.join(data_dir, 'professionals.csv'))
    professional_selected_categories = pd.read_csv(os.path.join(data_dir, 'professional_selected_categories.csv'))
    clients = pd.read_csv(os.path.join(data_dir, 'clients.csv'))
    job_required_categories = pd.read_csv(os.path.join(data_dir, 'job_required_categories.csv'))
    interactions = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))
    # 4. BERT Feature Engineering
    # Job node features
    job_texts = (jobs['title'] + ' ' + jobs['description']).tolist()
    job_features = get_bert_embeddings(
        job_texts,
        model_name=config['bert']['model_name'],
        batch_size=config['bert']['batch_size'],
        pooling=config['bert']['pooling'],
        max_length=config['bert']['max_length']
    )
    # Category node features
    cat_texts = categories['category_description'].tolist()
    category_features = get_bert_embeddings(
        cat_texts,
        model_name=config['bert']['model_name'],
        batch_size=config['bert']['batch_size'],
        pooling=config['bert']['pooling'],
        max_length=config['bert']['max_length']
    )
    # Professional node features: mean of selected category embeddings
    cat_id_to_idx = {cid: idx for idx, cid in enumerate(categories['category_id'])}
    prof_id_to_idx = {pid: idx for idx, pid in enumerate(professionals['professional_id'])}
    job_id_map = {jid: idx for idx, jid in enumerate(jobs['job_id'])}
    # For consistency, also define prof_id_map as an alias for prof_id_to_idx
    prof_id_map = prof_id_to_idx
    prof_features = np.zeros((len(professionals), category_features.shape[1]), dtype=np.float32)
    for i, prof in professionals.iterrows():
        prof_id = prof['professional_id']
        cat_ids = professional_selected_categories[professional_selected_categories['professional_id'] == prof_id]['category_id'].tolist()
        if cat_ids:
            cat_embs = [category_features[cat_id_to_idx[cid]] for cid in cat_ids if cid in cat_id_to_idx]
            if cat_embs:
                prof_features[i] = np.mean(cat_embs, axis=0)
            else:
                prof_features[i] = np.zeros(category_features.shape[1])
        else:
            prof_features[i] = np.zeros(category_features.shape[1])
    # 5. Graph Construction
    data_dfs = {
        'categories': categories,
        'jobs': jobs,
        'professionals': professionals,
        'professional_selected_categories': professional_selected_categories,
        'clients': clients,
        'job_required_categories': job_required_categories,
        'interactions': interactions
    }
    hetero_data = build_hetero_graph(data_dfs, prof_features, job_features, category_features)
    # 6. Data Splitting (RandomLinkSplit)
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
        edge_types=[('Professional', 'interacted_with', 'Job')],
        rev_edge_types=[('Job', 'interacted_with_rev', 'Professional')]
    )
    train_data, val_data, test_data = transform(hetero_data)
    # 7. Build all_positive_interactions_map from train_data
    train_edge_index = train_data['Professional', 'interacted_with', 'Job'].edge_index
    all_positive_interactions_map = defaultdict(set)
    for src, dst in zip(train_edge_index[0].tolist(), train_edge_index[1].tolist()):
        all_positive_interactions_map[src].add(dst)
    # 8. Model Instantiation
    metadata = train_data.metadata()
    model = HetGCNRecommender(
        metadata=metadata,
        input_feature_dims=config['model']['input_feature_dims'],
        hidden_channels=config['model']['hidden_channels'],
        out_channels=config['model']['out_channels'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    loss_fn = torch.nn.BCELoss()
    # 9. Training Loop Call
    train_edges = train_data['Professional', 'interacted_with', 'Job'].edge_index
    val_edges = val_data['Professional', 'interacted_with', 'Job'].edge_index
    model, history = train_model(
        model,
        train_data,
        train_edges,
        val_edges,
        optimizer,
        loss_fn,
        sample_negative_edges,
        epochs=config['training']['epochs'],
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_neg_per_pos=config['training']['num_neg_per_pos'],
        all_positive_interactions_map=all_positive_interactions_map
    )
    # 10. Evaluation
    print("\n--- Evaluation on Test Set ---")
    k_values = config['evaluation']['k_values']
    num_neg_eval = config['evaluation']['num_negative_samples_for_eval']
    test_edge_index = test_data['Professional', 'interacted_with', 'Job'].edge_index
    all_jobs = set(range(test_data['Job'].num_nodes))
    # Build a map of all positives (train+val+test) for each professional
    all_pos_map = defaultdict(set)
    for edge_index in [train_data['Professional', 'interacted_with', 'Job'].edge_index,
                      val_data['Professional', 'interacted_with', 'Job'].edge_index,
                      test_edge_index]:
        for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            all_pos_map[src].add(dst)
    # For each professional in test set, collect their test positives
    prof_to_test_jobs = defaultdict(list)
    for src, dst in zip(test_edge_index[0].tolist(), test_edge_index[1].tolist()):
        prof_to_test_jobs[src].append(dst)
    # Model in eval mode
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_data = test_data.to(device)
    metrics_per_prof = {k: [] for k in k_values}
    aucs = []
    maes = []
    rmses = []
    collected_y_trues = []
    collected_y_scores = []
    for prof, true_jobs in prof_to_test_jobs.items():
        # Sample negatives for this professional
        negatives = list(all_jobs - all_pos_map[prof])
        if len(negatives) > num_neg_eval:
            negatives = np.random.choice(negatives, num_neg_eval, replace=False)
        # Build edge_label_index for all (prof, true_job) and (prof, neg_job)
        pos_src = torch.tensor([prof]*len(true_jobs), dtype=torch.long)
        pos_dst = torch.tensor(true_jobs, dtype=torch.long)
        neg_src = torch.tensor([prof]*len(negatives), dtype=torch.long)
        neg_dst = torch.tensor(negatives, dtype=torch.long)
        all_src = torch.cat([pos_src, neg_src]).to(device)
        all_dst = torch.cat([pos_dst, neg_dst]).to(device)
        edge_label_index = (all_src, all_dst)
        # Predict scores
        with torch.no_grad():
            scores = model(test_data, {('Professional', 'interacted_with', 'Job'): edge_label_index}).cpu().numpy()
        y_true = np.array([1]*len(true_jobs) + [0]*len(negatives))
        y_score = scores
        for k in k_values:
            metrics_per_prof[k].append({
                'precision': precision_at_k(y_true, y_score, k),
                'recall': recall_at_k(y_true, y_score, k),
                'ndcg': ndcg_at_k(y_true, y_score, k)
            })
        aucs.append(auc_roc(y_true, y_score))
        maes.append(mae(y_true, y_score))
        rmses.append(rmse(y_true, y_score))
        collected_y_trues.extend(y_true)
        collected_y_scores.extend(y_score)
    # Aggregate and print
    for k in k_values:
        prec = np.mean([m['precision'] for m in metrics_per_prof[k]])
        rec = np.mean([m['recall'] for m in metrics_per_prof[k]])
        ndcg = np.mean([m['ndcg'] for m in metrics_per_prof[k]])
        print(f"K={k}: Precision@K={prec:.4f}, Recall@K={rec:.4f}, NDCG@K={ndcg:.4f}")
    print(f"AUC-ROC: {np.nanmean(aucs):.4f}")
    print(f"MAE: {np.nanmean(maes):.4f}")
    print(f"RMSE: {np.nanmean(rmses):.4f}")
    # --- Save training history with experiment name ---
    import datetime
    results_dir = os.path.join(os.path.dirname(__file__), '../results')
    os.makedirs(results_dir, exist_ok=True)
    experiment_name = config.get('experiment_name', 'default_experiment')
    history_df = pd.DataFrame(history)
    history_csv_path = os.path.join(results_dir, f'training_history_{experiment_name}.csv')
    history_df.to_csv(history_csv_path, index=False)
    print(f"Training history saved to: {history_csv_path}")
    # --- Save evaluation metrics to all_experiment_results.csv ---
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = {
        'timestamp': timestamp,
        'experiment_name': experiment_name,
        'num_layers': config['model']['num_layers'],
        'hidden_channels': config['model']['hidden_channels'],
        'epochs_run': len(history.get('train_loss', [])),
        'final_train_loss': float(history['train_loss'][-1]) if history.get('train_loss') else None,
        'final_val_loss': float(history['val_loss'][-1]) if history.get('val_loss') else None,
        'AUC-ROC': float(np.nanmean(aucs)),
        'MAE': float(np.nanmean(maes)),
        'RMSE': float(np.nanmean(rmses)),
    }
    for k in k_values:
        row[f'Precision@{k}'] = float(np.mean([m['precision'] for m in metrics_per_prof[k]]))
        row[f'Recall@{k}'] = float(np.mean([m['recall'] for m in metrics_per_prof[k]]))
        row[f'NDCG@{k}'] = float(np.mean([m['ndcg'] for m in metrics_per_prof[k]]))
    all_results_csv = os.path.join(results_dir, 'all_experiment_results.csv')
    import csv
    file_exists = os.path.isfile(all_results_csv)
    with open(all_results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"Experiment results appended to: {all_results_csv}")
    # 11. Generate Top-N Recommendations for a Sample Professional
    print("\n--- Top-N Recommendations for a Sample Professional ---")
    # job_id_map and prof_id_map are already defined in the node feature section
    original_job_ids = jobs['job_id'].tolist()
    job_node_indices_for_map = [job_id_map[jid] for jid in original_job_ids if jid in job_id_map]
    job_idx_to_original_id_map = {idx: jid for idx, jid in zip(job_node_indices_for_map, [jid for jid in original_job_ids if jid in job_id_map])}
    # Pick a sample professional
    professional_id_original = professionals['professional_id'].iloc[0]
    sample_prof_node_idx = prof_id_map.get(professional_id_original)
    all_job_node_indices = torch.arange(train_data['Job'].num_nodes)
    top_n_recs = generate_top_n_recommendations(
        model,
        train_data,
        sample_prof_node_idx,
        all_job_node_indices,
        N=5,
        all_positive_interactions_for_professional=all_pos_map.get(sample_prof_node_idx, set()),
        job_idx_to_original_id_map=job_idx_to_original_id_map
    )
    print(f"Top 5 Recommendations for Professional {professional_id_original}:")
    for original_job_id, score in top_n_recs:
        job_title = jobs[jobs['job_id'] == original_job_id]['title'].values[0]
        print(f"Job ID: {original_job_id}, Title: {job_title}, Score: {score:.4f}")

if __name__ == "__main__":
    main()
