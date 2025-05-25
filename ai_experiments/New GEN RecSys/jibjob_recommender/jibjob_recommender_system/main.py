"""
Main script for JibJob recommendation system.
This script orchestrates the entire process of:
1. Loading and preprocessing data
2. Engineering features
3. Building graphs
4. Training recommendation models
5. Evaluating model performance
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, Optional

# Import JibJob recommender system components
from jibjob_recommender_system.config.config_loader import ConfigLoader
from jibjob_recommender_system.utils.logging_config import setup_logging
from jibjob_recommender_system.data_handling.data_loader import DataLoader
from jibjob_recommender_system.data_handling.preprocessor import TextPreprocessor
from jibjob_recommender_system.feature_engineering.feature_orchestrator import FeatureOrchestrator
from jibjob_recommender_system.graph_construction.graph_builder import GraphBuilder
from jibjob_recommender_system.models.gcn_recommender import GCNRecommender, HeteroGCNRecommender
from jibjob_recommender_system.training.trainer import ModelTrainer
from jibjob_recommender_system.evaluation.evaluation import RecommendationEvaluator

def setup_argparser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="JibJob Recommendation System - Complete Pipeline")
        
    # General options
    parser.add_argument('--config', type=str, default='./config/settings.yaml',
                      help='Path to the configuration file')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                      help='Directory to save outputs')
    parser.add_argument('--data-dir', type=str, default='./sample_data',
                      help='Directory containing input data files')
                      
    # Pipeline control
    parser.add_argument('--skip-preprocessing', action='store_true',
                      help='Skip data preprocessing step')
    parser.add_argument('--skip-feature-engineering', action='store_true',
                      help='Skip feature engineering step')
    parser.add_argument('--skip-graph-building', action='store_true',
                      help='Skip graph building step')
    parser.add_argument('--skip-training', action='store_true',
                      help='Skip model training step')
    parser.add_argument('--skip-evaluation', action='store_true',
                      help='Skip model evaluation step')
    parser.add_argument('--load-features', action='store_true',
                      help='Load pre-computed features instead of computing them')
    parser.add_argument('--load-model', type=str,
                      help='Load a pre-trained model instead of training')
                      
    # Training options
    parser.add_argument('--epochs', type=int,
                      help='Number of training epochs (overrides config)')
    parser.add_argument('--batch-size', type=int,
                      help='Training batch size (overrides config)')
    parser.add_argument('--learning-rate', type=float,
                      help='Learning rate (overrides config)')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU for training if available')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
                      
    # Evaluation options
    parser.add_argument('--eval-k', type=int, nargs='+', default=[1, 5, 10, 20],
                      help='K values for evaluation metrics')
                      
    # Logging options
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--log-file', type=str,
                      help='Path to log file')
                      
    return parser

def override_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Override configuration with command-line arguments.
    
    Args:
        config: Original configuration dictionary.
        args: Command-line arguments.
        
    Returns:
        Dict[str, Any]: Updated configuration dictionary.
    """
    # Make a copy of the config to avoid modifying the original
    updated_config = config.copy()
    
    # Training overrides
    if args.epochs is not None:
        updated_config['training']['epochs'] = args.epochs
        
    if args.batch_size is not None:
        updated_config['training']['batch_size'] = args.batch_size
        
    if args.learning_rate is not None:
        updated_config['training']['learning_rate'] = args.learning_rate
        
    # Evaluation overrides
    if args.eval_k is not None:
        if 'evaluation' not in updated_config:
            updated_config['evaluation'] = {}
        updated_config['evaluation']['k_values'] = args.eval_k
        
    # Device override
    if args.use_gpu:
        updated_config['device'] = 'cuda'
    else:
        updated_config['device'] = 'cpu'
        
    # Random seed
    updated_config['random_seed'] = args.random_seed
    
    return updated_config

def create_output_dirs(base_dir: str) -> Dict[str, str]:
    """
    Create output directories.
    
    Args:
        base_dir: Base output directory.
        
    Returns:
        Dict[str, str]: Dictionary of directory paths.
    """
    dirs = {
        'base': base_dir,
        'preprocessed': os.path.join(base_dir, 'preprocessed'),
        'features': os.path.join(base_dir, 'features'),
        'models': os.path.join(base_dir, 'models'),
        'evaluation': os.path.join(base_dir, 'evaluation'),
        'logs': os.path.join(base_dir, 'logs')
    }
    
    # Create directories
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs

def load_or_preprocess_data(
    data_loader: DataLoader,
    text_preprocessor: TextPreprocessor,
    data_dir: str,
    output_dir: str,
    skip_preprocessing: bool
) -> Dict[str, Any]:
    """
    Load and preprocess data.
    
    Args:
        data_loader: DataLoader instance.
        text_preprocessor: TextPreprocessor instance.
        data_dir: Directory containing input data files.
        output_dir: Directory to save preprocessed data.
        skip_preprocessing: Whether to skip preprocessing.
        
    Returns:
        Dict[str, Any]: Dictionary of loaded and processed data.
    """
    logging.info("Loading data...")
    
    # Load raw data
    data_dict = data_loader.load_data(data_dir)
    
    if data_dict is None or not data_dict:
        logging.error("Failed to load data")
        return None
    
    logging.info(f"Loaded {len(data_dict)} data files")
    
    # Check if preprocessed data exists and skip_preprocessing is True
    preprocessed_path = os.path.join(output_dir, 'preprocessed_data.json')
    if skip_preprocessing and os.path.exists(preprocessed_path):
        logging.info("Loading preprocessed data...")
        with open(preprocessed_path, 'r') as f:
            preprocessed_data = json.load(f)
        return preprocessed_data
    
    # Preprocess data
    logging.info("Preprocessing data...")
    
    # Preprocess text fields
    if 'users' in data_dict:
        logging.info("Preprocessing user data...")
        data_dict['users']['profile_bio'] = data_dict['users']['profile_bio'].apply(
            lambda x: text_preprocessor.preprocess(x) if isinstance(x, str) else '')
    
    if 'jobs' in data_dict:
        logging.info("Preprocessing job data...")
        data_dict['jobs']['title'] = data_dict['jobs']['title'].apply(
            lambda x: text_preprocessor.preprocess(x) if isinstance(x, str) else '')
        data_dict['jobs']['description'] = data_dict['jobs']['description'].apply(
            lambda x: text_preprocessor.preprocess(x) if isinstance(x, str) else '')
    
    if 'categories' in data_dict:
        logging.info("Preprocessing category data...")
        data_dict['categories']['name'] = data_dict['categories']['name'].apply(
            lambda x: text_preprocessor.preprocess(x) if isinstance(x, str) else '')
        data_dict['categories']['description'] = data_dict['categories']['description'].apply(
            lambda x: text_preprocessor.preprocess(x) if isinstance(x, str) else '')
    
    # Save preprocessed data
    with open(preprocessed_path, 'w') as f:
        json.dump(data_dict, f)
    
    logging.info("Preprocessing complete")
    return data_dict

def engineer_features(
    feature_orchestrator: FeatureOrchestrator,
    data_dict: Dict[str, Any],
    output_dir: str,
    load_features: bool
) -> Dict[str, Any]:
    """
    Engineer features for the recommendation model.
    
    Args:
        feature_orchestrator: FeatureOrchestrator instance.
        data_dict: Dictionary of loaded and processed data.
        output_dir: Directory to save engineered features.
        load_features: Whether to load pre-computed features.
        
    Returns:
        Dict[str, Any]: Dictionary of engineered features.
    """
    features_dir = os.path.join(output_dir, 'features')
    
    # Check if features should be loaded instead of computed
    if load_features:
        logging.info("Loading pre-computed features...")
        features = feature_orchestrator.load_all_features()
        if features:
            logging.info("Features loaded successfully")
            return features
        logging.warning("Failed to load pre-computed features, will compute them")
    
    # Engineer features
    logging.info("Engineering features...")
    features = feature_orchestrator.process_and_save_all_features(data_dict)
    
    logging.info("Feature engineering complete")
    return features

def build_graph(
    graph_builder: GraphBuilder,
    data_dict: Dict[str, Any],
    features_dict: Dict[str, Any],
    output_dir: str
) -> Any:
    """
    Build graph for GCN/HeteroGCN model.
    
    Args:
        graph_builder: GraphBuilder instance.
        data_dict: Dictionary of loaded and processed data.
        features_dict: Dictionary of engineered features.
        output_dir: Directory to save the graph.
        
    Returns:
        Any: Built graph object.
    """
    graph_path = os.path.join(output_dir, 'graph.pt')
    
    logging.info("Building graph...")
    
    # Build graph based on configuration
    if graph_builder.use_heterogeneous:
        logging.info("Building heterogeneous graph...")
        graph = graph_builder.build_heterogeneous_graph(data_dict, features_dict)
    else:
        logging.info("Building homogeneous graph...")
        graph = graph_builder.build_homogeneous_graph(data_dict, features_dict)
    
    # Save graph
    import torch
    torch.save(graph, graph_path)
    
    logging.info("Graph building complete")
    return graph

def train_model(
    trainer: ModelTrainer,
    graph: Any,
    config: Dict[str, Any],
    output_dir: str,
    load_model_path: Optional[str]
) -> Any:
    """
    Train the recommendation model.
    
    Args:
        trainer: ModelTrainer instance.
        graph: Graph object.
        config: Configuration dictionary.
        output_dir: Directory to save the model.
        load_model_path: Path to a pre-trained model to load.
        
    Returns:
        Any: Trained model.
    """
    models_dir = os.path.join(output_dir, 'models')
    
    # Check if a pre-trained model should be loaded
    if load_model_path and os.path.exists(load_model_path):
        logging.info(f"Loading pre-trained model from {load_model_path}...")
        import torch
        
        # Determine model type from config
        use_heterogeneous = config.get('graph', {}).get('use_heterogeneous', True)
        
        if use_heterogeneous:
            model = HeteroGCNRecommender(config)
        else:
            model = GCNRecommender(config)
            
        # Load model weights
        checkpoint = torch.load(load_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logging.info("Model loaded successfully")
        return model
    
    # Train model
    logging.info("Training model...")
    model, train_stats = trainer.train(graph)
    
    # Save model and training stats
    model_path = os.path.join(models_dir, 'trained_model.pt')
    stats_path = os.path.join(models_dir, 'training_stats.json')
    
    trainer.save_model(model, model_path)
    
    with open(stats_path, 'w') as f:
        json.dump(train_stats, f, indent=2)
    
    logging.info("Model training complete")
    return model

def evaluate_model(
    evaluator: RecommendationEvaluator,
    model: Any,
    graph: Any,
    data_dict: Dict[str, Any],
    features_dict: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Evaluate the recommendation model.
    
    Args:
        evaluator: RecommendationEvaluator instance.
        model: Trained model.
        graph: Graph object.
        data_dict: Dictionary of loaded and processed data.
        features_dict: Dictionary of engineered features.
        output_dir: Directory to save evaluation results.
        
    Returns:
        Dict[str, Any]: Evaluation results.
    """
    eval_dir = os.path.join(output_dir, 'evaluation')
    
    logging.info("Evaluating model...")
    
    # Generate recommendations for test users
    test_user_ids = []
    if 'job_applications' in data_dict and 'user_id' in data_dict['job_applications'].columns:
        # Use users from job_applications as test users
        test_user_ids = data_dict['job_applications']['user_id'].unique().tolist()
    else:
        # Fallback to all professional users
        test_user_ids = data_dict['users'][data_dict['users']['user_type'] == 'professional']['user_id'].tolist()
    
    # Limit number of test users for efficiency if needed
    max_test_users = 100
    if len(test_user_ids) > max_test_users:
        import random
        random.shuffle(test_user_ids)
        test_user_ids = test_user_ids[:max_test_users]
    
    logging.info(f"Generating recommendations for {len(test_user_ids)} test users...")
    
    # Generate recommendations
    recommendations = {}
    for user_id in test_user_ids:
        # Skip users not in the graph
        if user_id not in graph.professional_id_to_idx:
            continue
            
        # Get user's node index
        user_idx = graph.professional_id_to_idx[user_id]
        
        # Get predictions
        import torch
        with torch.no_grad():
            predictions = model.predict_for_user(user_idx)
            
        # Convert tensor to numpy
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            
        # Map job indices to IDs and sort by score
        job_scores = {}
        for job_idx, score in enumerate(predictions):
            if job_idx in graph.job_idx_to_id:
                job_id = graph.job_idx_to_id[job_idx]
                job_scores[job_id] = float(score)
                
        # Sort by score and get job IDs
        sorted_jobs = sorted(job_scores.items(), key=lambda x: x[1], reverse=True)
        rec_job_ids = [job_id for job_id, _ in sorted_jobs]
        
        # Add to recommendations
        recommendations[user_id] = rec_job_ids
    
    # Create ground truth from job applications if available
    ground_truth = {}
    if 'job_applications' in data_dict:
        for user_id in test_user_ids:
            user_applications = data_dict['job_applications'][
                data_dict['job_applications']['user_id'] == user_id
            ]
            ground_truth[user_id] = user_applications['job_id'].tolist()
    
    # Evaluate recommendations
    evaluation_results = {}
    if ground_truth:
        logging.info("Evaluating against ground truth...")
        evaluation_results = evaluator.evaluate_all_metrics(
            recommendations,
            ground_truth,
            job_categories={
                row['job_id']: row['required_category_id']
                for _, row in data_dict['jobs'].iterrows()
                if 'required_category_id' in row
            } if 'jobs' in data_dict else None,
            user_profiles={
                row['user_id']: data_dict['professional_categories'][
                    data_dict['professional_categories']['user_id'] == row['user_id']
                ]['category_id'].tolist()
                for _, row in data_dict['users'][
                    data_dict['users']['user_type'] == 'professional'
                ].iterrows()
            } if 'professional_categories' in data_dict else None,
            all_items=data_dict['jobs']['job_id'].tolist() if 'jobs' in data_dict else None
        )
    
    # Save recommendations and evaluation results
    recommendations_path = os.path.join(eval_dir, 'recommendations.json')
    evaluation_path = os.path.join(eval_dir, 'evaluation_results.json')
    
    with open(recommendations_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    if evaluation_results:
        with open(evaluation_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
    
    logging.info("Evaluation complete")
    return evaluation_results

def main() -> None:
    """
    Main function to run the complete pipeline.
    """
    # Set up argument parser
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file or os.path.join(args.output_dir, 'logs', 'pipeline.log')
    setup_logging(log_level=log_level, log_file=log_file)
    
    logging.info("Starting JibJob recommendation system pipeline")
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    
    if config is None:
        logging.error("Failed to load configuration")
        sys.exit(1)
    
    # Override config with command-line arguments
    config = override_config_with_args(config, args)
    
    # Create output directories
    output_dirs = create_output_dirs(args.output_dir)
    
    # Initialize components
    data_loader = DataLoader(config)
    text_preprocessor = TextPreprocessor(config)
    feature_orchestrator = FeatureOrchestrator(config)
    graph_builder = GraphBuilder(config)
    trainer = ModelTrainer(config)
    evaluator = RecommendationEvaluator(config)
    
    # Step 1: Load and preprocess data
    data_dict = None
    if not args.skip_preprocessing:
        data_dict = load_or_preprocess_data(
            data_loader,
            text_preprocessor,
            args.data_dir,
            output_dirs['preprocessed'],
            args.skip_preprocessing
        )
        
        if data_dict is None:
            logging.error("Data loading/preprocessing failed")
            sys.exit(1)
    else:
        # Load previously preprocessed data
        preprocessed_path = os.path.join(output_dirs['preprocessed'], 'preprocessed_data.json')
        if os.path.exists(preprocessed_path):
            with open(preprocessed_path, 'r') as f:
                data_dict = json.load(f)
        else:
            logging.error("Preprocessed data not found, cannot skip preprocessing")
            sys.exit(1)
    
    # Step 2: Engineer features
    features_dict = None
    if not args.skip_feature_engineering:
        features_dict = engineer_features(
            feature_orchestrator,
            data_dict,
            output_dirs['features'],
            args.load_features
        )
        
        if features_dict is None:
            logging.error("Feature engineering failed")
            sys.exit(1)
    else:
        # Load previously engineered features
        features_dict = feature_orchestrator.load_all_features()
        if not features_dict:
            logging.error("Engineered features not found, cannot skip feature engineering")
            sys.exit(1)
    
    # Step 3: Build graph
    graph = None
    if not args.skip_graph_building:
        graph = build_graph(
            graph_builder,
            data_dict,
            features_dict,
            output_dirs['models']
        )
    else:
        # Load previously built graph
        graph_path = os.path.join(output_dirs['models'], 'graph.pt')
        if os.path.exists(graph_path):
            import torch
            graph = torch.load(graph_path)
        else:
            logging.error("Graph not found, cannot skip graph building")
            sys.exit(1)
    
    # Step 4: Train model
    model = None
    if not args.skip_training:
        model = train_model(
            trainer,
            graph,
            config,
            output_dirs['models'],
            args.load_model
        )
    else:
        # Load previously trained model
        model_path = args.load_model or os.path.join(output_dirs['models'], 'trained_model.pt')
        model = train_model(
            trainer,
            graph,
            config,
            output_dirs['models'],
            model_path
        )
    
    # Step 5: Evaluate model
    if not args.skip_evaluation:
        evaluation_results = evaluate_model(
            evaluator,
            model,
            graph,
            data_dict,
            features_dict,
            output_dirs['base']
        )
        
        # Log key evaluation metrics
        if evaluation_results:
            for metric, values in evaluation_results.items():
                if isinstance(values, dict) and 10 in values:  # Report @10 metrics
                    logging.info(f"{metric}@10: {values[10]:.4f}")
    
    logging.info("JibJob recommendation system pipeline complete")

if __name__ == "__main__":
    main()
