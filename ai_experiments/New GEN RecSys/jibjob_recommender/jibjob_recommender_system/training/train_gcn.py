"""
Script for training the GCN/HeteroGCN recommendation model.
"""

import os
import sys
import argparse
import logging
import torch
import json
from typing import Dict, List, Tuple, Any, Optional

# Add the parent directory to the path to import the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jibjob_recommender_system.config.config_loader import ConfigLoader
from jibjob_recommender_system.utils.logging_config import setup_logging
from jibjob_recommender_system.data_handling.data_loader import DataLoader
from jibjob_recommender_system.feature_engineering.feature_orchestrator import FeatureOrchestrator
from jibjob_recommender_system.graph_construction.graph_builder import GraphBuilder
from jibjob_recommender_system.models.gcn_recommender import GCNRecommender, HeteroGCNRecommender
from jibjob_recommender_system.training.model_trainer import ModelTrainer

def setup_argparser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Train GCN/HeteroGCN Recommendation Model")
        
    # General options
    parser.add_argument('--config', type=str, default='../config/settings.yaml',
                      help='Path to the configuration file')
    parser.add_argument('--output-dir', type=str, default='../saved_models',
                      help='Directory to save trained models')
    parser.add_argument('--data-dir', type=str, default='../sample_data',
                      help='Directory containing input data files')
    parser.add_argument('--features-dir', type=str, default=None,
                      help='Directory containing pre-computed features')
                      
    # Model options
    parser.add_argument('--model-type', type=str, choices=['gcn', 'heterogcn'], default='heterogcn',
                      help='Type of GNN model to train')
    parser.add_argument('--embedding-dim', type=int, default=128,
                      help='Embedding dimension for model')
    parser.add_argument('--hidden-dim', type=int, default=64,
                      help='Hidden dimension for model')
    parser.add_argument('--num-layers', type=int, default=2,
                      help='Number of GNN layers')
                      
    # Training options
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                      help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                      help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.5,
                      help='Dropout rate')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU for training if available')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
                      
    # Logging options
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('--log-file', type=str, default=None,
                      help='Path to log file')
                      
    return parser

def update_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Update configuration with command-line arguments.
    
    Args:
        config: Original configuration dictionary.
        args: Command-line arguments.
        
    Returns:
        Dict[str, Any]: Updated configuration dictionary.
    """
    # Make a deep copy to avoid modifying the original
    updated_config = config.copy()
    
    # Update model configuration
    if 'model' not in updated_config:
        updated_config['model'] = {}
        
    updated_config['model']['embedding_dim'] = args.embedding_dim
    updated_config['model']['hidden_dim'] = args.hidden_dim
    updated_config['model']['num_layers'] = args.num_layers
    updated_config['model']['dropout'] = args.dropout
    
    # Update training configuration
    if 'training' not in updated_config:
        updated_config['training'] = {}
        
    updated_config['training']['epochs'] = args.epochs
    updated_config['training']['batch_size'] = args.batch_size
    updated_config['training']['learning_rate'] = args.learning_rate
    updated_config['training']['weight_decay'] = args.weight_decay
    
    # Update graph configuration based on model type
    if 'graph' not in updated_config:
        updated_config['graph'] = {}
        
    updated_config['graph']['use_heterogeneous'] = (args.model_type == 'heterogcn')
    
    # Set device based on GPU availability
    if args.use_gpu and torch.cuda.is_available():
        updated_config['device'] = 'cuda'
    else:
        updated_config['device'] = 'cpu'
        
    # Set random seed
    updated_config['random_seed'] = args.random_seed
    
    return updated_config

def load_or_build_graph(
    graph_builder: GraphBuilder,
    data_dict: Dict[str, Any],
    features_dict: Dict[str, Any],
    output_dir: str,
    model_name: str
) -> Any:
    """
    Load existing graph or build a new one.
    
    Args:
        graph_builder: GraphBuilder instance.
        data_dict: Dictionary of loaded data.
        features_dict: Dictionary of engineered features.
        output_dir: Directory to save outputs.
        model_name: Name of the model being trained.
        
    Returns:
        Any: Graph object.
    """
    graph_path = os.path.join(output_dir, f"{model_name}_graph.pt")
    
    # Check if graph exists
    if os.path.exists(graph_path):
        logging.info(f"Loading existing graph from {graph_path}")
        graph = torch.load(graph_path)
        return graph
    
    # Build a new graph
    logging.info("Building graph...")
    if graph_builder.use_heterogeneous:
        graph = graph_builder.build_heterogeneous_graph(data_dict, features_dict)
    else:
        graph = graph_builder.build_homogeneous_graph(data_dict, features_dict)
    
    # Save the graph
    os.makedirs(os.path.dirname(graph_path), exist_ok=True)
    torch.save(graph, graph_path)
    logging.info(f"Graph saved to {graph_path}")
    
    return graph

def save_training_config(config: Dict[str, Any], output_dir: str, model_name: str) -> None:
    """
    Save training configuration to a file.
    
    Args:
        config: Configuration dictionary.
        output_dir: Directory to save outputs.
        model_name: Name of the model being trained.
    """
    config_path = os.path.join(output_dir, f"{model_name}_config.json")
    
    # Filter out only the relevant parts of the config
    training_config = {
        'model': config.get('model', {}),
        'training': config.get('training', {}),
        'graph': config.get('graph', {}),
        'device': config.get('device', 'cpu'),
        'random_seed': config.get('random_seed', 42)
    }
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
        
    logging.info(f"Training configuration saved to {config_path}")

def setup_trainer(config: Dict[str, Any], model_type: str) -> ModelTrainer:
    """
    Set up model trainer.
    
    Args:
        config: Configuration dictionary.
        model_type: Type of GNN model to train.
        
    Returns:
        ModelTrainer: Configured model trainer.
    """
    # Create trainer instance
    trainer = ModelTrainer(config)
    
    # Set model type
    if model_type == 'heterogcn':
        trainer.model_class = HeteroGCNRecommender
    else:
        trainer.model_class = GCNRecommender
        
    return trainer

def main() -> None:
    """
    Main function for training the GCN/HeteroGCN model.
    """
    # Set up argument parser
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_file = args.log_file or os.path.join(args.output_dir, 'training_log.txt')
    setup_logging(log_level=log_level, log_file=log_file)
    
    logging.info(f"Starting training of {args.model_type} model")
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    
    if config is None:
        logging.error(f"Failed to load configuration from {args.config}")
        sys.exit(1)
        
    # Update configuration with command-line arguments
    config = update_config_with_args(config, args)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate model name
    model_name = f"{args.model_type}_dim{args.embedding_dim}_layers{args.num_layers}"
    
    # Save training configuration
    save_training_config(config, args.output_dir, model_name)
    
    # Load data
    data_loader = DataLoader(config)
    data_dict = data_loader.load_data(args.data_dir)
    
    if data_dict is None or not data_dict:
        logging.error(f"Failed to load data from {args.data_dir}")
        sys.exit(1)
        
    # Load or compute features
    feature_orchestrator = FeatureOrchestrator(config)
    
    if args.features_dir and os.path.exists(args.features_dir):
        logging.info(f"Loading features from {args.features_dir}")
        feature_orchestrator.features_dir = args.features_dir
        features_dict = feature_orchestrator.load_all_features()
    else:
        logging.info("Computing features...")
        features_dict = feature_orchestrator.process_and_save_all_features(data_dict)
        
    if features_dict is None or not features_dict:
        logging.error("Failed to load or compute features")
        sys.exit(1)
        
    # Build graph
    graph_builder = GraphBuilder(config)
    graph = load_or_build_graph(graph_builder, data_dict, features_dict, args.output_dir, model_name)
    
    # Set up trainer
    trainer = setup_trainer(config, args.model_type)
    
    # Train model
    logging.info("Training model...")
    model, train_stats = trainer.train(graph)
    
    # Save model
    model_path = os.path.join(args.output_dir, f"{model_name}.pt")
    trainer.save_model(model, model_path)
    
    # Save training stats
    stats_path = os.path.join(args.output_dir, f"{model_name}_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(train_stats, f, indent=2)
        
    logging.info(f"Model training complete. Model saved to {model_path}")
    logging.info(f"Final loss: {train_stats['loss'][-1]}")
    
    # Print key model parameters
    logging.info("Model parameters:")
    logging.info(f"  Model type: {args.model_type}")
    logging.info(f"  Embedding dimension: {args.embedding_dim}")
    logging.info(f"  Hidden dimension: {args.hidden_dim}")
    logging.info(f"  Number of layers: {args.num_layers}")

if __name__ == "__main__":
    main()
