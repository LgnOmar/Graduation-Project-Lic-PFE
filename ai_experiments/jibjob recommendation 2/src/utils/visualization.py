"""
Visualization utilities for recommendation system analysis.
This module provides functions to visualize data, model performance, and recommendations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set(style="whitegrid")

def plot_recommendation_quality(
    actual_ratings: np.ndarray,
    predicted_ratings: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    title: str = 'Recommendation Quality',
    xlabel: str = 'Actual Ratings',
    ylabel: str = 'Predicted Ratings'
) -> plt.Figure:
    """
    Plot actual vs predicted ratings to visualize recommendation quality.
    
    Args:
        actual_ratings: Array of actual ratings
        predicted_ratings: Array of predicted ratings
        figsize: Figure size as (width, height)
        alpha: Transparency for scatter points
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of actual vs. predicted ratings
    scatter = ax.scatter(actual_ratings, predicted_ratings, alpha=alpha)
    
    # Add a perfect prediction line
    min_val = min(actual_ratings.min(), predicted_ratings.min())
    max_val = max(actual_ratings.max(), predicted_ratings.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate and display correlation
    correlation = np.corrcoef(actual_ratings, predicted_ratings)[0, 1]
    rmse = np.sqrt(np.mean((actual_ratings - predicted_ratings) ** 2))
    
    # Add text with metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = f'Correlation: {correlation:.3f}\nRMSE: {rmse:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_rating_distribution(
    ratings: Union[np.ndarray, List[float]],
    title: str = "Rating Distribution",
    save_path: Optional[str] = None,
    bins: int = 10,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot the distribution of ratings.
    
    Args:
        ratings: Array or list of rating values.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        bins: Number of histogram bins.
        figsize: Figure size as (width, height).
    """
    plt.figure(figsize=figsize)
    sns.histplot(ratings, bins=bins, kde=True)
    plt.title(title)
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_user_job_heatmap(
    interaction_matrix: np.ndarray,
    title: str = "User-Job Interactions",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    max_users: int = 50,
    max_jobs: int = 50
):
    """
    Plot a heatmap of user-job interactions.
    
    Args:
        interaction_matrix: Matrix where rows are users and columns are jobs.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
        max_users: Maximum number of users to include in the plot.
        max_jobs: Maximum number of jobs to include in the plot.
    """
    # Limit size for better visualization
    plot_matrix = interaction_matrix[:max_users, :max_jobs]
    
    plt.figure(figsize=figsize)
    sns.heatmap(plot_matrix, cmap="YlGnBu", cbar_kws={'label': 'Rating'})
    plt.title(title)
    plt.xlabel(f"Jobs (showing {min(max_jobs, plot_matrix.shape[1])} of {interaction_matrix.shape[1]})")
    plt.ylabel(f"Users (showing {min(max_users, plot_matrix.shape[0])} of {interaction_matrix.shape[0]})")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = None,
    title: str = "Training History",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary mapping metric names to lists of values.
        metrics: List of metrics to plot. If None, plot all metrics.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
    """
    # If metrics not specified, use all available metrics
    if metrics is None:
        metrics = list(history.keys())
    
    plt.figure(figsize=figsize)
    
    for metric in metrics:
        if metric in history:
            plt.plot(history[metric], label=metric)
    
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_embeddings_2d(
    embeddings: np.ndarray,
    labels: Optional[List[Any]] = None,
    title: str = "Embeddings Visualization",
    method: str = "tsne",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.7,
    max_points: int = 5000
):
    """
    Plot 2D visualization of embeddings using dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings.
        labels: Optional list of labels for coloring points.
        title: Plot title.
        method: Dimensionality reduction method ('tsne' or 'pca').
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
        alpha: Transparency of points.
        max_points: Maximum number of points to plot.
    """
    # Limit number of points for better visualization and performance
    if len(embeddings) > max_points:
        indices = np.random.choice(len(embeddings), max_points, replace=False)
        embeddings = embeddings[indices]
        if labels is not None:
            labels = [labels[i] for i in indices]
    
    # Convert torch tensor to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    # Apply dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    plt.figure(figsize=figsize)
    
    # Plot with or without labels
    if labels is not None:
        # Convert labels to categorical if they're not already
        unique_labels = np.unique(labels)
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        label_ids = np.array([label_to_id[label] for label in labels])
        
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=label_ids,
            cmap='tab20',
            alpha=alpha
        )
        
        # Add legend if not too many labels
        if len(unique_labels) <= 20:
            plt.legend(handles=scatter.legend_elements()[0], labels=unique_labels, 
                      title="Labels", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            alpha=alpha
        )
    
    plt.title(f"{title} ({method.upper()})")
    plt.xlabel(f"{method.upper()} Dimension 1")
    plt.ylabel(f"{method.upper()} Dimension 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_job_similarity_heatmap(
    job_similarities: np.ndarray,
    job_names: Optional[List[str]] = None,
    title: str = "Job Similarity Heatmap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
    max_jobs: int = 30
):
    """
    Plot a heatmap of job similarities.
    
    Args:
        job_similarities: Square matrix of job similarities.
        job_names: Optional list of job names.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
        max_jobs: Maximum number of jobs to include in the plot.
    """
    # Limit size for better visualization
    n_jobs = min(job_similarities.shape[0], max_jobs)
    plot_matrix = job_similarities[:n_jobs, :n_jobs]
    
    plt.figure(figsize=figsize)
    
    if job_names and len(job_names) >= n_jobs:
        short_names = [name[:20] + '...' if len(name) > 20 else name for name in job_names[:n_jobs]]
        sns.heatmap(plot_matrix, cmap="viridis", annot=False, 
                   xticklabels=short_names, yticklabels=short_names)
    else:
        sns.heatmap(plot_matrix, cmap="viridis", annot=False)
    
    plt.title(title)
    plt.xlabel("Jobs")
    plt.ylabel("Jobs")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_recommendation_metrics(
    metrics: Dict[str, float],
    title: str = "Recommendation Metrics",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8),
    sort_by_k: bool = True
):
    """
    Plot recommendation metrics.
    
    Args:
        metrics: Dictionary mapping metric names to values.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
        sort_by_k: Whether to sort metrics by k value.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)
    # Filtrer uniquement les métriques autorisées
    allowed = ["precision", "recall", "ndcg", "mae", "rmse"]
    filtered = {k: v for k, v in metrics.items() if any(a in k for a in allowed)}
    names = list(filtered.keys())
    values = list(filtered.values())
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel("Value")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5,
    title: str = "Recommendation Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot confusion matrix for binary recommendation prediction.
    
    Args:
        predictions: Array of predicted scores.
        ground_truth: Array of true labels.
        threshold: Threshold for converting scores to binary predictions.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # Convert scores to binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, binary_preds)
    
    # Plot
    plt.figure(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Recommended", "Recommended"])
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_top_jobs_per_user(
    recommendations: Dict[str, List[Tuple[str, float]]],
    top_n: int = 5,
    title: str = "Top Job Recommendations per User",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    max_users: int = 10
):
    """
    Plot top job recommendations for each user.
    
    Args:
        recommendations: Dictionary mapping user IDs to lists of (job_id, score) tuples.
        top_n: Number of top recommendations to show per user.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
        max_users: Maximum number of users to include in the plot.
    """
    # Limit number of users
    user_ids = list(recommendations.keys())[:max_users]
    
    # Create DataFrame for plotting
    data = []
    for user_id in user_ids:
        for i, (job_id, score) in enumerate(recommendations[user_id][:top_n]):
            data.append({
                'User': str(user_id),
                'Rank': i + 1,
                'Job': str(job_id),
                'Score': score
            })
    
    df = pd.DataFrame(data)
    
    plt.figure(figsize=figsize)
    chart = sns.barplot(data=df, x='User', y='Score', hue='Rank', palette='viridis')
    plt.title(title)
    plt.xlabel("User ID")
    plt.ylabel("Recommendation Score")
    plt.legend(title="Recommendation Rank")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add job IDs as annotations
    for i, user_id in enumerate(user_ids):
        for j, (job_id, score) in enumerate(recommendations[user_id][:top_n]):
            plt.text(
                i, score, f"Job: {job_id}", 
                ha='center', va='bottom', 
                rotation=90, fontsize=8
            )
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_recommendation_diversity(
    recommendations: Dict[str, List[Any]],
    title: str = "Recommendation Diversity Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
):
    """
    Plot analysis of recommendation diversity.
    
    Args:
        recommendations: Dictionary mapping user IDs to lists of recommended items.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
    """
    # Count frequency of each item in recommendations
    item_counts = {}
    for items in recommendations.values():
        for item in items:
            if isinstance(item, tuple):
                item = item[0]  # Extract item ID if (item_id, score) format
            item_counts[item] = item_counts.get(item, 0) + 1
    
    # Convert to DataFrame for plotting
    df = pd.DataFrame({
        'Item': list(item_counts.keys()),
        'Count': list(item_counts.values())
    }).sort_values('Count', ascending=False)
    
    plt.figure(figsize=figsize)
    
    # Plot frequency distribution
    sns.histplot(df['Count'], kde=True, bins=30)
    plt.title(title)
    plt.xlabel("Number of Times Recommended")
    plt.ylabel("Number of Items")
    plt.grid(True, alpha=0.3)
    
    # Add statistics to the plot
    stats = {
        'Total Items': len(item_counts),
        'Mean Frequency': np.mean(df['Count']),
        'Median Frequency': np.median(df['Count']),
        'Max Frequency': np.max(df['Count']),
        'Min Frequency': np.min(df['Count'])
    }
    
    stat_text = "\n".join(f"{name}: {value:.2f}" if isinstance(value, float) else f"{name}: {value}" 
                          for name, value in stats.items())
    plt.figtext(0.02, 0.02, stat_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_interactive_recommendation_graph(
    user_job_graph: Dict[str, List[Dict[str, Any]]],
    output_path: str,
    title: str = "Interactive Recommendation Graph",
    max_nodes: int = 100
):
    """
    Generate an interactive HTML visualization of the recommendation graph.
    
    Args:
        user_job_graph: Dictionary with graph data in a format like:
            {
                'nodes': [{'id': 'u1', 'label': 'User 1', 'type': 'user'}, ...],
                'edges': [{'from': 'u1', 'to': 'j1', 'value': 0.8}, ...]
            }
        output_path: Path to save the HTML file.
        title: Title for the visualization.
        max_nodes: Maximum number of nodes to include.
    """
    try:
        import networkx as nx
        from pyvis.network import Network
    except ImportError:
        logger.warning("pyvis or networkx not installed. Cannot create interactive graph.")
        logger.info("Install with: pip install pyvis networkx")
        return
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Limit the number of nodes if needed
    nodes = user_job_graph['nodes'][:max_nodes]
    node_ids = {node['id'] for node in nodes}
    
    # Add nodes
    for node in nodes:
        G.add_node(
            node['id'],
            label=node['label'],
            title=node['label'],
            group=node['type'],
            size=30 if node['type'] == 'user' else 20
        )
    
    # Add edges (only for nodes that are included)
    for edge in user_job_graph['edges']:
        if edge['from'] in node_ids and edge['to'] in node_ids:
            G.add_edge(
                edge['from'],
                edge['to'],
                value=edge['value'],
                title=f"Score: {edge['value']:.2f}"
            )
    
    # Create a pyvis network
    net = Network(notebook=False, height='800px', width='100%', bgcolor="#ffffff", font_color="black")
    net.from_nx(G)
    
    # Set options
    net.set_options("""
    {
        "nodes": {
            "shape": "dot",
            "scaling": {
                "min": 10,
                "max": 30,
                "label": {
                    "enabled": true
                }
            }
        },
        "edges": {
            "color": {
                "inherit": true
            },
            "smooth": {
                "enabled": false
            }
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {
                "enabled": true,
                "iterations": 1000
            }
        },
        "interaction": {
            "tooltipDelay": 200,
            "hideEdgesOnDrag": true
        }
    }
    """)
    
    # Add title
    net.heading = title
    
    # Save the visualization
    net.save_graph(output_path)
    logger.info(f"Interactive graph saved to: {output_path}")


def plot_sentiment_vs_ratings(
    sentiment_scores: List[float],
    ratings: List[float],
    title: str = "Sentiment Scores vs. Explicit Ratings",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    add_regression: bool = True
):
    """
    Plot sentiment scores against explicit ratings.
    
    Args:
        sentiment_scores: List of sentiment scores.
        ratings: List of explicit ratings.
        title: Plot title.
        save_path: Path to save the plot. If None, the plot is displayed.
        figsize: Figure size as (width, height).
        add_regression: Whether to add a regression line.
    """
    plt.figure(figsize=figsize)
    
    # Create scatter plot with hexbin for density
    plt.hexbin(ratings, sentiment_scores, gridsize=20, cmap='viridis', mincnt=1)
    plt.colorbar(label='Count')
    
    # Add regression line if requested
    if add_regression:
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        X = np.array(ratings).reshape(-1, 1)
        y = np.array(sentiment_scores)
        
        model = LinearRegression().fit(X, y)
        x_range = np.linspace(min(ratings), max(ratings), 100).reshape(-1, 1)
        y_pred = model.predict(x_range)
        
        plt.plot(x_range, y_pred, color='red', linewidth=2)
        
        # Add correlation coefficient
        corr = np.corrcoef(ratings, sentiment_scores)[0, 1]
        r2 = model.score(X, y)
        plt.text(
            0.05, 0.95, f"Correlation: {corr:.3f}\nR²: {r2:.3f}", 
            transform=plt.gca().transAxes,
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )
    
    plt.title(title)
    plt.xlabel("Explicit Rating")
    plt.ylabel("Sentiment Score")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_recommendation_dashboard(
    metrics: Dict[str, float],
    history: Dict[str, List[float]],
    recommendations: Dict[str, List[Any]],
    output_dir: str
):
    """
    Create a simple dashboard with multiple plots for recommendation analysis.
    
    Args:
        metrics: Dictionary of evaluation metrics.
        history: Dictionary of training history.
        recommendations: Dictionary of user recommendations.
        output_dir: Directory to save the dashboard plots.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Plot training history
    plot_training_history(
        history, 
        title="Training History",
        save_path=f"{output_dir}/training_history.png"
    )
    
    # 2. Plot recommendation metrics
    plot_recommendation_metrics(
        metrics,
        title="Recommendation Performance Metrics",
        save_path=f"{output_dir}/recommendation_metrics.png"
    )
    
    # 3. Plot recommendation diversity
    plot_recommendation_diversity(
        recommendations,
        title="Recommendation Diversity Analysis",
        save_path=f"{output_dir}/recommendation_diversity.png"
    )
    
    # 4. Plot top recommendations per user
    plot_top_jobs_per_user(
        recommendations,
        title="Top Job Recommendations per User",
        save_path=f"{output_dir}/top_recommendations.png"
    )
    
    # Create simple HTML dashboard
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Recommendation System Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; margin-bottom: 20px; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin-bottom: 20px; }}
            .metric {{ padding: 10px; background-color: #f0f0f0; border-radius: 5px; }}
            .plots {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 20px; }}
            .plot {{ width: 100%; box-shadow: 0 0 5px rgba(0, 0, 0, 0.2); }}
            h1, h2, h3 {{ color: #333; }}
            .metric-value {{ font-weight: bold; font-size: 1.2em; color: #0066cc; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>JibJob Recommendation System Dashboard</h1>
                <p>Performance metrics and visualizations</p>
            </div>
            
            <h2>Key Metrics</h2>
            <div class="metrics">
    """
    
    # Add metrics
    for name, value in metrics.items():
        html_content += f"""
                <div class="metric">
                    <h3>{name}</h3>
                    <div class="metric-value">{value:.4f}</div>
                </div>
        """
    
    html_content += """
            </div>
            
            <h2>Visualizations</h2>
            <div class="plots">
                <div>
                    <h3>Training History</h3>
                    <img class="plot" src="training_history.png" alt="Training History">
                </div>
                <div>
                    <h3>Recommendation Metrics</h3>
                    <img class="plot" src="recommendation_metrics.png" alt="Recommendation Metrics">
                </div>
                <div>
                    <h3>Recommendation Diversity</h3>
                    <img class="plot" src="recommendation_diversity.png" alt="Recommendation Diversity">
                </div>
                <div>
                    <h3>Top Recommendations per User</h3>
                    <img class="plot" src="top_recommendations.png" alt="Top Recommendations">
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(f"{output_dir}/dashboard.html", 'w') as f:
        f.write(html_content)
    
    logger.info(f"Dashboard created in: {output_dir}")
    logger.info(f"Open {output_dir}/dashboard.html in a web browser to view the dashboard")


def plot_graph(edge_index, num_users, num_jobs, figsize=(10, 8), user_color='blue', job_color='red'):
    """
    Plot the bipartite graph of users and jobs.
    
    Args:
        edge_index (torch.Tensor): The edge index of shape [2, num_edges]
        num_users (int): Number of user nodes
        num_jobs (int): Number of job nodes
        figsize (tuple): Figure size
        user_color (str): Color for user nodes
        job_color (str): Color for job nodes
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import torch
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    user_nodes = [f"U{i}" for i in range(num_users)]
    job_nodes = [f"J{i}" for i in range(num_jobs)]
    
    G.add_nodes_from(user_nodes, bipartite=0)
    G.add_nodes_from(job_nodes, bipartite=1)
    
    # Add edges
    edges = []
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    
    for i in range(edge_index.shape[1]):
        user_id = edge_index[0, i]
        job_id = edge_index[1, i]
        edges.append((f"U{user_id}", f"J{job_id}"))
    
    G.add_edges_from(edges)
    
    # Plot the graph
    plt.figure(figsize=figsize)
    
    # Create positions
    pos = nx.bipartite_layout(G, [f"U{i}" for i in range(num_users)])
    
    # Draw nodes
    node_color = [user_color] * num_users + [job_color] * num_jobs
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=user_nodes,
                          node_color=user_color,
                          node_size=200,
                          alpha=0.8,
                          label="Users")
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=job_nodes,
                          node_color=job_color,
                          node_size=200,
                          alpha=0.8,
                          label="Jobs")
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(f"Bipartite Graph: {num_users} Users, {num_jobs} Jobs, {len(edges)} Interactions")
    plt.legend()
    plt.axis('off')
    
    return plt


def plot_rating_distribution(ratings, bins=10, figsize=(8, 6)):
    """
    Plot the distribution of ratings.
    
    Args:
        ratings (array-like): Array of rating values
        bins (int): Number of bins for the histogram
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    plt.figure(figsize=figsize)
    
    # Plot histogram
    plt.hist(ratings, bins=bins, color='skyblue', edgecolor='black')
    
    # Add vertical line at the mean
    mean_rating = np.mean(ratings)
    plt.axvline(mean_rating, color='red', linestyle='dashed', linewidth=1, 
               label=f'Mean: {mean_rating:.2f}')
    
    # Add labels and title
    plt.xlabel('Rating Value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Ratings')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    
    return plt


def plot_recommendation_quality(
    actual_ratings: np.ndarray,
    predicted_ratings: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
    alpha: float = 0.5,
    title: str = 'Recommendation Quality',
    xlabel: str = 'Actual Ratings',
    ylabel: str = 'Predicted Ratings'
) -> plt.Figure:
    """
    Plot actual vs predicted ratings to visualize recommendation quality.
    
    Args:
        actual_ratings: Array of actual ratings
        predicted_ratings: Array of predicted ratings
        figsize: Figure size as (width, height)
        alpha: Transparency for scatter points
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot of actual vs. predicted ratings
    scatter = ax.scatter(actual_ratings, predicted_ratings, alpha=alpha)
    
    # Add a perfect prediction line
    min_val = min(actual_ratings.min(), predicted_ratings.min())
    max_val = max(actual_ratings.max(), predicted_ratings.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    
    # Calculate and display correlation
    correlation = np.corrcoef(actual_ratings, predicted_ratings)[0, 1]
    rmse = np.sqrt(np.mean((actual_ratings - predicted_ratings) ** 2))
    
    # Add text with metrics
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    textstr = f'Correlation: {correlation:.3f}\nRMSE: {rmse:.3f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    
    # Add titles and labels
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    return fig
