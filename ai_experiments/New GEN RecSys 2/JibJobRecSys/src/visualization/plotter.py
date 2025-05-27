import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def plot_actual_vs_predicted_scores(y_true_all, y_pred_scores_all, output_plot_path):
    """
    Scatter plot of actual (0/1) vs predicted (0-1) scores for recommendation quality analysis.
    Annotates with Pearson correlation and RMSE.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true_all, y_pred_scores_all, alpha=0.3, label='Test Samples')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Interaction (0 or 1)')
    plt.ylabel('Predicted Score (0-1 Scale)')
    plt.title('JibJob Recommendation Quality')
    correlation, _ = pearsonr(y_true_all, y_pred_scores_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_scores_all))
    stats_text = f"Correlation: {correlation:.3f}\nRMSE: {rmse:.3f}"
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
