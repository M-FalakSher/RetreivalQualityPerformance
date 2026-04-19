import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from config import OUTPUT_DIR

def plot_metrics_comparison(results_df, metric_col, title, filename):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y=metric_col, data=results_df, palette='viridis')
    plt.title(title)
    plt.ylabel(metric_col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_confusion_matrix(cm, model_name, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Hallucination (0)', 'Correct (1)'], 
                yticklabels=['Hallucination (0)', 'Correct (1)'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def plot_all_results(results_df, best_model_name, best_model_cm):
    print("Generating evaluation plots...")
    
    # 1. Accuracy Comparison
    plot_metrics_comparison(results_df, 'Accuracy (%)', 'Model Accuracy Comparison', 'accuracy_comparison.png')
    
    # 2. Log Loss Comparison
    plot_metrics_comparison(results_df, 'Loss', 'Log Loss Comparison', 'log_loss_comparison.png')
    
    # 3. Profit Score Comparison
    plot_metrics_comparison(results_df, 'Profit Score', 'Model Efficiency (Profit Score) Comparison', 'profit_score_comparison.png')
    
    # 4. Confusion Matrix for the best model
    if best_model_cm is not None:
        plot_confusion_matrix(best_model_cm, best_model_name, 'best_model_confusion_matrix.png')
        
    print(f"Plots saved to {OUTPUT_DIR}")