# utils/visualization.py
"""
Visualization utilities for model performance
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
import os

def set_plot_style():
    """Set consistent plot styling"""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

def plot_accuracy_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Plot model accuracy comparison"""
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = results_df['Model'].tolist()
    accuracies = results_df['Accuracy (%)'].tolist()
    
    bars = ax.bar(models, accuracies, color='steelblue', edgecolor='black')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison for Hallucination Detection', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(accuracies) + 10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_log_loss_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Plot log loss comparison"""
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out NaN values
    valid_data = results_df[results_df['Loss'].notna()]
    models = valid_data['Model'].tolist()
    losses = valid_data['Loss'].tolist()
    
    bars = ax.bar(models, losses, color='coral', edgecolor='black')
    
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{loss:.3f}', ha='center', va='bottom')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Log Loss', fontsize=12, fontweight='bold')
    ax.set_title('Model Log Loss Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_profit_score_comparison(results_df: pd.DataFrame, save_path: str = None):
    """Plot profit score comparison (accuracy vs efficiency)"""
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = results_df['Model'].tolist()
    profit_scores = results_df['Profit Score'].tolist()
    
    colors = ['green' if score > 0 else 'red' for score in profit_scores]
    bars = ax.bar(models, profit_scores, color=colors, edgecolor='black')
    
    for bar, score in zip(bars, profit_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{score:.1f}', ha='center', va='bottom')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Efficiency (Profit Score = Accuracy - Complexity Penalty)', 
                 fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(cm: np.ndarray, model_name: str, save_path: str = None):
    """Plot confusion matrix for a model"""
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Correct', 'Hallucination'],
                yticklabels=['Correct', 'Hallucination'])
    
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_noise_impact_analysis(experiment_results: pd.DataFrame, save_path: str = None):
    """Plot noise impact analysis results"""
    set_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Hallucination rate by noise type
    noise_rates = experiment_results.groupby('experiment_type')['label'].mean() * 100
    noise_rates = noise_rates.reindex(['gold_only', 'retrieval_only', 'with_random', 
                                        'with_distractors', 'mixed_noise'])
    
    axes[0].bar(noise_rates.index, noise_rates.values, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Experiment Type', fontsize=11)
    axes[0].set_ylabel('Hallucination Rate (%)', fontsize=11)
    axes[0].set_title('Impact of Noise Type on Hallucination Rate', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    for i, (bar, rate) in enumerate(zip(axes[0].patches, noise_rates.values)):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Context length impact
    experiment_results['context_length_group'] = pd.cut(experiment_results['context_length'], 
                                                         bins=[0, 500, 1000, 2000, 5000],
                                                         labels=['<500', '500-1000', '1000-2000', '>2000'])
    length_impact = experiment_results.groupby('context_length_group')['label'].mean() * 100
    
    axes[1].bar(range(len(length_impact)), length_impact.values, color='coral', edgecolor='black')
    axes[1].set_xticks(range(len(length_impact)))
    axes[1].set_xticklabels(length_impact.index)
    axes[1].set_xlabel('Context Length (tokens)', fontsize=11)
    axes[1].set_ylabel('Hallucination Rate (%)', fontsize=11)
    axes[1].set_title('Impact of Context Length on Hallucination Rate', fontsize=12, fontweight='bold')
    
    for i, (bar, rate) in enumerate(zip(axes[1].patches, length_impact.values)):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()