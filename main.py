import warnings
import os
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Ensure the current directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiments.noise_analysis import run_experiment
from utils.helpers import plot_all_results

def main():
    print("==================================================")
    print("RAG Retrieval Quality & Hallucination Analysis")
    print("==================================================")
    
    # Run the experiment
    results_df, confusion_matrices = run_experiment()
    
    print("\n==================================================")
    print("Final Results Summary")
    print("==================================================")
    
    # Drop confusion matrix column for neat printing
    print_df = results_df.drop(columns=['Confusion Matrix'])
    print(print_df.to_string(index=False))
    
    # Find the best model based on Accuracy (or Profit Score)
    best_model_row = results_df.loc[results_df['Accuracy (%)'].idxmax()]
    best_model_name = best_model_row['Model']
    print(f"\nBest Model based on Accuracy: {best_model_name}")
    
    # Generate Plots
    plot_all_results(results_df, best_model_name, confusion_matrices[best_model_name])
    
    print("\nAnalysis complete. Check the 'output' folder for graphs.")

if __name__ == "__main__":
    main()