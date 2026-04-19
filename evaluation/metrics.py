from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix
from config import TIME_PENALTY_FACTOR

def calculate_metrics(y_true, y_pred, y_proba, training_time):
    """
    Calculates Accuracy, Log Loss, F1 Score, and a custom Profit Score.
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    loss = None
    if y_proba is not None:
        try:
            loss = log_loss(y_true, y_proba)
        except ValueError:
            loss = float('nan') # Handle cases with only one class in y_true
            
    # Profit Score Calculation
    # Trade-off between accuracy and computational cost
    profit = (acc * 100) - (training_time * TIME_PENALTY_FACTOR)
    
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "Accuracy (%)": acc * 100,
        "Loss": loss if loss is not None else float('inf'),
        "F1 Score": f1,
        "Profit Score": profit,
        "Training Time (s)": training_time,
        "Confusion Matrix": cm
    }