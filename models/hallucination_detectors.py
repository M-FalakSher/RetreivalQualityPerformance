import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from config import RANDOM_STATE

def get_models():
    """
    Returns a dictionary of un-trained ML models for hallucination detection.
    """
    return {
        "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE)
    }

def train_and_predict(model_name, model, X_train, y_train, X_test):
    """
    Trains the model, predicts on the test set, and tracks training time.
    """
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    
    return y_pred, y_proba, training_time