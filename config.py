import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data_cache')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset configuration
DATASET_NAME = "squad_v2" # Real dataset with questions, contexts, and answers
SAMPLE_SIZE = 500  # Number of samples to process (keep small for local performance but large enough for real data simulation)

# LLM Configuration for Answer Generation
# We use a very small text generation model locally to keep things efficient.
# In a full-scale deployment, this might be Llama-3 or similar.
GENERATOR_MODEL_NAME = "google/flan-t5-small"

# Retriever Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ML Model Evaluation
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Profit Score Formulation:
# profit = (accuracy * 100) - (training_time_seconds * TIME_PENALTY_FACTOR)
TIME_PENALTY_FACTOR = 0.5