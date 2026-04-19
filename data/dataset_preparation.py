import pandas as pd
import numpy as np
from datasets import load_dataset
import random
from config import DATASET_NAME, SAMPLE_SIZE

def load_real_dataset():
    print(f"Loading {DATASET_NAME} dataset...")
    # SQuAD v2 has unanswerable questions too, but we will focus on answerable ones for our context needs
    dataset = load_dataset(DATASET_NAME, split='train')
    
    # Filter only answerable questions
    df = dataset.to_pandas()
    df = df[df['answers'].apply(lambda x: len(x['text']) > 0)]
    
    # Sample real data
    df = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)
    
    questions = df['question'].tolist()
    gold_contexts = df['context'].tolist()
    # Extract the first answer
    true_answers = df['answers'].apply(lambda x: x['text'][0]).tolist()
    
    return questions, gold_contexts, true_answers

def prepare_noise_dataset(questions, gold_contexts, true_answers):
    print("Preparing noise variations (Gold, Distractor, Random)...")
    data_rows = []
    
    # For Random Context, we just shuffle the gold contexts completely
    random_contexts = gold_contexts.copy()
    random.shuffle(random_contexts)
    
    for i in range(len(questions)):
        q = questions[i]
        ans = true_answers[i]
        
        # 1. Gold Context (Correct)
        data_rows.append({
            'question': q,
            'context': gold_contexts[i],
            'noise_type': 'gold',
            'true_answer': ans
        })
        
        # 2. Random Context (Benign Noise)
        data_rows.append({
            'question': q,
            'context': random_contexts[i],
            'noise_type': 'random',
            'true_answer': ans
        })
        
        # 3. Distractor Context (Harmful Noise)
        # For simplicity in this local setup, we use another random context
        # In a full retriever setup, this would be a high-similarity but wrong document
        # Let's pick a context that has some similar words but is not the gold one.
        # We'll just use a shifted context to represent a "different" context.
        distractor_idx = (i + 1) % len(questions)
        data_rows.append({
            'question': q,
            'context': gold_contexts[distractor_idx],
            'noise_type': 'distractor',
            'true_answer': ans
        })
        
    return pd.DataFrame(data_rows)

if __name__ == "__main__":
    q, c, a = load_real_dataset()
    df = prepare_noise_dataset(q, c, a)
    print(df.head())