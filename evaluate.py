import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.utils import get_linguistic_features, preprocess_text

# Configuration
MAX_SEQ_LENGTH = 250
MODEL_PATH = 'models/checkorigin_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pickle'
TEST_DATA_PATH = 'data/reddit_filtered_dataset.csv'

def evaluate():
    print("Loading resources...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Error: Model or Tokenizer not found.")
        return

    model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    print(f"Loading test data from {TEST_DATA_PATH}...")
    df = pd.read_csv(TEST_DATA_PATH)
    df.dropna(inplace=True)
    
    # Take a sample if dataset is too large, or just use whole for evaluation
    # df = df.sample(1000) 
    
    texts = df['text'].tolist()
    labels = df['class'].tolist()

    print(f"Evaluating on {len(texts)} samples...")
    
    # Determine expected linguistic feature count from model
    try:
        num_ling_features = model.input[1].shape[1]
    except:
        num_ling_features = 2

    # Preprocess all
    X_padded = np.array([preprocess_text(t, tokenizer, MAX_SEQ_LENGTH)[0] for t in texts])
    X_ling = np.array([get_linguistic_features(t, num_features=num_ling_features) for t in texts])
    
    # Predict
    predictions = model.predict([X_padded, X_ling], batch_size=32)
    binary_predictions = (predictions >= 0.5).astype(int).flatten()

    # Metrics
    acc = accuracy_score(labels, binary_predictions)
    print(f"\nOverall Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:\n")
    print(classification_report(labels, binary_predictions, target_names=['Human', 'AI']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, binary_predictions))

if __name__ == "__main__":
    evaluate()
