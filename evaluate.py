import pandas as pd
import numpy as np
import pickle
import os
import sys
import onnxruntime as ort

# Workaround for Keras 3 pickle load compatibility with Keras 2 tokenizer
try:
    import keras.src.legacy.preprocessing.text
    sys.modules['keras.src.preprocessing.text'] = keras.src.legacy.preprocessing.text
except ImportError:
    pass

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.utils import get_linguistic_features, preprocess_text

# Configuration
MAX_SEQ_LENGTH = 150
MODEL_PATH = 'models/checkorigin_model.onnx'
TOKENIZER_PATH = 'models/tokenizer.pickle'
TEST_DATA_PATH = 'data/reddit_filtered_dataset.csv'

def evaluate():
    print("Loading resources...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TOKENIZER_PATH):
        print("Error: Model or Tokenizer not found.")
        return

    model = ort.InferenceSession(MODEL_PATH)
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    print(f"Loading test data from {TEST_DATA_PATH}...")
    df = pd.read_csv(TEST_DATA_PATH)
    df.dropna(inplace=True)
    
    # Map columns if necessary
    if 'Data' in df.columns:
        df.rename(columns={'Data': 'text'}, inplace=True)
    if 'Labels' in df.columns:
        df.rename(columns={'Labels': 'class'}, inplace=True)
    elif 'Label' in df.columns:
        df.rename(columns={'Label': 'class'}, inplace=True)
    
    texts = df['text'].tolist()
    labels = df['class'].tolist()

    print(f"Evaluating on {len(texts)} samples...")
    
    # Determine expected linguistic feature count from model
    try:
        num_ling_features = model.get_inputs()[1].shape[1]
        if isinstance(num_ling_features, str) or num_ling_features is None:
            num_ling_features = 2
    except:
        num_ling_features = 2

    # Preprocess all
    X_padded = np.array([preprocess_text(t, tokenizer, MAX_SEQ_LENGTH)[0] for t in texts])
    X_ling = np.array([get_linguistic_features(t, num_features=num_ling_features) for t in texts])
    
    # Predict
    input_name_1 = model.get_inputs()[0].name
    input_name_2 = model.get_inputs()[1].name
    output_name = model.get_outputs()[0].name
    
    padded_seq_onnx = X_padded.astype(np.int32)
    ling_features_onnx = X_ling.astype(np.float32)
    
    outputs = model.run([output_name], {input_name_1: padded_seq_onnx, input_name_2: ling_features_onnx})
    predictions = outputs[0]
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
