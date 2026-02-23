import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from src.utils import get_linguistic_features, preprocess_text
from src.model_factory import create_hybrid_model

# Constants
MAX_WORDS = 10000
MAX_SEQ_LENGTH = 250
DATA_PATH = 'data/reddit_filtered_dataset.csv' # User seems to prefer this one based on recent file
MODEL_SAVE_PATH = 'models/checkorigin_model.h5'
TOKENIZER_SAVE_PATH = 'models/tokenizer.pickle'

def train():
    print("Loading dataset...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    
    # Standardize column names if necessary
    # Assuming 'text' and 'class' based on Datathon_AI.py
    if 'generated' in df.columns:
        df.rename(columns={'generated': 'class'}, inplace=True)
    
    df.dropna(inplace=True)
    
    print(f"Dataset loaded. Total samples: {len(df)}")
    print("Class distribution:\n", df['class'].value_counts())

    # Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['class'], test_size=0.2, random_state=42
    )

    # Tokenization
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_padded = [preprocess_text(t, tokenizer, MAX_SEQ_LENGTH)[0] for t in X_train_text]
    X_test_padded = [preprocess_text(t, tokenizer, MAX_SEQ_LENGTH)[0] for t in X_test_text]
    
    X_train_padded = np.array(X_train_padded)
    X_test_padded = np.array(X_test_padded)

    # Features
    print("Extracting linguistic features (3-feature set)...")
    X_train_ling = np.array([get_linguistic_features(t, num_features=3) for t in X_train_text])
    X_test_ling = np.array([get_linguistic_features(t, num_features=3) for t in X_test_text])

    # Model
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)
    model = create_hybrid_model(vocab_size, max_seq_length=MAX_SEQ_LENGTH)
    
    print("Training CheckOrigin model...")
    model.fit(
        [X_train_padded, X_train_ling], np.array(y_train),
        validation_data=([X_test_padded, X_test_ling], np.array(y_test)),
        epochs=10,
        batch_size=32
    )

    # Save
    print(f"Saving model to {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    
    print(f"Saving tokenizer to {TOKENIZER_SAVE_PATH}")
    with open(TOKENIZER_SAVE_PATH, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("Training complete!")

if __name__ == "__main__":
    train()
