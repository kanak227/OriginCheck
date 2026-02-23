import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def get_linguistic_features(text, num_features=3):
    """
    Extracts statistical linguistic features from text.
    - Text Length
    - Average Word Length
    - Unique Word Ratio (if num_features=3)
    """
    words = text.split()
    if not words:
        return np.zeros(num_features)
    
    text_length = len(words)
    avg_word_length = np.mean([len(word) for word in words])
    
    if num_features == 2:
        return np.array([text_length, avg_word_length])
    
    unique_word_ratio = len(set(words)) / text_length
    return np.array([text_length, avg_word_length, unique_word_ratio])

def preprocess_text(text, tokenizer, max_length=250):
    """
    Tokenizes and pads text for the CNN path.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    return padded
