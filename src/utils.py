import numpy as np
def pad_sequences(sequences, maxlen, padding='post', truncating='post', value=0):
    """
    Pads sequences to the same length.
    """
    padded_sequences = []
    for seq in sequences:
        if len(seq) > maxlen:
            if truncating == 'pre':
                seq = seq[-maxlen:]
            else:
                seq = seq[:maxlen]
        elif len(seq) < maxlen:
            pad_len = maxlen - len(seq)
            if padding == 'pre':
                seq = [value] * pad_len + list(seq)
            else:
                seq = list(seq) + [value] * pad_len
        padded_sequences.append(seq)
    return np.array(padded_sequences)

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
