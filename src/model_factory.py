from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, concatenate, Dropout

def create_hybrid_model(vocab_size, embedding_dim=128, filters=128, kernel_size=5, dense_units=18, max_seq_length=250):
    """
    Creates the CheckOrigin Hybrid CNN-MLP Model.
    """
    # CNN path for text tokens
    text_input = Input(shape=(max_seq_length,), name='text_sequence_input')
    x = Embedding(vocab_size, embedding_dim)(text_input)
    x = Conv1D(filters, kernel_size, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.3)(x)
    
    # MLP path for linguistic features
    ling_input = Input(shape=(3,), name='linguistic_features_input')
    y = Dense(dense_units, activation='relu')(ling_input)
    y = Dropout(0.3)(y)
    
    # Combined Fusion Layer
    combined = concatenate([x, y])
    z = Dense(1, activation='sigmoid', name='output')(combined)
    
    model = Model(inputs=[text_input, ling_input], outputs=z)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
