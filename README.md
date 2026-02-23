# CheckOrigin: AI vs. Human Text Identification System

CheckOrigin is a research-driven project designed to distinguish between human-authored text and AI-generated content. By leveraging a hybrid neural network architecture that combines structural linguistic analysis with deep semantic processing, CheckOrigin achieves high accuracy in identifying the source of textual content.

## Project Overview

In the era of Large Language Models (LLMs), the boundary between human and machine-generated content is increasingly blurred. CheckOrigin addresses this challenge by analyzing not just the semantic content, but also the "linguistic fingerprints"—statistical and structural patterns unique to human writing versus algorithmic generation.

### Core Features

- **Hybrid Architecture**: Combines a Convolutional Neural Network (CNN) for sequence modeling and a Multi-Layer Perceptron (MLP) for structural feature analysis.
- **Structural Analysis**: Incorporates features such as average word length, sentence complexity, and unique word ratios.
- **High Precision**: Trained on diverse datasets including human-written essays and AI outputs, achieving 99% accuracy on benchmark tests.
- **Real-time Detection**: Interactive web-based interface for instant text classification.

## Research Methodology

This research focuses on the integration of statistical linguistic features into a neural network framework. While deep learning models are excellent at capturing contextual relationships, human writing often exhibits subtle statistical variances in vocabulary diversity and structural consistency that machines occasionally struggle to mimic perfectly.

### Data Flow

1. **Collection**: Aggregated datasets containing labeled human and AI-generated text.
2. **Preprocessing**: Tokenization, padding, and normalization of text data.
3. **Feature Engineering**: Extraction of linguistic metrics (e.g., lexical diversity, word length distributions).
4. **Dual-Path Modeling**:
   - **Text Path**: CNN layers process tokenized sequences to understand semantic flow.
   - **Linguistic Path**: Dense layers process structural features.
5. **Fusion Layer**: Concatenates both paths for a comprehensive final classification.

## Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- NumPy, Pandas, Scikit-learn
- Streamlit (for the web app)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/CheckOrigin.git
cd CheckOrigin

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Machine Learning Pipeline
To train or evaluate the model, you can use the provided scripts:
- `train.py`: Train the CheckOrigin hybrid model.
- `evaluate.py`: Test the model against new datasets.

#### Web Interface
Launch the interactive dashboard to test individual text samples:
```bash
streamlit run app.py
```

## Results

CheckOrigin has been validated against multiple datasets, including the Reddit filtered dataset and Kaggle AI-Human text benchmarks. It demonstrates robust performance across different writing styles and topics.

| Metric | Score |
|--------|-------|
| Accuracy | 99.1% |
| Precision | 98.9% |
| Recall | 99.3% |

## Future Directions

- **Cross-Model Generalization**: Enhancing detection for newer LLMs (e.g., GPT-5, Gemini 2.0).
- **Adversarial Robustness**: Testing against "jailbroken" or obfuscated AI text.
- **Multilingual Support**: Extending structural analysis to other languages.

---

*This project was developed as part of a research initiative to enhance digital authenticity and academic integrity.*
