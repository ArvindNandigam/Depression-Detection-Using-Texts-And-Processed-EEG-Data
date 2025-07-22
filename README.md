# Depression-Detection-Using-Texts-And-Processed-EEG-Data
This repository contains a multimodal deep learning framework that integrates social media text analysis and clinical EEG signals to detect signs of depression. By combining Natural Language Processing (NLP) and neural signal analysis, this approach aims to improve early detection accuracy and enhance interpretability in mental health applications.

Key Features
Text-Based Detection: Uses pre-trained BERT with CNN layers and self-attention to identify depressive cues from Reddit and mental health corpora.
EEG-Based Detection: Deep Neural Network trained on clinical EEG data to classify depressive disorders using neurophysiological patterns.
Multimodal Integration:
Ensemble strategies (soft/hard voting)
Cross-modal probability calibration
EEG duplication and text sampling for alignment
Comprehensive Evaluation: Accuracy measured across multiple fusion strategies with experimental variations and visual analytics.
Datasets Used
Reddit Depression Dataset (Kaggle)
Mental Health Text Corpus (Kaggle)
EEG Psychiatric Disorders Dataset (Kaggle)
Architecture Highlights
BERT + CNN + Multihead Attention for text
Fully Connected Layers for EEG
Decision-level ensemble with various normalization and sampling strategies
Trained using PyTorch and Hugging Face Transformers
Technologies
PyTorch, Transformers (HuggingFace), scikit-learn
StandardScaler, AdamW, cosine annealing scheduler
EEG feature engineering and dropout-based regularization
Results
Text-only model: High accuracy and contextual understanding
EEG-only model: Robust neurophysiological signal classification
Best Ensemble: Hard voting with random BERT sampling reached ~77.25% accuracy
