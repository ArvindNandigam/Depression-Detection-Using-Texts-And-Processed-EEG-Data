# Depression-Detection-Using-Texts-And-Processed-EEG-Data

This repository presents a **multimodal deep learning framework** that integrates **social media text analysis** with **clinical EEG signal processing** to detect signs of **depression**. By combining Natural Language Processing (NLP) with neural signal analysis, the system enhances the accuracy and interpretability of early depression detection, making it a valuable tool in modern mental health diagnostics.

## Key Features

- **Text-Based Detection**  
  Utilizes a pre-trained BERT model enhanced with CNN layers and multi-head self-attention to extract depressive cues from Reddit posts and mental health corpora.

- **EEG-Based Detection**  
  Employs a deep neural network trained on processed EEG signals to identify neurophysiological patterns indicative of depressive disorders.

- **Multimodal Integration**  
  - Ensemble strategies: Soft and hard voting mechanisms  
  - Cross-modal probability calibration  
  - EEG duplication and text sampling for data alignment

- **Comprehensive Evaluation**  
  - Accuracy assessment across multiple fusion strategies  
  - Experimental analysis of fusion techniques  
  - Visual analytics for model behavior and interpretability

## Datasets Used

- Reddit Depression Dataset (Kaggle)  
- Mental Health Text Corpus (Kaggle)  
- EEG Psychiatric Disorders Dataset (Kaggle)

All datasets are preprocessed for training and evaluation.

## Architecture Overview

### Text Processing

- BERT (Hugging Face Transformers)  
- Convolutional Neural Networks (CNN)  
- Multi-Head Self-Attention  

### EEG Processing

- EEG feature engineering  
- Fully Connected Deep Neural Network (DNN)  
- Dropout-based regularization  

### Multimodal Fusion

- Decision-level ensembling  
- Cross-modal normalization  
- Data duplication and sampling alignment  

## Technologies and Libraries

- PyTorch  
- Hugging Face Transformers  
- scikit-learn  
- StandardScaler  
- AdamW optimizer  
- Cosine Annealing Learning Rate Scheduler  

## Results Summary

- **Text-Only Model**  
  Achieved high accuracy with contextual understanding of depressive language.

- **EEG-Only Model**  
  Demonstrated robust performance in classifying EEG patterns related to depression.

- **Multimodal Ensemble**  
  Outperformed unimodal models using decision-level fusion and calibration strategies.
