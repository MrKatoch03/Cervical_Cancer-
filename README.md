# Cervical Cancer Image Classification with Ensemble Deep Learning

## Overview
This project applies **deep learning and ensemble techniques** for cervical cancer image classification using the **SipakMed dataset**. It leverages **EfficientNet**, **Vision Transformer**, and **DenseNet** as base models for feature extraction, followed by a **Random Forest ensemble classifier** for final predictions. The approach combines transfer learning, fine-tuning, and classical ensemble methods to achieve high classification accuracy.

---

## Key Features
- Data augmentation (resizing, random flips, rotations, color jitter, normalization)  
- Fine-tuned deep learning models:
  - **EfficientNet_v2_s**
  - **Vision Transformer (ViT-B16)**
  - **DenseNet201**
- Feature extraction from pre-trained models  
- Random Forest ensemble for classification  
- Performance evaluation with accuracy, F1-score, precision, MSE, and confusion matrix  
- Modular design for training, evaluation, and feature extraction  

---

## Technology and Libraries
- **Frameworks:** PyTorch, Torchvision  
- **Machine Learning:** scikit-learn (Random Forest, metrics, preprocessing)  
- **Data Processing & Visualization:** numpy, matplotlib, seaborn  
- **Optimization & Training:** AdamW optimizer, CosineAnnealingLR scheduler  
- **Hardware:** GPU/CPU support via `torch.device`  

---

## Project Performance
- **Training Accuracy:** ~98.71%  
- **Validation Accuracy:** ~96.55%  
- **Test Accuracy:** ~97.81%  
- **F1-score (Test):** 0.9781  
- **Precision (Test):** 0.9783  
- **MSE (Test):** 0.049  

#### These results demonstrate strong performance of the ensemble, with minimal overfitting due to feature extraction and normalization strategies.
---

## Dataset and Models
- **Cervical Cancer Detection and Classification**  
- **Pretrained Models:** [Google Drive Link](https://drive.google.com/drive/folders/1rEwp4gOaPNI51jNfcob5H0reA8oQcS4m?usp=sharing)  
- **Dataset:** [SipakMed on Kaggle](https://www.kaggle.com/datasets/marinaeplissiti/sipakmed)  

---

## Limitations and Notes
- Pretrained weights (`.pth` files) are required for inference and need to be loaded manually.  
- Current implementation uses Random Forest as the ensemble layer; additional ensembles (e.g., XGBoost, Stacking) could improve generalization.  
- Dataset paths (`D:/capstone/Train Dataset` and `D:/capstone/Test Dataset`) should be updated for different environments.  
- Some PyTorch methods raise **deprecation warnings** (e.g., `pretrained` → use `weights` argument).  
- Future work: Online training pipelines, SHAP/LIME explainability, and real-world clinical validation.  

---
