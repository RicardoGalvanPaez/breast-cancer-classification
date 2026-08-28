# Breast Cancer Classification

Binary classification model to detect malignant vs. benign breast tumors using 
Logistic Regression and dimensionality reduction techniques.

## Problem Statement

Early and accurate detection of breast cancer is critical to improve patient outcomes. 
This project builds a classification pipeline to predict whether a tumor is malignant (M) 
or benign (B) based on morphological features extracted from biopsy images.

## Dataset

- **Source:** UCI Machine Learning Repository — Wisconsin Breast Cancer Dataset
- **Records:** 569 samples
- **Features:** 30 numerical features (mean, standard error, and worst values of 
  radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, 
  and fractal dimension)
- **Target:** `diagnosis` — Malignant (M) or Benign (B)

## Tech Stack

- Python 3.9
- Pandas, NumPy
- Scikit-learn (LogisticRegression, PCA, StandardScaler, PowerTransformer, Pipeline)
- Matplotlib, Seaborn
- Google Colab

## Methodology

Three progressively improved pipelines were built and compared:

| Model | Recall | Precision | Accuracy |
|---|---|---|---|
| Baseline (Correlation + Clean) | 76.2% | 88.9% | 87.7% |
| Standard Scaler + PCA | 90.5% | 97.4% | 95.6% |
| **Yeo-Johnson + Correlation** | **95.2%** | **100.0%** | **98.2%** |

## Key Results

The best model achieved **98.2% accuracy** and **100% precision** with a recall of 
**95.2%** — meaning it correctly identified nearly all malignant tumors while 
generating zero false positives.

Given the clinical context, **recall** was prioritized as the critical metric: 
missing a malignant tumor (false negative) carries a far higher cost than a false 
positive.

## What I Learned

- Impact of feature selection and correlation analysis on model performance
- How PowerTransformer (Yeo-Johnson) handles skewed medical data better than 
  standard scaling
- Trade-offs between precision and recall in high-stakes classification problems
- Building reproducible ML pipelines with Scikit-learn's Pipeline API
