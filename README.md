# Census Income Prediction: Machine Learning Project

## Overview
This project implements a binary classification model to predict whether an individual's annual income exceeds $50,000 based on census data from 1994. The analysis follows the complete machine learning lifecycle from data exploration through model evaluation and improvement.

## Problem Statement
**Objective:** Predict income bracket (≤$50K or >$50K) using demographic and employment features

**Type:** Supervised learning, binary classification

**Business Value:** This model can help financial institutions assess loan eligibility and risk, while enabling policymakers to understand socioeconomic patterns for better resource allocation and policy development.

## Dataset
- **Source:** Census data (1994)
- **Size:** 32,561 records
- **Features:** 14 predictive features including age, education, occupation, work hours, marital status, and demographic information
- **Target:** Binary income classification (75.9% ≤$50K, 24.1% >$50K)

## Methodology

### Data Preparation
- Handled missing values in age, work hours, workclass, occupation, and native country
- Applied one-hot encoding to categorical features
- Standardized numerical features using StandardScaler
- Addressed class imbalance (3:1 ratio)

### Models Evaluated
1. **Logistic Regression** (Baseline)
2. **Random Forest Classifier**
3. **Gradient Boosting Classifier**

### Model Performance

| Model | Accuracy | ROC-AUC | Precision (>50K) | Recall (>50K) |
|-------|----------|---------|------------------|---------------|
| Logistic Regression | 85% | 0.906 | 74% | 61% |
| Random Forest | 84% | 0.891 | 69% | 63% |
| **Gradient Boosting** | **87%** | **0.923** | **79%** | **63%** |

## Key Findings
- **Top Predictive Features:** Age, hours per week, capital gain, marital status (married), and education level
- **Best Model:** Gradient Boosting achieved the highest performance with 87% accuracy and 0.923 ROC-AUC score
- Successfully balanced precision and recall for minority class prediction

## Technologies Used
- Python (Pandas, NumPy, Scikit-learn)
- Visualization: Matplotlib, Seaborn
- ML Pipeline: StandardScaler, OneHotEncoder, ColumnTransformer
