# 🚢 Titanic Survival Prediction — End-to-End ML Pipeline

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Domain-Binary%20Classification-green" />
  <img src="https://img.shields.io/badge/Best%20ROC--AUC-0.88-brightgreen" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> **Author:** Mohamed El Otmany — Engineering Student, Finance & Actuarial Science · FST Errachidia  
> **Dataset:** Titanic Passenger Manifest — 1,309 observations  
> **Objective:** Predict binary survival outcome using a rigorous, production-oriented supervised learning pipeline.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Methodology](#-methodology)
  - [1. Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Training & Benchmarking](#3-model-training--benchmarking)
  - [4. Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [5. Evaluation](#5-evaluation)
- [Business Insights](#-business-insights)
- [Installation & Usage](#-installation--usage)
- [Future Work](#-future-work)
- [References](#-references)

---

## 🎯 Project Overview

This project implements a **complete machine learning pipeline** for the classic Titanic survival prediction problem — a binary classification task that serves as a benchmark for evaluating supervised learning frameworks. Beyond its competition context, the analytical framework mirrors **actuarial risk segmentation** and **credit scoring** pipelines commonly deployed in Banking and Insurance.

The pipeline covers:
- Systematic EDA with statistical interpretation of survival drivers
- Domain-informed feature engineering (family dynamics, fare quantiles, age buckets)
- Comparative benchmarking of **5 classification algorithms**
- Hyperparameter optimisation via **5-fold stratified GridSearchCV**
- Model selection based on **ROC-AUC** — the appropriate metric for imbalanced binary targets
- Serialisation of the best model for downstream inference

---

## 📊 Key Results

| Metric | Best Model (Random Forest — tuned) |
|--------|-------------------------------------|
| **Accuracy** | ~83% |
| **ROC-AUC** | ~0.88 |
| **F1-Score** | ~0.78 |
| **Precision** | ~0.80 |
| **Recall** | ~0.76 |

> ROC-AUC is used as the **primary selection criterion** because the target class is imbalanced (~26% survivors). AUC is threshold-independent and measures the discriminative power of the classifier across all operating points — a standard practice in actuarial and credit risk modelling.

---

## 🗂️ Repository Structure

```
titanic-survival-prediction/
│
├── titanic-survival-prediction.ipynb   # Main notebook — full pipeline
├── train_and_test2.csv                 # Source dataset (Kaggle)
│
├── outputs/
│   ├── best_model_randomforest.joblib  # Serialised best estimator
│   ├── standard_scaler.joblib          # Fitted StandardScaler
│   └── model_metadata.json            # Best params + evaluation metrics
│
└── README.md
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis

A structured EDA investigates the statistical relationship between each feature and the survival outcome before any modelling is performed.

| Analysis | Key Finding |
|----------|-------------|
| **Survival distribution** | Only ~26% of passengers survived — moderately imbalanced dataset |
| **Gender effect** | Female survival rate ~74% vs male ~19%; dominant predictor |
| **Passenger class** | 1st class ~63% survival vs 3rd class ~24%; strong socioeconomic signal |
| **Age** | Children show higher survival; non-linear relationship with outcome |
| **Fare** | Right-skewed distribution; log-transformed for modelling; positive correlation with survival |
| **Correlation matrix** | Pclass and Fare are the most correlated numeric features (inverse relationship) |

Visualisations produced: survival count/proportion plots, grouped bar charts, age histograms and box-plots, fare density plots (raw and log-scaled), Pearson correlation heatmap.

---

### 2. Feature Engineering

Raw features are augmented with four domain-informed constructs:

| New Feature | Construction | Rationale |
|-------------|--------------|-----------|
| `FamilySize` | `SibSp + Parch + 1` | Captures total group size aboard |
| `IsAlone` | Binary flag if `FamilySize == 1` | Solo travellers showed lower survival odds |
| `AgeGroup` | Ordinal cut: Child / Young / Middle / Senior | Non-linear age effect; reduces sensitivity to imputed values |
| `FareBin` | Quartile-based encoding (0–3) | Smooths fare outliers; preserves ordinal wealth signal |

Missing value imputation strategy:
- `Age` → **median imputation** (robust to outliers, appropriate for right-skewed continuous variable)
- `Fare` → **median imputation**
- `Embarked` → **mode imputation** (2 missing values only)

All features are subsequently standardised with `StandardScaler` — mandatory for distance-based and regularised models (KNN, Logistic Regression).

---

### 3. Model Training & Benchmarking

Five classifiers are trained under identical experimental conditions (80/20 stratified split, `random_state=42`) to establish a reproducible baseline:

| Model | Rationale |
|-------|-----------|
| **Logistic Regression** | Linear baseline; interpretable coefficients; L2-regularised |
| **Decision Tree** | Non-linear; interpretable; susceptible to overfitting at depth |
| **Random Forest** | Ensemble of decorrelated trees; robust to noise |
| **Gradient Boosting** | Sequential residual correction; typically highest discriminative power |
| **K-Nearest Neighbours** | Instance-based; sensitive to feature scaling |

---

### 4. Hyperparameter Tuning

Each model is optimised via exhaustive **GridSearchCV** with `StratifiedKFold(n_splits=5)`, scoring on `roc_auc`. Parallelised with `n_jobs=-1`.

| Model | Key Tuned Parameters |
|-------|---------------------|
| Logistic Regression | `C` ∈ {0.01, 0.1, 1, 10, 100}, `penalty=l2` |
| Decision Tree | `max_depth` ∈ {3,5,7,10,None}, `criterion` ∈ {gini, entropy} |
| Random Forest | `n_estimators` ∈ {100,200,300}, `max_depth` ∈ {5,10,15,None} |
| Gradient Boosting | `learning_rate` ∈ {0.05,0.1,0.2}, `max_depth` ∈ {3,5}, `subsample` ∈ {0.8,1.0} |
| KNN | `n_neighbors` ∈ {3,5,7,9,11,15}, `weights`, `metric` |

---

### 5. Evaluation

Post-tuning evaluation reports:
- **Confusion matrices** for all 5 models (side-by-side)
- **ROC curves** overlay with per-model AUC annotation
- **Feature importance** from tree-based models (Random Forest & Gradient Boosting)
- **Full classification report** for the best model
- Grouped bar chart comparing Accuracy / F1 / ROC-AUC across models

---

## 💼 Business Insights

The survival determinants map directly to concepts central to **actuarial science** and **insurance risk modelling**:

| Survival Driver | Actuarial Parallel |
|----------------|--------------------|
| **Sex** — strongest predictor (Δ ≈ 55pp) | Gender as a risk segmentation variable in life/health underwriting |
| **Pclass** — proxy for wealth and deck location | Socioeconomic tier as an exposure metric |
| **Age** — non-linear effect; children prioritised | Age bands in mortality tables and life insurance pricing |
| **FamilySize** — optimal at 2–4 members | Group vs. individual risk pooling dynamics |
| **Fare** — higher fare ↔ higher survival | Wealth as a confounding variable in survival models |
| **Embarked** — Cherbourg passengers survived more (1st-class heavy embarkation) | Port/geography as socioeconomic proxy |

---

## ⚙️ Installation & Usage

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### Running the Notebook

```bash
# Clone the repository
git clone https://github.com/<your-username>/titanic-survival-prediction.git
cd titanic-survival-prediction

# Launch Jupyter
jupyter notebook titanic-survival-prediction.ipynb
```

> **Dataset path:** The notebook resolves the dataset at `/kaggle/input/datasets/heptapod/titanic/train_and_test2.csv` (Kaggle runtime) with a local fallback to `train_and_test2.csv`. Place the CSV in the project root for local execution.

### Inference on New Data

```python
import joblib, pandas as pd

model  = joblib.load('best_model_randomforest.joblib')
scaler = joblib.load('standard_scaler.joblib')

FEATURES = ['Age', 'Fare', 'Sex', 'SibSp', 'Parch',
            'Pclass', 'Embarked', 'FamilySize', 'IsAlone', 'AgeGroup', 'FareBin']

sample = pd.DataFrame([{
    'Age': 28, 'Fare': 72.0, 'Sex': 1,
    'SibSp': 1, 'Parch': 0, 'Pclass': 1,
    'Embarked': 0, 'FamilySize': 2, 'IsAlone': 0,
    'AgeGroup': 1, 'FareBin': 3,
}])

X_sample   = scaler.transform(sample[FEATURES])
prediction = model.predict(X_sample)[0]
proba      = model.predict_proba(X_sample)[0][1]

print(f"Prediction        : {'SURVIVED' if prediction == 1 else 'DIED'}")
print(f"Survival probability : {proba:.2%}")
```

---

## 🔮 Future Work

| Enhancement | Expected Value |
|-------------|----------------|
| **SHAP values** | Per-passenger explainability — critical for regulatory AI compliance (XAI / GDPR Article 22) |
| **SMOTE / class weighting** | Improve recall on the minority survivor class |
| **Stacking / Blending** | Meta-learner combining all 5 base models for performance uplift |
| **Shallow MLP (Keras)** | Neural network benchmark to quantify non-linearity gains |
| **Threshold optimisation** | Tune decision threshold using a domain-specific cost function |
| **MLflow tracking** | Full experiment versioning and reproducibility |
| **FastAPI endpoint** | Wrap best model in a REST micro-service for real-time scoring |

---

## 📚 References

- Kaggle Titanic Competition: https://www.kaggle.com/c/titanic
- scikit-learn documentation: https://scikit-learn.org/stable/
- VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly Media.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* (3rd ed.). O'Reilly Media.

---

<p align="center">
  Made with 🧠 by <strong>Mohamed El Otmany</strong> · FST Errachidia
</p>
