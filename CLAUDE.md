# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational machine learning repository containing Jupyter notebooks for coursework and projects. Two main projects plus several assignment notebooks covering ML fundamentals through deep learning.

## Running Notebooks

All code lives in Jupyter notebooks. There is no build system, test suite, or CLI tooling. To run:

```bash
jupyter notebook           # Launch Jupyter in this directory
jupyter lab                # Or use JupyterLab
```

Execute notebooks via "Kernel → Restart & Run All" or run cells individually.

## Dependencies

No `requirements.txt` exists. The implicit dependencies are:

```
numpy, pandas, matplotlib, seaborn, scikit-learn, xgboost, tensorflow (2.x), imbalanced-learn
```

Install with: `pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow imbalanced-learn`

## Key Projects

### Bank Marketing Prediction (`BankMarketing_Ibraiz.ipynb`)
- Binary classification: predicts term deposit subscriptions from `bank-additional-full.csv`
- Highly imbalanced dataset (88.7% / 11.3%)
- Models: Logistic Regression, Random Forest, XGBoost, Deep Neural Network (all with tuning variants)
- Techniques: SMOTE, GridSearchCV with StratifiedKFold, threshold tuning, early stopping, BatchNorm, Dropout, L2 regularization
- Encoding: one-hot for linear/DNN models, factorize for tree-based models
- Scaling: StandardScaler applied to linear/DNN models only

### Fashion MNIST Classification (`DeepLearningAssignment.ipynb`)
- Multi-class image classification (10 classes, 28x28 grayscale)
- Models: Logistic Regression, Naive Bayes, Random Forest, Dense NN, CNN (with architecture variants)
- Data loaded via `tensorflow.keras.datasets.fashion_mnist`

## Reproducibility

Notebooks use fixed seeds: `np.random.seed(73)`, `tf.random.set_seed(73)`, and `random_state=42` in scikit-learn calls.

## File Organization

Flat structure at root level. Notebooks follow the naming pattern `Assignment_<Topic>.ipynb` with corresponding `-solution.ipynb` files. PDFs are lecture slides and reference textbooks.
