Higgs Boson Event Detection Using Machine Learning
Overview

This project applies machine learning techniques to the problem of Higgs boson event detection using simulated collision data from the ATLAS Higgs Machine Learning Challenge (2014).
The goal is to distinguish rare Higgs (signal) events from dominant background processes by learning patterns in high-dimensional physics features.

The project implements a complete end-to-end pipeline, including:

Data exploration and preprocessing

Handling class imbalance and encoded missing values

Model training using multiple supervised learning algorithms

Evaluation using metrics suitable for imbalanced datasets

Comparative analysis and visualisation of model performance

Dataset

Source: ATLAS Higgs Machine Learning Challenge 2014 (CERN Open Data)

Events: ~818,000 simulated proton–proton collisions

Features: 30 numerical physics-derived variables

Target:

s → Higgs (signal)

b → Background

Key challenges:

Severe class imbalance

Missing measurements encoded as -999

High-dimensional, non-linear feature interactions

Project Structure
Higgs-Boson/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_xgboost.ipynb
│   └── 06_model_comparison.ipynb
│
├── data/
│   ├── X_train_scaled.npy
│   ├── X_test_scaled.npy
│   ├── X_train_unscaled.npy
│   ├── X_test_unscaled.npy
│   ├── y_train.npy
│   └── y_test.npy
│
├── results/
│   ├── figures/
│   │   ├── lr_confusion_matrix.png
│   │   ├── rf_confusion_matrix.png
│   │   ├── xgb_confusion_matrix.png
│   │   ├── *_roc_curve.png
│   │   ├── *_pr_curve.png
│   │   └── model_comparison_bar.png
│   ├── model_comparison.csv
│   └── roc_comparison.png
│
├── requirements.txt
└── README.md

Data Exploration

Exploratory analysis was performed to understand:

Dataset structure and feature distributions

Class imbalance between signal and background events

Encoded missing values (-999) common in jet-related features

Feature correlations motivating the use of tree-based models

Visualisations include:

Class distribution plots

Feature histograms

Correlation heatmaps (random feature subsets)

Preprocessing

Key preprocessing steps:

Label encoding (b → 0, s → 1)

Replacement of encoded missing values (-999 → NaN)

Median imputation for robustness against skewed distributions

Removal of non-numeric features

Stratified train–test split (80/20)

Feature scaling using StandardScaler (for Logistic Regression)

Handling class imbalance via class weights

Processed datasets are saved to disk for reproducibility and consistency across models.

Models Implemented
1. Logistic Regression (Baseline)

Used as an interpretable baseline model

Trained on scaled features

Handles class imbalance via balanced class weights

Provides a reference point for more complex models

2. Random Forest

Ensemble of decision trees using bagging

Captures non-linear feature interactions

Robust to noise and overfitting

Provides feature importance analysis

3. XGBoost

Gradient boosting ensemble model

Optimised for imbalanced classification using scale_pos_weight

Sequential learning to correct previous errors

Achieves the strongest overall performance

Evaluation Metrics

Due to class imbalance, evaluation focuses on:

Precision

Recall

F1-score

ROC-AUC

Additional visual diagnostics:

Confusion matrices

ROC curves

Precision–Recall curves

Cross-model ROC comparison

Model Performance Summary
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.9619	0.9078	0.9891	0.9467	0.9946
Random Forest	0.9995	0.9999	0.9986	0.9993	~1.000
XGBoost	0.99998	0.99995	0.99998	0.99996	~1.000

Ensemble-based models significantly outperform the linear baseline, demonstrating their ability to capture complex physics-driven relationships.

Key Findings

Logistic Regression provides strong recall but limited expressive power

Random Forest and XGBoost achieve near-perfect classification

Ensemble models are highly effective for rare-event detection

Results highlight the importance of proper preprocessing and metric selection

Technologies Used

Python

NumPy, Pandas

Scikit-learn

XGBoost

Matplotlib, Seaborn

Jupyter Notebook

Git & GitHub

Notes on Generalisation

While ensemble models achieved near-perfect scores, further validation using:

Cross-validation

Independent test sets
would be required in real experimental deployments to fully confirm generalisation performance.

Author

Bishnu Parajuli
Final Year AI Coursework Project

License

This project is for academic and educational purposes only.
