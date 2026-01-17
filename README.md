# Higgs Boson Event Detection Using Machine Learning

## Overview

This project applies machine learning techniques to the problem of Higgs boson event detection using simulated collision data from the ATLAS Higgs Machine Learning Challenge (2014). The objective is to distinguish rare Higgs (signal) events from dominant background processes by learning patterns from high-dimensional physics features.

The work follows a complete end-to-end machine learning pipeline, including data exploration, preprocessing, model training, evaluation, and comparative analysis across multiple algorithms.

## Dataset

Source: ATLAS Higgs Machine Learning Challenge 2014 (CERN Open Data)

- Approximately 818,000 simulated proton–proton collision events  
- 30 numerical physics-derived features  
- Target classes:
  - `s` → Higgs (signal)
  - `b` → Background  

Key challenges associated with the dataset include severe class imbalance, missing measurements encoded as `-999`, and complex non-linear feature interactions.

## Project Structure

```text
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
```

## Data Exploration

Exploratory analysis was conducted to understand the structure and characteristics of the dataset. This included examining feature distributions, identifying class imbalance between signal and background events, and analysing the presence of encoded missing values (`-999`), particularly in jet-related variables.

Correlation analysis on subsets of numerical features revealed complex interdependencies, motivating the use of ensemble tree-based models capable of capturing non-linear relationships.

Visual outputs include class distribution plots, feature histograms, and correlation heatmaps.

## Preprocessing

The preprocessing pipeline was designed to prepare the data for robust and fair model training. Target labels were encoded numerically (`b → 0`, `s → 1`). Encoded missing values (`-999`) were replaced with `NaN` and imputed using the median of each feature to reduce sensitivity to skewed distributions and outliers.

Non-numeric features were removed, and the dataset was split into training and test sets using a stratified 80/20 split to preserve class proportions. Feature scaling using `StandardScaler` was applied where required, and class imbalance was addressed using class weighting strategies.

All processed datasets were saved to disk to ensure reproducibility and consistent use across models.

## Models Implemented

### Logistic Regression

Logistic Regression was implemented as a baseline classifier due to its simplicity and interpretability. The model was trained on scaled features and incorporated balanced class weights to address class imbalance. While computationally efficient, its linear decision boundary limits its ability to model complex interactions.

### Random Forest

Random Forest was used as an ensemble learning method based on bagging. By combining multiple decision trees, the model effectively captures non-linear feature interactions and demonstrates strong robustness to noise and overfitting. Feature importance analysis provides additional interpretability.

### XGBoost

XGBoost, a gradient boosting algorithm, was employed to further improve performance. The model was optimised for imbalanced classification using `scale_pos_weight` and trained with regularisation and subsampling strategies. Among all models, XGBoost achieved the strongest overall performance.

## Evaluation Metrics

Given the class imbalance, evaluation focused on precision, recall, F1-score, and ROC-AUC rather than accuracy alone. Model behaviour was further analysed using confusion matrices, ROC curves, precision–recall curves, and cross-model ROC comparisons.

## Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|--------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.9619   | 0.9078    | 0.9891 | 0.9467   | 0.9946  |
| Random Forest       | 0.9995   | 0.9999    | 0.9986 | 0.9993   | ~1.000  |
| XGBoost             | 0.99998  | 0.99995   | 0.99998| 0.99996  | ~1.000  |

The ensemble-based models significantly outperform the linear baseline, demonstrating their ability to learn complex, physics-driven relationships within the data.


## Key Findings

Logistic Regression provides strong recall but limited expressive capacity. Random Forest and XGBoost achieve near-perfect classification performance, highlighting the effectiveness of ensemble methods for rare-event detection. The results also emphasise the importance of careful preprocessing and appropriate evaluation metrics.

## Technologies Used

Python, NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook, Git, and GitHub.

## Notes on Generalisation

Although ensemble models achieved near-perfect scores on the test set, additional validation using cross-validation or independent test datasets would be required in real experimental settings to fully assess generalisation performance.

## Author

Bishnu Parajuli  
Final Year AI Coursework Project

Higgs Boson Event Detection Using Machine Learning
Overview

This project applies machine learning techniques to the problem of Higgs boson event detection using simulated collision data from the ATLAS Higgs Machine Learning Challenge (2014). The objective is to distinguish rare Higgs (signal) events from dominant background processes by learning patterns from high-dimensional physics features.

The work follows a complete end-to-end machine learning pipeline, including data exploration, preprocessing, model training, evaluation, cross-validation, hyperparameter tuning, and deployment-oriented extension.

Dataset

Source: ATLAS Higgs Machine Learning Challenge 2014 (CERN Open Data)

Approximately 818,000 simulated proton–proton collision events

30 numerical physics-derived features

Target classes:

s → Higgs (signal)

b → Background

Key challenges associated with the dataset include severe class imbalance, missing measurements encoded as -999, and complex non-linear feature interactions.

Project Structure
Higgs-Boson/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_random_forest.ipynb
│   ├── 05_xgboost.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_cross_validation.ipynb
│   ├── 08_hyperparameter_tuning_logistic_regression.ipynb
│   └── 09_hyperparameter_tuning_random_forest_and_xgboost.ipynb
│
├── Higgs_CLI/
│   ├── predict.py
│   ├── xgb_model.pkl
│   ├── feature_names.pkl
│   └── sample_data/
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

Exploratory analysis was conducted to understand the structure and characteristics of the dataset. This included examining feature distributions, identifying class imbalance between signal and background events, and analysing the presence of encoded missing values (-999), particularly in jet-related variables.

Correlation analysis on subsets of numerical features revealed complex interdependencies, motivating the use of ensemble tree-based models capable of capturing non-linear relationships.

Visual outputs include class distribution plots, feature histograms, and correlation heatmaps.

Preprocessing

The preprocessing pipeline was designed to prepare the data for robust and fair model training. Target labels were encoded numerically (b → 0, s → 1). Encoded missing values (-999) were replaced with NaN and imputed using the median of each feature to reduce sensitivity to skewed distributions and outliers.

Non-numeric features were removed, and the dataset was split into training and test sets using a stratified 80/20 split to preserve class proportions. Feature scaling using StandardScaler was applied where required, and class imbalance was addressed using class weighting strategies.

All processed datasets were saved to disk to ensure reproducibility and consistent use across models.

Models Implemented
Logistic Regression

Logistic Regression was implemented as a baseline classifier due to its simplicity and interpretability. The model was trained on scaled features and incorporated balanced class weights to address class imbalance. While computationally efficient, its linear decision boundary limits its ability to model complex interactions.

Random Forest

Random Forest was used as an ensemble learning method based on bagging. By combining multiple decision trees, the model effectively captures non-linear feature interactions and demonstrates strong robustness to noise and overfitting. Feature importance analysis provides additional interpretability.

XGBoost

XGBoost, a gradient boosting algorithm, was employed to further improve performance. The model was optimised for imbalanced classification using scale_pos_weight and trained with regularisation and subsampling strategies. Among all models, XGBoost achieved the strongest overall performance.

Evaluation Metrics

Given the class imbalance, evaluation focused on precision, recall, F1-score, and ROC-AUC rather than accuracy alone. Model behaviour was further analysed using confusion matrices, ROC curves, precision–recall curves, and cross-model ROC comparisons.

Cross-Validation

To evaluate robustness beyond a single train–test split, stratified 5-fold cross-validation was performed on the training data. Stratification ensured that each fold preserved the original class distribution between Higgs (signal) and background events.

ROC-AUC was used as the primary evaluation metric due to its threshold-independent nature and suitability for imbalanced classification. Cross-validation results showed extremely stable performance across folds with negligible variance, indicating strong generalization and reducing the likelihood that near-perfect results were caused by a favourable data split.

Hyperparameter Tuning

Hyperparameter tuning was conducted to confirm model stability rather than to aggressively optimise performance. RandomizedSearchCV was used due to its computational efficiency on large datasets.

Logistic Regression tuning identified an optimal regularization strength of C = 100.0.

Random Forest tuning confirmed that near-default structural parameters were sufficient.

XGBoost tuning produced parameters closely aligned with the original configuration.

Cross-validated ROC-AUC scores remained near-saturated, indicating that ensemble models were robust and not sensitive to small hyperparameter variations.

Model Performance Summary
Model	Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.9619	0.9078	0.9891	0.9467	0.9946
Random Forest	0.9995	0.9999	0.9986	0.9993	~1.000
XGBoost	0.99998	0.99995	0.99998	0.99996	~1.000

The ensemble-based models significantly outperform the linear baseline, demonstrating their ability to learn complex, physics-driven relationships within the data.

Practical Application: Command-Line Interface (CLI)

To demonstrate real-world applicability, a lightweight command-line interface (CLI) was developed using the trained XGBoost model. The CLI allows users to provide a CSV file of event-level features and returns predicted Higgs probabilities and binary classifications.

The CLI includes feature validation, consistent preprocessing, configurable decision thresholds, and CSV-based outputs, illustrating how research models can be transitioned into deployable tools.

Key Findings

Logistic Regression provides strong recall but limited expressive capacity. Random Forest and XGBoost achieve near-perfect classification performance, highlighting the effectiveness of ensemble methods for rare-event detection. The results also emphasise the importance of careful preprocessing, robust evaluation, and validation strategies.

Technologies Used

Python, NumPy, Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn, Jupyter Notebook, Git, and GitHub.

Notes on Generalisation

Although ensemble models achieved near-perfect scores on both test sets and cross-validation, further validation using independent datasets or real detector data would be required in practical experimental environments to fully assess generalisation performance.

Author

Bishnu Parajuli
Final Year AI Coursework Project

License

This project is intended for academic and educational purposes only.

## License

This project is intended for academic and educational purposes only.
