# Bank Marketing Classification — scikit-learn ML Pipeline

An end-to-end scikit-learn / imbalanced-learn classification pipeline for predicting a binary target (Type A vs Type B) from a bank-marketing-style dataset. Covers missing-value handling, outlier capping, feature selection, class-imbalance correction with SMOTE, and a grid-searched comparison across five classifiers.

**Data**: A modified/anonymised variant of the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing), provided as part of TU059 coursework (job categories relabelled to `JobCat1`, `JobCat2`, etc., target classes renamed to `TypeA`/`TypeB`). Not redistributed here — see `.gitignore`.

## Results

| Model | Test Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| KNN | 88.41% | 52.78% | 23.85% | 0.325 |
| Random Forest | 85.94% | 41.34% | 43.83% | **0.426** (best F1) |
| Naive Bayes | 84.82% | 37.07% | 39.81% | 0.384 |
| Logistic Regression | 76.84% | 27.19% | **56.48%** (best recall) | 0.367 |
| Gradient Boosting | 88.03% | **49.44%** (best precision) | 33.95% | 0.403 |

Class imbalance in the target variable made accuracy a misleading metric on its own — high-accuracy models like KNN achieved that by under-predicting the minority class. **Random Forest gave the best precision/recall balance**, with the highest F1 score (0.428) among all five models — beating Naive Bayes (F1 ≈ 0.384) despite the latter's slightly higher raw recall. Logistic Regression maximised recall at the cost of precision; Gradient Boosting maximised precision at the cost of recall. Model choice in practice should depend on whether false positives or false negatives are more costly for the use case.

## Pipeline overview

1. **Data loading & inspection** — null-value audit, plus identification of `"unknown"` as a disguised missing-value marker across several categorical columns (not caught by a standard null check).
2. **Cleaning**
   - `duration` dropped (constant/zero-only column, no signal)
   - `"unknown"` values imputed with each column's mode
   - `poutcome` retained despite a high rate of unknowns — empirically improved accuracy, precision, and recall when included vs. excluded
3. **Preprocessing pipeline** (`ColumnTransformer`)
   - Numerical: custom `OutlierCapper` (1st/99th percentile clipping) → median imputation → standard scaling
   - Categorical: one-hot encoding
4. **Feature selection** — `SelectKBest` (top 20, F-test) followed by `RFE` (top 10, logistic regression estimator) for a two-stage reduction
5. **Class imbalance** — SMOTE oversampling of the minority class within the pipeline (applied only to training folds, avoiding test-set leakage)
6. **Model comparison** — KNN, Random Forest, Naive Bayes, Logistic Regression, and Gradient Boosting (`HistGradientBoostingClassifier`), each tuned via `GridSearchCV` (5-fold, scored on accuracy)
7. **Evaluation** — accuracy, precision, recall, F1, and ROC/AUC compared across all five models
8. **Inference** — best pipeline applied to a held-out query set, predictions written to a results file

## Tech stack

- scikit-learn — pipelines, preprocessing, feature selection, grid search, metrics
- imbalanced-learn — SMOTE, pipeline integration
- pandas / numpy — data handling
- matplotlib / seaborn — exploratory plots, boxplots, ROC curves

## Possible next steps

- Investigate whether the residual class imbalance after SMOTE is still suppressing recall/precision across models
- Score grid search on F1 or recall rather than accuracy, given the imbalance
- Try threshold tuning on the chosen model's predicted probabilities rather than the default 0.5 cutoff

## Running it

```bash
pip install -r requirements.txt
```

Requires `trainingset.txt` and `queries.txt` in the working directory (not included in this repo — see `.gitignore`). Run the notebook top to bottom; final predictions are written to `D24126843.txt`.

---
*Built as part of an MSc Data Science assignment on machine learning pipelines.*
