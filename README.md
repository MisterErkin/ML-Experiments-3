# ML-Experiments 3 — Fraud Detection

**Goal:** Binary fraud detection with modular preprocessing, optional sampling, and multiple classifiers.  
**Dataset:** `card_transdata.csv` • Target: `fraud` with positive label `1.0` • Positive rate ≈ 8.71% (sample)

## Setup
- Metric focus: **pr_auc**
- Split: test_size = **0.2**, random_state = **42**
- Preprocessing: numeric→ median impute (+ missingness indicator), scaler = **True**; categorical→ most_frequent impute + OneHotEncoder(ignore unknown)
- Imbalance handling: **class_weight**
- Models trained: LogisticRegression, RandomForest, GradientBoosting, KNN, DecisionTree

## Results
**Best model:** **rf** by **pr_auc = 1.0000**

### Per-model summary
(from `summary.tsv`)
| model | accuracy | precision | recall | f1 | pr_auc | roc_auc |
|---|---:|---:|---:|---:|---:|---:|
| logreg | 0.9348 | 0.5773 | 0.9479 | 0.7176 | 0.7574 | 0.9795 |
| rf | 1.0000 | 1.0000 | 0.9997 | 0.9999 | 1.0000 | 1.0000 |
| gb | 0.9996 | 0.9997 | 0.9954 | 0.9975 | 1.0000 | 1.0000 |
| knn | 0.9987 | 0.9957 | 0.9895 | 0.9926 | 0.9995 | 0.9998 |
| dt | 1.0000 | 0.9999 | 0.9999 | 0.9999 | 0.9998 | 0.9999 |

### Timings
(from `timings.json`)
| model | fit_seconds | predict_seconds |
|---|---:|---:|
| logreg | 0.5960 | 0.0030 |
| rf | 32.5240 | 0.3430 |
| gb | 200.5700 | 0.2490 |
| knn | 2.7440 | 46.3770 |
| dt | 2.5970 | 0.0120 |

### Per-model confusion matrices
**Decisiontree**

![](/mnt/data/assets/decisiontree_cm.png)

**Gradientboosting**

![](/mnt/data/assets/gradientboosting_cm.png)

**Knn**

![](/mnt/data/assets/knn_cm.png)

**Logisticregression**

![](/mnt/data/assets/logisticregression_cm.png)

**Randomforest**

![](/mnt/data/assets/randomforest_cm.png)

## Repro / Config
See `config.json` in the session folder for exact options used.
