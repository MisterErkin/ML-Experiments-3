# 🧠 ML Pipeline on UCI Adult Income Dataset

**ML-Experiments-3** targets bank-transaction fraud detection with an emphasis on class imbalance and clear, repeatable evaluation. Point it at a CSV (tested with `TestData.csv`), define the positive class, and compare baseline models with consistent preprocessing.

What you get:
- Consistent preprocessing (impute numerics/categoricals, one-hot encode, optionally scale)
- Optional imbalance handling (`class_weight`, `SMOTE`, or undersampling)
- Multiple models trained on the same split for fair comparison
- Precision/Recall, PR-AUC, ROC-AUC, and confusion matrices
- A tidy results folder ready for documentation (`summary.tsv`, plots, timings, config)

**About `TestData.csv`:**  
This sample file contains anonymized transaction records with a mix of numeric and categorical fields (e.g., amounts, timestamps or time-like indices, and ID/category-style columns), plus a binary target indicating whether a transaction is fraudulent. You can swap in any similarly structured dataset by selecting it in the app.

> **Note:** Metrics and rankings will vary across datasets depending on class balance, feature quality, and labeling. Prefer PR-AUC and inspect Precision/Recall to choose an operating threshold that fits your use case.

## 📌 Overview
This project explores the impact of various sampling techniques on classification models trained on the **UCI Adult Income dataset**. The primary goal is to predict whether a person earns more than $50K/year and evaluate how imbalanced data treatments affect performance.

Key features:
- Support for 4 classification models
- Optional LDA dimensionality reduction
- Visualization of distributions, confusion matrices, and result summaries

---

## ⚙️ Environment & Requirements

- **Python version**: `3.12.3`
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### `requirements.txt` includes:
```
pandas==2.2.3
numpy==1.26.4
matplotlib==3.9.0
seaborn==0.13.2
scikit-learn==1.6.1
imbalanced-learn==0.13.0
```
---

## 🧠 Classification Models
- **Logistic Regression** (`logreg`)
- **Random Forest** (`random_forest`)
- **K-Nearest Neighbors** (`knn`)
- **Gradient Boosting** (`gradient_boost`)

> ❌ Support Vector Machine (`svm`) was excluded due to high computation time.

---

## 🔁 Sampling Techniques
To address data imbalance:
- Random Oversampling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Undersampling
- Tomek Links

Each model is trained with all sampling methods and optionally with **LDA**-reduced features.

---

## 🧪 Test Configuration
- `test_size = 0.3`
- `random_state = 66`
- `scaler = MinMaxScaler`
- `apply_lda = True`

---

## 📊 Performance Summary

### Logistic Regression
| Method               | Acc    | Prec   | Recall | F1     | ROC-AUC |
|----------------------|--------|--------|--------|--------|---------|
| Baseline             | 0.8543 | 0.7418 | 0.6059 | 0.6670 | 0.9055  |
| Random Oversampling  | 0.8071 | 0.5661 | 0.8525 | 0.6804 | 0.9058  |
| SMOTE                | 0.8108 | 0.5748 | 0.8236 | 0.6770 | 0.9009  |
| Random Undersampling | 0.8021 | 0.5582 | 0.8537 | 0.6751 | 0.9044  |
| Tomek Links          | 0.8503 | 0.6983 | 0.6662 | 0.6819 | 0.9046  |
| LDA                  | 0.8405 | 0.7153 | 0.5608 | 0.6287 | 0.8925  |

### Random Forest
| Method               | Acc    | Prec   | Recall | F1     | ROC-AUC |
|----------------------|--------|--------|--------|--------|---------|
| Baseline             | 0.8547 | 0.7341 | 0.6220 | 0.6734 | 0.9037  |
| Random Oversampling  | 0.8446 | 0.6758 | 0.6815 | 0.6787 | 0.8999  |
| SMOTE                | 0.8357 | 0.6439 | 0.7105 | 0.6756 | 0.8947  |
| Random Undersampling | 0.8079 | 0.5688 | 0.8350 | 0.6767 | 0.9027  |
| Tomek Links          | 0.8522 | 0.6949 | 0.6884 | 0.6916 | 0.9032  |
| LDA                  | 0.7762 | 0.5346 | 0.5455 | 0.5400 | 0.8163  |

### K-Nearest Neighbors (KNN)
| Method               | Acc    | Prec   | Recall | F1     | ROC-AUC |
|----------------------|--------|--------|--------|--------|---------|
| Baseline             | 0.8246 | 0.6524 | 0.5816 | 0.6150 | 0.8434  |
| Random Oversampling  | 0.7700 | 0.5148 | 0.7751 | 0.6187 | 0.8316  |
| SMOTE                | 0.7892 | 0.5456 | 0.7457 | 0.6301 | 0.8377  |
| Random Undersampling | 0.7679 | 0.5114 | 0.8121 | 0.6276 | 0.8528  |
| Tomek Links          | 0.8201 | 0.6192 | 0.6569 | 0.6375 | 0.8453  |
| LDA                  | 0.8171 | 0.6373 | 0.5574 | 0.5947 | 0.8333  |

### Gradient Boosting
| Method               | Acc    | Prec   | Recall | F1     | ROC-AUC |
|----------------------|--------|--------|--------|--------|---------|
| Baseline             | 0.8683 | 0.7934 | 0.6122 | 0.6911 | 0.9208  |
| Random Oversampling  | 0.8221 | 0.5893 | 0.8614 | 0.6998 | 0.9209  |
| SMOTE                | 0.8243 | 0.5986 | 0.8206 | 0.6923 | 0.9132  |
| Random Undersampling | 0.8179 | 0.5824 | 0.8614 | 0.6949 | 0.9203  |
| Tomek Links          | 0.8678 | 0.7520 | 0.6730 | 0.7103 | 0.9204  |
| LDA                  | 0.8384 | 0.7160 | 0.5446 | 0.6187 | 0.8890  |

---

## 🖼️ Visual Results (Confusion Matrices)
Visual results are stored in the `results/` folder and include all 26 image outputs:
- Distribution plots
- Confusion matrices per model and method
- LDA-reduced model comparisons

### Distribution:
- ![Distribution](results/Distribution_of_Numerical_Features.png)
- ![Income Class](results/Income_Class_Distribution.png)

### Baseline Confusion Matrices:
- ![Logreg](results/Confusion_Matrix_-_logreg.png)
- ![Random Forest](results/Confusion_Matrix_-_random_forest.png)
- ![KNN](results/Confusion_Matrix_-_knn.png)
- ![Gradient Boost](results/Confusion_Matrix_-_gradient_boost.png)

### Sampling + Model Confusion Matrices:

#### Logistic Regression
- ![RO logreg](results/Confusion_Matrix_-_Random_Oversampling_logreg.png)
- ![SMOTE logreg](results/Confusion_Matrix_-_SMOTE_logreg.png)
- ![RU logreg](results/Confusion_Matrix_-_Random_Undersampling_logreg.png)
- ![Tomek logreg](results/Confusion_Matrix_-_Tomek_Links_logreg.png)

#### Random Forest
- ![RO RF](results/Confusion_Matrix_-_Random_Oversampling_random_forest.png)
- ![SMOTE RF](results/Confusion_Matrix_-_SMOTE_random_forest.png)
- ![RU RF](results/Confusion_Matrix_-_Random_Undersampling_random_forest.png)
- ![Tomek RF](results/Confusion_Matrix_-_Tomek_Links_random_forest.png)

#### KNN
- ![RO KNN](results/Confusion_Matrix_-_Random_Oversampling_knn.png)
- ![SMOTE KNN](results/Confusion_Matrix_-_SMOTE_knn.png)
- ![RU KNN](results/Confusion_Matrix_-_Random_Undersampling_knn.png)
- ![Tomek KNN](results/Confusion_Matrix_-_Tomek_Links_knn.png)

#### Gradient Boost
- ![RO GB](results/Confusion_Matrix_-_Random_Oversampling_gradient_boost.png)
- ![SMOTE GB](results/Confusion_Matrix_-_SMOTE_gradient_boost.png)
- ![RU GB](results/Confusion_Matrix_-_Random_Undersampling_gradient_boost.png)
- ![Tomek GB](results/Confusion_Matrix_-_Tomek_Links_gradient_boost.png)

### LDA Confusion Matrices:
- ![LDA logreg](results/Confusion_Matrix_-_LDA_logreg.png)
- ![LDA RF](results/Confusion_Matrix_-_LDA_random_forest.png)
- ![LDA KNN](results/Confusion_Matrix_-_LDA_knn.png)
- ![LDA GB](results/Confusion_Matrix_-_LDA_gradient_boost.png)

---

## 🚀 How to Run
```bash
python Main.py
```
You'll be prompted for:
- Test size (e.g., `0.3`)
- Random state (e.g., `66`)
- Model (`logreg`, `knn`, `random_forest`, `gradient_boost`, or `all`)
- Sampling (`smote`, `undersample`, `tomek`, `random`, or `all`)
- Whether to apply LDA (`True` or `False`)

All outputs will be saved to the `results/` folder.

---

Created by: **Mesut Erkin Özokutgen**