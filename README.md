# Vehicle Data Preprocessing & Machine Learning

A Python-based machine learning project that demonstrates **data preprocessing, categorical encoding, normalization, data splitting, regression, and classification** using a vehicle dataset.

The project applies multiple encoding techniques to categorical features and evaluates different machine learning models for predicting vehicle prices and classifying vehicles based on price.

---

## 📌 Project Overview

This project uses a vehicle dataset (`vehicles.csv`) and processes the first **1,000 records**.

The following features are used:

* `price`
* `year`
* `odometer`
* `fuel`
* `transmission`
* `manufacturer`
* `condition`

Missing values are removed before preprocessing.

---

## 🛠️ Technologies & Libraries

* Python
* Pandas
* Scikit-learn
* Category Encoders

### Main Libraries

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
    MinMaxScaler
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import category_encoders as ce
```

---

## 🔄 Data Preprocessing

The project demonstrates several categorical encoding techniques.

### 1. Label Encoding

Label Encoding is applied to:

* `fuel`
* `transmission`
* `manufacturer`
* `condition`

Each categorical value is converted into a numerical label.

---

### 2. Ordinal Encoding

`OrdinalEncoder` is used to convert categorical features into numerical values.

```python
oe = OrdinalEncoder()
```

---

### 3. One-Hot Encoding

One-Hot Encoding converts categorical values into separate binary columns.

The project uses:

```python
OneHotEncoder(sparse_output=False, drop='first')
```

The resulting dataset contains **44 columns** after encoding.

---

### 4. Count Encoding

Count Encoding replaces each categorical value with its frequency in the dataset.

```python
ce.CountEncoder()
```

---

### 5. Target Encoding

Target Encoding replaces categorical values based on the mean target value (`price`).

```python
ce.TargetEncoder(cols=categorical_features)
```

---

## 📏 Normalization

`MinMaxScaler` is used to normalize:

* `price`
* `year`
* `odometer`

The values are scaled to a common range before training the machine learning models.

---

## ✂️ Train/Test Split

The processed dataset is divided into training and testing sets.

```python
test_size = 0.2
random_state = 42
```

80% of the data is used for training and 20% for testing.

---

# 🤖 Machine Learning Models

The project implements four machine learning approaches.

## 1. Linear Regression

Linear Regression is used to predict the vehicle `price`.

### Evaluation Metric

**R² Score**

### Result

```text
Linear Regression R² Score: 0.3390121122476535
```

---

## 2. Logistic Regression

The vehicle price is converted into a binary classification problem.

A vehicle is classified according to whether its price is above or below the median price.

### Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report

### Result

```text
Logistic Regression Accuracy: 0.7009345794392523
```

### Confusion Matrix

```text
[[36 12]
 [20 39]]
```

---

## 3. KNN Regression

`KNeighborsRegressor` is used to predict vehicle prices.

```python
KNeighborsRegressor(n_neighbors=5)
```

### Result

```text
KNN Regression R² Score: 0.48792646468291934
```

---

## 4. KNN Classification

`KNeighborsClassifier` is used for binary vehicle price classification.

```python
KNeighborsClassifier(n_neighbors=5)
```

### Result

```text
KNN Classification Accuracy: 0.7850467289719626
```

### Confusion Matrix

```text
[[35 13]
 [10 49]]
```

### Classification Report

```text
              precision    recall  f1-score   support

0                 0.78      0.73      0.75        48
1                 0.79      0.83      0.81        59

accuracy                              0.79       107
macro avg         0.78      0.78      0.78       107
weighted avg      0.78      0.79      0.78       107
```

---

# 📊 Model Performance

| Model               | Task             | Metric   | Result |
| ------------------- | ---------------- | -------- | -----: |
| Linear Regression   | Price Prediction | R²       | 0.3390 |
| Logistic Regression | Classification   | Accuracy | 0.7009 |
| KNN Regression      | Price Prediction | R²       | 0.4879 |
| KNN Classification  | Classification   | Accuracy | 0.7850 |

Based on the current results, **KNN Classification achieved the highest classification accuracy (78.5%)**, while **KNN Regression achieved the highest R² score (0.488)** among the regression models tested.

---

## 📁 Project Structure

```text
Vehicle-ML-Project/
│
├── main.py
├── vehicles.csv
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
scikit-learn
category-encoders
```

---

## ▶️ Run the Project

Make sure `vehicles.csv` is located in the same directory as `main.py`.

Then run:

```bash
python main.py
```

The program will display:

* Original dataset
* Label Encoding results
* Ordinal Encoding results
* One-Hot Encoding results
* Count Encoding results
* Target Encoding results
* Linear Regression R² Score
* Logistic Regression results
* KNN Regression R² Score
* KNN Classification results

---

## 🎯 Learning Objectives

This project demonstrates practical usage of:

* Data loading with Pandas
* Data cleaning
* Handling missing values
* Categorical feature encoding
* Label Encoding
* Ordinal Encoding
* One-Hot Encoding
* Count Encoding
* Target Encoding
* Feature normalization
* Train/Test splitting
* Regression
* Classification
* Model evaluation
* Confusion matrices
* Classification reports
* K-Nearest Neighbors

---

## 👨‍💻 Author

**Engineer**

GitHub: `https://github.com/sar-3mar`

---

## 📄 License

This project is created for educational and learning purposes.
