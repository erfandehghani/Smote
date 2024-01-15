from sklearn.model_selection import train_test_split
from SMOTEENN import SMOTEENN
from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load the dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target

# Check for class imbalance
print(pd.Series(y).value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smoteenn = SMOTEENN(X_train, y_train, over_sample_percentage=200, n_nearest_neighbors=3)
X_resampled, y_resampled = smoteenn.resample()

from sklearn.linear_model import LogisticRegression

# Train a classifier without resampling
clf_no_resample = LogisticRegression()
clf_no_resample.fit(X_train, y_train)

# Train a classifier with resampled data
clf_resampled = LogisticRegression()
clf_resampled.fit(X_resampled, y_resampled)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Evaluate performance for both models
y_pred_no_resample = clf_no_resample.predict(X_test)
y_pred_resampled = clf_resampled.predict(X_test)

print("Accuracy (no resampling):", accuracy_score(y_test, y_pred_no_resample))
print("Accuracy (resampled):", accuracy_score(y_test, y_pred_resampled))

print("Precision (no resampling):", precision_score(y_test, y_pred_no_resample))
print("Precision (resampled):", precision_score(y_test, y_pred_resampled))

print("Recall (no resampling):", recall_score(y_test, y_pred_no_resample))
print("Recall (resampled):", recall_score(y_test, y_pred_resampled))

print("F1-score (no resampling):", f1_score(y_test, y_pred_no_resample))
print("F1-score (resampled):", f1_score(y_test, y_pred_resampled))

print("ROC AUC (no resampling):", roc_auc_score(y_test, y_pred_no_resample))
print("ROC AUC (resampled):", roc_auc_score(y_test, y_pred_resampled))
