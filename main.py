import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score
from Smote import Smote
from SMOTEENN import SMOTEENN
from sklearn.datasets import load_breast_cancer
from sklearn.utils import check_X_y
import numpy as np
from imblearn.under_sampling import EditedNearestNeighbours


def get_minority_class_samples(data, target):
    """Returns the samples belonging to the minority class in a dataset.

    Args:
        data: The input data matrix (features).
        target: The target vector (classes).

    Returns:
        A tuple containing:
            - The minority class samples (subset of the input data).
            - The corresponding target values for the minority class samples.
    """

    X, y = check_X_y(data, target)

    unique_classes, counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]

    mask = y == minority_class
    minority_samples = X[mask]
    minority_targets = y[mask]

    return minority_samples, minority_targets


data = load_breast_cancer()
X = data.data
y = data.target


smoteenn = SMOTEENN(X, y, 200, 3, "not majority")
print(len(smoteenn.resample_x_train()))
print(len(X))


# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)


# # Apply SMOTE only to the minority class samples
# minority_samples, minority_targets = get_minority_class_samples(X_train, y_train)
#
# smote = Smote(minority_samples, over_sample_percentage=100, n_nearest_neighbors=2)
#
# X_oversampled = smote.get_synthetic_samples()
# y_oversampled = pd.Series(minority_targets[0]).repeat(smote.get_created_sample_count())
#
# # Combine oversampled minority samples with original majority samples
# X_train_smote = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_oversampled)], ignore_index=True)
# y_train_smote = pd.concat([pd.DataFrame(y_train), y_oversampled], ignore_index=True)
#
# # Train a model with the oversampled data
# model_smote = LogisticRegression()
# model_smote.fit(X_train_smote, y_train_smote)
# y_pred_smote = model_smote.predict(X_test)
#
# normal_model = LogisticRegression()
# normal_model.fit(X_train, y_train)
# y_pred_normal = normal_model.predict(X_test)
#
# # Evaluate performance metrics on the Smoted dataset
# accuracy_smote = accuracy_score(y_test, y_pred_smote)
# f1_smote = f1_score(y_test, y_pred_smote)
#
# # Evaluate performance metrics on the normal dataset
# accuracy_normal = accuracy_score(y_test, y_pred_normal)
# f1_normal = f1_score(y_test, y_pred_normal)
#
# print("Accuracy with SMOTE (minority class only):", accuracy_smote)
# print("F1-score with SMOTE (minority class only):", f1_smote)
#
# print("Accuracy with Normal:", accuracy_normal)
# print("F1-score with Normal:", f1_normal)
#
#
# # Evaluate model on the original test set (without SMOTE)
# y_true = y_test
# y_pred = y_pred_smote  # Use predictions from the model trained with SMOTE
#
# # Generate ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_true, y_pred)
# roc_auc = roc_auc_score(y_true, y_pred)
#
# print("Number of Minority Class targets: ", len(minority_targets), "\nNumber of Minority Class samples: ", len(minority_samples))
# print("Number of targets after oversampling: ", len(y_oversampled), "\nNumber of samples after oversampling: ", len(X_oversampled))
# print("Number of smote_y_train: ", len(y_train_smote), "\nNumber of smote_x_train: ", len(X_train_smote))