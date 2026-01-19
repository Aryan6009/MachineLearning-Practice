#Model Evaluation
# This module provides functions to evaluate machine learning models using various metrics.
# It includes functions for calculating accuracy, precision, recall, F1-score, and generating confusion matrices.

"""Model Evaluation Module
Available Functions:
- accuracy_score(y_true, y_pred): Calculate the accuracy of predictions.
where accuracy is the ratio of correctly predicted observations to the total observations.

- precision_score(y_true, y_pred): Calculate the precision of predictions.
where precision is the ratio of correctly predicted positive observations to the total predicted positive observations.

- recall_score(y_true, y_pred): Calculate the recall of predictions.
where recall is the ratio of correctly predicted positive observations to all actual positive observations
-How many real positive cases we were able to predict correctly. 

- f1_score(y_true, y_pred): Calculate the F1-score of predictions.
where F1-score is the weighted average of Precision and Recall.

- confusion_matrix(y_true, y_pred): Generate a confusion matrix.
"""

# Import necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# -------------------------------------------------------------

# Example usage:

# True labels
y_true = [0, 1, 1, 0, 1, 1, 0]
# Predicted labels
y_pred = [0, 1, 0, 0, 1, 1, 1]
# Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

