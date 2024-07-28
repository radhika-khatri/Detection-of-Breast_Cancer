import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the data from the file
data = pd.read_csv("/content/breast-cancer.data", header=None)
categorical_columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Perform one-hot encoding for the categorical columns
data = pd.get_dummies(data, columns=categorical_columns)

# Split the dataset into features (X) and the target variable (y)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Encode the target labels ('M' for malignant, 'B' for benign) to numerical values (1 and 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Cross-validation to evaluate the model
scores = cross_val_score(dt_classifier, X_train, y_train, scoring='accuracy', cv=10)
mean_accuracy = np.mean(scores)
print(f"The mean accuracy with 10-fold cross-validation is {mean_accuracy:.2f}")

# Training the Decision Tree classifier
dt_classifier.fit(X_train, y_train)

# Predicting on the test set
y_pred = dt_classifier.predict(X_test)

# Calculating accuracy on the test data
acc_test = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"The accuracy on test data is {acc_test:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
