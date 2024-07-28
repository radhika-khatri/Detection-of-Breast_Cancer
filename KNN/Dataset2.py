import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read the dataset, assuming it's a CSV file with a header row
data = pd.read_csv("/content/breast-cancer.data", header=None)

# Define the columns that contain categorical data
categorical_columns = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Perform one-hot encoding for the categorical columns
data = pd.get_dummies(data, columns=categorical_columns)

# Split the dataset into features (X) and the target variable (y)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the K-Nearest Neighbors (KNN) classifier
knn_classifier = KNeighborsClassifier()

# Fit the KNN classifier on the training data
knn_classifier.fit(X_train, y_train)

# Predict on the test data
y_pred = knn_classifier.predict(X_test)

# Calculate accuracy on the test data
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1-score using "weighted" averaging
precision_weighted = precision_score(y_test, y_pred, average='weighted')
recall_weighted = recall_score(y_test, y_pred, average='weighted')
f1_weighted = f1_score(y_test, y_pred, average='weighted')

print(f"The accuracy on the test data is {accuracy * 100:.2f}%")
print(f"Weighted Precision: {precision_weighted:.2f}")
print(f"Weighted Recall: {recall_weighted:.2f}")
print(f"Weighted F1-score: {f1_weighted:.2f}")