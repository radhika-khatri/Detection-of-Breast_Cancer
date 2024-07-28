import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data'
df = pd.read_csv(url, header=None)

# Replace '?' with NaN values
df = df.replace('?', pd.NA)

# Drop rows containing NaN values
df = df.dropna()

# Split the dataset into features and target
X = df.iloc[:, 2:]
y = df.iloc[:, 1]

# Encode target labels (assuming 'R' for recurrence-events and 'N' for non-recurrence-events)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Create a Support Vector Machine Classifier object
svc = SVC(kernel='linear', random_state=42)

# Train the model on the training set
svc.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = svc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the accuracy score of the model on the test set
print(f"Accuracy: {accuracy}")

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
