import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

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
svm = SVC(random_state=42)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")