import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Create and train the logistic regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Make predictions on training and testing data
train_pred = lr.predict(X_train)
train_acc = accuracy_score(y_train, train_pred)
print(f'Accuracy on training data: {train_acc}')

test_pred = lr.predict(X_test)
test_acc = accuracy_score(y_test, test_pred)
print(f'Accuracy on testing data: {test_acc}')

# Calculate precision, recall, and F1-score for testing data
precision = precision_score(y_test, test_pred, average='weighted')
recall = recall_score(y_test, test_pred, average='weighted')
f1 = f1_score(y_test, test_pred, average='weighted')

print(f"Weighted Precision: {precision:.2f}")
print(f"Weighted Recall: {recall:.2f}")
print(f"Weighted F1-score: {f1:.2f}")