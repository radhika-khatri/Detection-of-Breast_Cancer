import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load the dataset (replace with your dataset URL)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data"

# Load the dataset without headers (no header line)
df = pd.read_csv(url, header=None, na_values="?")

# Extract the Outcome column (second column) for binary classification
y = df.iloc[:, 1].values

# Convert boolean labels to integers (0 or 1)
y = (y == 'R').astype(int)

# Extract all other columns except the first one (ID) and the second one (Outcome)
X = df.iloc[:, 2:].values

# Data preprocessing
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Data training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode labels if needed
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train_encoded)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Create and train the Decision Tree model with the best hyperparameters
decision_tree_classifier = DecisionTreeClassifier(random_state=42, **best_params)
decision_tree_classifier.fit(X_train, y_train_encoded)

# Predict using the Decision Tree model
y_pred = decision_tree_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print("Accuracy:", accuracy)

# Confusion matrix and classification report
cm = confusion_matrix(y_test_encoded, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test_encoded, y_pred))

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Nonrecur', 'Recur'])
plt.yticks(tick_marks, ['Nonrecur', 'Recur'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# Predict using the Decision Tree model
y_pred = decision_tree_classifier.predict(X_test)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test_encoded, y_pred)
recall = recall_score(y_test_encoded, y_pred)
f1 = f1_score(y_test_encoded, y_pred)

# Print the results
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
