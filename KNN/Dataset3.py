import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wpbc.data'
df = pd.read_csv(url, header=None)

# Replace '?' with NaN values
df = df.replace('?', pd.NaT)

# Drop rows containing NaN values
df = df.dropna()

# Encode the target labels ('N' for negative and 'R' for positive) to numerical values (0 and 1)
label_encoder = LabelEncoder()
df[1] = label_encoder.fit_transform(df[1])  # Assuming column 1 contains labels

# Split the dataset into features and target
X = df.iloc[:, 2:]
y = df.iloc[:, 1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Upsample the minority class (positive class) to address class imbalance
X_train_upsampled, y_train_upsampled = resample(X_train[y_train == 1], y_train[y_train == 1], replace=True, n_samples=sum(y_train == 0), random_state=42)
X_train_balanced = pd.concat([X_train[y_train == 0], X_train_upsampled])
y_train_balanced = pd.concat([y_train[y_train == 0], y_train_upsampled])

# Create a K-Nearest Neighbors Classifier object
knn = KNeighborsClassifier()

# Hyperparameter tuning using GridSearchCV
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_balanced, y_train_balanced)

# Get the best K-Nearest Neighbors model
best_knn = grid_search.best_estimator_

# Predict the target variable for the test set
y_pred = best_knn.predict(X_test)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")
