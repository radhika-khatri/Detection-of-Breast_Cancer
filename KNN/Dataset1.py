import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/Jean-njoroge/Breast-cancer-risk-prediction/master/data/clean-data.csv")

# Map diagnosis labels to binary values
data['diagnosis'] = data['diagnosis'].map({'M':1,'B':0})

# Separate features and labels
X = data.drop('diagnosis', axis=1).values
Y = data['diagnosis'].values

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, Y_train)

# Predict the test set results
y_pred_knn = knn_classifier.predict(X_test)

# Calculate precision, recall, F1 score, and accuracy for KNN
precision_knn = precision_score(Y_test, y_pred_knn)
recall_knn = recall_score(Y_test, y_pred_knn)
f1_knn = f1_score(Y_test, y_pred_knn)
accuracy_knn = accuracy_score(Y_test, y_pred_knn)

# Print the metrics
print("K-Nearest Neighbors (KNN) Metrics:")
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1 Score:", f1_knn)
print("Accuracy:", accuracy_knn)
