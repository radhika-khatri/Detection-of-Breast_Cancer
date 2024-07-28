import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv("https://raw.githubusercontent.com/Jean-njoroge/Breast-cancer-risk-prediction/master/data/clean-data.csv")

# Map diagnosis labels to binary values
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and labels
X = data.drop('diagnosis', axis=1).values
Y = data['diagnosis'].values

# Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=18)
rf_classifier.fit(X_train, Y_train)

# Predict the test set results
y_pred_rf = rf_classifier.predict(X_test)

# Calculate precision, recall, F1 score, and accuracy for Random Forest
precision_rf = precision_score(Y_test, y_pred_rf)
recall_rf = recall_score(Y_test, y_pred_rf)
f1_rf = f1_score(Y_test, y_pred_rf)
accuracy_rf = accuracy_score(Y_test, y_pred_rf)

# Print the metrics
print("\nRandom Forest Metrics:")
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1 Score:", f1_rf)
print("Accuracy:", accuracy_rf)
