import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
# Importing DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/Jean-njoroge/Breast-cancer-risk-prediction/master/data/clean-data.csv")
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

data_mean = data[['diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]

predictors = data_mean.columns[2:11]
target = "diagnosis"

X = data.drop('diagnosis', axis=1).values
Y = data['diagnosis'].values

# Split the dataset into training and test sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initiating the model:
dt_classifier = DecisionTreeClassifier()

# Training the Decision Tree classifier
dt_classifier.fit(X_train, Y_train)

# Predicting on the test set
predicted = dt_classifier.predict(X_test)

# Calculate precision, recall, and F1 score
precision = precision_score(Y_test, predicted)
recall = recall_score(Y_test, predicted)
f1 = f1_score(Y_test, predicted)

print("Decision Tree Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)