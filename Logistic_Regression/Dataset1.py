import pandas as pd
import sklearn.datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the breast cancer dataset
breast_cancer = sklearn.datasets.load_breast_cancer()
x = breast_cancer.data
y = breast_cancer.target

# Load the dataset from the URL
url = "https://raw.githubusercontent.com/Jean-njoroge/Breast-cancer-risk-prediction/master/data/clean-data.csv"
df = pd.read_csv(url)

# Preprocessing: Separate features and target variable
X = df.drop('diagnosis', axis=1).values
Y = df['diagnosis'].values

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# Encode the target variable
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

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, pos_label=1)
recall = recall_score(y_test, y_pred, pos_label=1)
f1 = f1_score(y_test, y_pred, pos_label=1)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

