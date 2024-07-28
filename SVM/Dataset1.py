import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#url = "https://raw.githubusercontent.com/Jean-njoroge/Breast-cancer-risk-prediction/master/data/clean-data.csv"
url = "https://raw.githubusercontent.com/Jean-njoroge/Breast-cancer-risk-prediction/master/data/clean-data.csv"
df = pd.read_csv(url)

X = df.drop('diagnosis', axis=1).values
y = df['diagnosis'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm = SVC(random_state=42)

svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate precision, recall, and F1 score
precision = precision_score(y_test, y_pred, pos_label='M')
recall = recall_score(y_test, y_pred, pos_label='M')
f1 = f1_score(y_test, y_pred, pos_label='M')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)