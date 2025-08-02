import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/Acer/OneDrive/Desktop/Assignment/Fraud.csv')  # Replace with your actual file path
print(df.head())
print(df.info())

# Drop nameOrig and nameDest (high cardinality)
df.drop(['nameOrig', 'nameDest'], axis=1, inplace=True)

# One-hot encode 'type'
df = pd.get_dummies(df, columns=['type'])

# Create derived features
df['balanceChangeOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceChangeDest'] = df['newbalanceDest'] - df['oldbalanceDest']

# Fill missing merchant balances with 0
df[['oldbalanceDest', 'newbalanceDest']] = df[['oldbalanceDest', 'newbalanceDest']].fillna(0)

# Define features and target
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

importances = model.feature_importances_
features = X.columns
sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()