#!/usr/bin/env python
# coding: utf-8

# In[3]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
)

# Load the dataset
credit_card_data = pd.read_csv('C:\\Users\\AVLN RAGHURAM\\Downloads\\archive (1)\\creditcard.csv')

# Initial exploration
print("Dataset Overview:")
print(credit_card_data.info())

# Check for missing values
print("\nMissing Values:")
print(credit_card_data.isnull().sum())

# Data distribution in the target variable
print("\nClass Distribution:")
print(credit_card_data['Class'].value_counts())

# Visualizing the class imbalance
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=credit_card_data)
plt.title('Class Distribution')
plt.xlabel('Class (0 = Legit, 1 = Fraud)')
plt.ylabel('Count')
plt.show()

# Descriptive statistics of transaction amounts
print("\nTransaction Amount Descriptions:")
print("Legit Transactions:")
print(credit_card_data[credit_card_data.Class == 0]['Amount'].describe())
print("\nFraudulent Transactions:")
print(credit_card_data[credit_card_data.Class == 1]['Amount'].describe())

# Under-sampling for balanced dataset
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

legit_sample = legit.sample(n=len(fraud))  # Sample an equal number of legit transactions
balanced_data = pd.concat([legit_sample, fraud], axis=0)

# Visualize the new balanced class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=balanced_data)
plt.title('Balanced Class Distribution')
plt.xlabel('Class (0 = Legit, 1 = Fraud)')
plt.ylabel('Count')
plt.show()

# Feature-target split
X = balanced_data.drop(columns='Class', axis=1)
Y = balanced_data['Class']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
print("\nDataset Shapes:")
print(f"X: {X.shape}, X_train: {X_train.shape}, X_test: {X_test.shape}")

# Logistic Regression model
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X_train, Y_train)

# Training set evaluation
train_predictions = model.predict(X_train)
train_accuracy = accuracy_score(Y_train, train_predictions)
print("\nTraining Set Evaluation:")
print(f"Accuracy: {train_accuracy:.2f}")
print(classification_report(Y_train, train_predictions))

# Test set evaluation
test_predictions = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, test_predictions)
precision = precision_score(Y_test, test_predictions)
recall = recall_score(Y_test, test_predictions)
f1 = f1_score(Y_test, test_predictions)
auc = roc_auc_score(Y_test, model.predict_proba(X_test)[:, 1])

print("\nTest Set Evaluation:")
print(f"Accuracy: {test_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"AUC-ROC: {auc:.2f}")
print(classification_report(Y_test, test_predictions))

# Visualizing ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(Y_test, model.predict_proba(X_test)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0, 1], [0, 1], 'r--')  # Random classifier line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# In[ ]:




