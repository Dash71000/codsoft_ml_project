# -*- coding: utf-8 -*-
"""customer_churn_pred_codsoft.ipynb


"""

import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import kagglehub

path = kagglehub.dataset_download("shantanudhakadd/bank-customer-churn-prediction")

print("Path to dataset files:", path)

# Dataset directory
df_dir = "/kaggle/input/bank-customer-churn-prediction/Churn_Modelling.csv"

df = pd.read_csv("/content/Churn_Modelling.csv")
display(df.head())
display(df.info())

df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Apply one-hot encoding to 'Geography' and 'Gender'
df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

#  Identify and scale numerical features
# Numerical features are all columns except the target 'Exited' and the newly created dummy variables.
numerical_features = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Display the first few rows and the data types of the processed DataFrame
display(df.head())
display(df.info())

# Define features (X) and target (y)
X = df.drop('Exited', axis=1)
y = df['Exited']

# Split  data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

display(X_train.head())
display(y_train.head())

# Instantiate and train the Logistic Regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_logistic = logistic_model.predict(X_test)

# Evaluate the Logistic Regression model
print("Logistic Regression Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_logistic)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_logistic))
print("Classification Report:")
print(classification_report(y_test, y_pred_logistic))

# Instantiate and train the Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
print("\nRandom Forest Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
