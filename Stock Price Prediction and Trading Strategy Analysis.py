# -*- coding: utf-8 -*-
"""Princy_ML_Project1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jYIg2Z7Bv3DYlJWnKgUXEHMSMMaKvAIu
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# Define the stock symbol (CRM for Salesforce Inc.)
symbol = "CRM"

# Define the date range (from the beginning of 2015 to the end of last year)
start_date = "2015-01-01"
end_date = "2022-12-31"

# Fetch the historical data
data = yf.download(symbol, start=start_date, end=end_date)

# Calculate the 50-day and 200-day moving averages
data['50_MA'] = data['Close'].rolling(window=50).mean()
data['200_MA'] = data['Close'].rolling(window=200).mean()

# Strategy 1: If the next trading day's close price is greater than today's close price, 'buy', otherwise 'sell'
data['Signal_Strategy_1'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)

# Strategy 2: Golden Cross (50-day MA crosses above 200-day MA) is a 'buy' signal, otherwise 'sell'
data['Signal_Strategy_2'] = 0  # Default to no signal
data.loc[data['50_MA'] > data['200_MA'], 'Signal_Strategy_2'] = 1  # Buy signal (Golden Cross)
data.loc[data['50_MA'] <= data['200_MA'], 'Signal_Strategy_2'] = 0  # Sell signal (Death Cross)

# Drop rows with NaN values due to rolling averages
data.dropna(inplace=True)

# Define feature variables (X) and target variables (y) for Strategy 1
X_strategy_1 = data[['Close']].values
y_strategy_1 = data['Signal_Strategy_1'].values

# Define feature variables (X) and target variables (y) for Strategy 2
X_strategy_2 = data[['50_MA', '200_MA']].values
y_strategy_2 = data['Signal_Strategy_2'].values

# Split data into training and test sets (80/20 split) for Strategy 1
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_strategy_1, y_strategy_1, test_size=0.2, random_state=42)

# Split data into training and test sets (80/20 split) for Strategy 2
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_strategy_2, y_strategy_2, test_size=0.2, random_state=42)

# Initialize classifiers
knn_classifier = KNeighborsClassifier()
rf_classifier = RandomForestClassifier()
gb_classifier = GradientBoostingClassifier()
svm_classifier = SVC()
xgb_classifier = XGBClassifier()

# Train classifiers on the training data for Strategy 1
knn_classifier.fit(X_train_1, y_train_1)
rf_classifier.fit(X_train_1, y_train_1)
gb_classifier.fit(X_train_1, y_train_1)
svm_classifier.fit(X_train_1, y_train_1)
xgb_classifier.fit(X_train_1, y_train_1)

# Predictions for Strategy 1
y_pred_knn_1 = knn_classifier.predict(X_test_1)
y_pred_rf_1 = rf_classifier.predict(X_test_1)
y_pred_gb_1 = gb_classifier.predict(X_test_1)
y_pred_svm_1 = svm_classifier.predict(X_test_1)
y_pred_xgb_1 = xgb_classifier.predict(X_test_1)

# Evaluate classifiers for Strategy 1
print("Strategy 1 Classifier Evaluation:")
print("K-Nearest Neighbors:")
print(classification_report(y_test_1, y_pred_knn_1))  # Classification report for KNN in Strategy 1
print("Random Forest:")
print(classification_report(y_test_1, y_pred_rf_1))   # Classification report for Random Forest in Strategy 1
print("Gradient Boosting:")
print(classification_report(y_test_1, y_pred_gb_1))   # Classification report for Gradient Boosting in Strategy 1
print("Support Vector Machines:")
print(classification_report(y_test_1, y_pred_svm_1))  # Classification report for SVM in Strategy 1
print("XGBoost:")
print(classification_report(y_test_1, y_pred_xgb_1))  # Classification report for XGBoost in Strategy 1

# Initialize KNN classifier for Strategy 2
knn_classifier_2 = KNeighborsClassifier()

# Selecting the 50-day moving average as the feature for KNN in Strategy 2
X_strategy_2_knn = data[['50_MA']].values
y_strategy_2_knn = data['Signal_Strategy_2'].values

# Split data into training and test sets (80/20 split) for KNN in Strategy 2
X_train_2_knn, X_test_2_knn, y_train_2_knn, y_test_2_knn = train_test_split(X_strategy_2_knn, y_strategy_2_knn, test_size=0.2, random_state=42)

# Train KNN classifier on the training data for Strategy 2
knn_classifier_2.fit(X_train_2_knn, y_train_2_knn)

# Predictions for Strategy 2 using KNN
y_pred_knn_2 = knn_classifier_2.predict(X_test_2_knn)

# Evaluate KNN classifier for Strategy 2
print("K-Nearest Neighbors (Strategy 2) Classifier Evaluation:")
print(classification_report(y_test_2_knn, y_pred_knn_2))  # Classification report for KNN in Strategy 2


# Plotting the data for Strategy 1
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='black')
plt.plot(data.index, data['50_MA'], label='50-Day MA', color='blue', linestyle='--')
plt.plot(data.index, data['200_MA'], label='200-Day MA', color='red', linestyle='--')
plt.plot(data[data['Signal_Strategy_1'] == 1].index, data[data['Signal_Strategy_1'] == 1]['Close'], '^', markersize=8, color='g', label='Buy Signal (Strategy 1)')
plt.plot(data[data['Signal_Strategy_1'] == 0].index, data[data['Signal_Strategy_1'] == 0]['Close'], 'v', markersize=8, color='r', label='Sell Signal (Strategy 1)')
plt.title('Salesforce Inc. (CRM) Stock Price and Trading Signals (Strategy 1)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Plotting the data for Strategy 2
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Close Price', color='black')
plt.plot(data.index, data['50_MA'], label='50-Day MA', color='blue', linestyle='--')
plt.plot(data.index, data['200_MA'], label='200-Day MA', color='red', linestyle='--')
plt.plot(data[data['Signal_Strategy_2'] == 1].index, data[data['Signal_Strategy_2'] == 1]['Close'], '^', markersize=8, color='g', label='Buy Signal (Strategy 2)')
plt.plot(data[data['Signal_Strategy_2'] == 0].index, data[data['Signal_Strategy_2'] == 0]['Close'], 'v', markersize=8, color='r', label='Sell Signal (Strategy 2)')
plt.title('Salesforce Inc. (CRM) Stock Price and Trading Signals (Strategy 2)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# Accuracy for Strategy 1 Classifiers
accuracy_knn_1 = accuracy_score(y_test_1, y_pred_knn_1)
accuracy_rf_1 = accuracy_score(y_test_1, y_pred_rf_1)
accuracy_gb_1 = accuracy_score(y_test_1, y_pred_gb_1)
accuracy_svm_1 = accuracy_score(y_test_1, y_pred_svm_1)
accuracy_xgb_1 = accuracy_score(y_test_1, y_pred_xgb_1)

# Accuracy for Strategy 2 (K-Nearest Neighbors)
accuracy_knn_2 = accuracy_score(y_test_2_knn, y_pred_knn_2)

# Print the accuracies
print("Accuracy for Strategy 1 Classifiers:")
print(f"K-Nearest Neighbors: {accuracy_knn_1}")
print(f"Random Forest: {accuracy_rf_1}")
print(f"Gradient Boosting: {accuracy_gb_1}")
print(f"Support Vector Machines: {accuracy_svm_1}")
print(f"XGBoost: {accuracy_xgb_1}")

print("Accuracy for Strategy 2 (K-Nearest Neighbors):")
print(f"K-Nearest Neighbors: {accuracy_knn_2}")

#extra credit


# Split data into training and test sets (80/20 split) for Strategy 1
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X_strategy_1, y_strategy_1, test_size=0.2, random_state=42)

# Initialize classifiers
rf_classifier = RandomForestClassifier(random_state=42)
gb_classifier = GradientBoostingClassifier(random_state=42)
svm_classifier = SVC(random_state=42)

# Define hyperparameter grids for grid search
param_grid_rf = {
    'n_estimators': [50, 100, 200],                 # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],               # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],              # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]                # Minimum number of samples required to be at a leaf node
}

param_grid_gb = {
    'n_estimators': [50, 100, 200],               # Number of boosting stages (trees)
    'learning_rate': [0.01, 0.1, 0.2],           # Step size shrinkage used to prevent overfitting
    'max_depth': [3, 4, 5]                       # Maximum depth of each tree
}

param_grid_svc = {
    'C': [0.1, 1, 10],                           # Regularization parameter
    'kernel': ['linear', 'rbf'],                 # Kernel type for SVM
    'gamma': ['scale', 'auto', 0.1, 1]          # Kernel coefficient (scale for 'rbf', auto for 'linear')
}

# Perform grid search for each classifier
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search_svc = GridSearchCV(svm_classifier, param_grid_svc, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit the grid search models
grid_search_rf.fit(X_train_1, y_train_1)
grid_search_gb.fit(X_train_1, y_train_1)
grid_search_svc.fit(X_train_1, y_train_1)

# Get the best parameters for each classifier
best_params_rf = grid_search_rf.best_params_
best_params_gb = grid_search_gb.best_params_
best_params_svc = grid_search_svc.best_params_

# Create new classifiers with the best parameters
best_clf_rf = RandomForestClassifier(**best_params_rf, random_state=42)
best_clf_gb = GradientBoostingClassifier(**best_params_gb, random_state=42)
best_clf_svc = SVC(**best_params_svc, random_state=42)

# Fit the best classifiers on the training data
best_clf_rf.fit(X_train_1, y_train_1)
best_clf_gb.fit(X_train_1, y_train_1)
best_clf_svc.fit(X_train_1, y_train_1)

# Predictions for Strategy 1 with tuned classifiers
y_pred_rf_tuned = best_clf_rf.predict(X_test_1)
y_pred_gb_tuned = best_clf_gb.predict(X_test_1)
y_pred_svc_tuned = best_clf_svc.predict(X_test_1)

# Evaluate tuned classifiers for Strategy 1
print("Tuned Random Forest Classifier (Strategy 1) Evaluation:")
print(classification_report(y_test_1, y_pred_rf_tuned))  # Classification report for Random Forest in Strategy 1

print("Tuned Gradient Boosting Classifier (Strategy 1) Evaluation:")
print(classification_report(y_test_1, y_pred_gb_tuned))  # Classification report for Gradient Boosting in Strategy 1

print("Tuned Support Vector Machines Classifier (Strategy 1) Evaluation:")
print(classification_report(y_test_1, y_pred_svc_tuned))  # Classification report for SVM in Strategy 1