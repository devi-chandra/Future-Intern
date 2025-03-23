import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
# Make sure to replace 'house_prices.csv' with the path to your dataset
dataset = pd.read_csv('house_prices.csv')

# Display the first few rows
print("First few rows of the dataset:")
print(dataset.head())

# Fill missing values for the target variable
dataset['SalePrice'].fillna(dataset['SalePrice'].mean(), inplace=True)

# OneHotEncoding for categorical features
categorical_cols = dataset.select_dtypes(include=['object']).columns
dataset = pd.get_dummies(dataset, columns=categorical_cols, drop_first=True)

# Drop irrelevant columns
dataset.drop(['Id'], axis=1, inplace=True)

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Separate features and target variable
X = dataset.drop('SalePrice', axis=1)
y = dataset['SalePrice']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
print('Linear Regression MAE:', mean_absolute_error(y_test, y_pred_linear))
print('Linear Regression MSE:', mean_squared_error(y_test, y_pred_linear))

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print('Random Forest MAE:', mean_absolute_error(y_test, y_pred_rf))
print('Random Forest MSE:', mean_squared_error(y_test, y_pred_rf))

# Support Vector Regressor (SVR)
svr_model = SVR(kernel='linear')
svr_model.fit(X_train, y_train)
y_pred_svr = svr_model.predict(X_test)
print('SVR MAE:', mean_absolute_error(y_test, y_pred_svr))
print('SVR MSE:', mean_squared_error(y_test, y_pred_svr))

# Conclusion
print("\nModel evaluation completed.")