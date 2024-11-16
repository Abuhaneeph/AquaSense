# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('water_quality_data.csv')  

# Display the first few rows to understand the structure
print("First few rows of the dataset:")
print(data.head())

# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# Display basic statistics to understand the data distribution
print("\nStatistical summary of the dataset:")
print(data.describe())

# Prepare Features (X) and Target (y)
X = data[['Temperature (°C)', 'pH', 'Turbidity (NTU)', 'BOD (mg/L)', 'Nitrate (mg/L)', 'Phosphate (mg/L)']]
y = data['DO (mg/L)']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Fit the model on the training data
linear_model.fit(X_train, y_train)

# Print model coefficients for interpretation
print("\nLinear Regression Coefficients:", linear_model.coef_)
print("Intercept:", linear_model.intercept_)

# Make predictions using Linear Regression
linear_pred = linear_model.predict(X_test)

# Calculate performance metrics for Linear Regression
linear_mse = mean_squared_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

print("\nLinear Regression Performance:")
print("Mean Squared Error (MSE):", linear_mse)
print("R-squared (R²):", linear_r2)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the Random Forest model on the training data
rf_model.fit(X_train, y_train)

# Make predictions using Random Forest
rf_pred = rf_model.predict(X_test)

# Calculate performance metrics for Random Forest
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\nRandom Forest Regression Performance:")
print("Mean Squared Error (MSE):", rf_mse)
print("R-squared (R²):", rf_r2)

# Visualize actual vs. predicted values for both models
plt.figure(figsize=(12, 6))

# Linear Regression plot
plt.subplot(1, 2, 1)
plt.scatter(y_test, linear_pred, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual DO (mg/L)")
plt.ylabel("Predicted DO (mg/L)")
plt.title("Linear Regression: Actual vs. Predicted")

# Random Forest plot
plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, alpha=0.7, color='g')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual DO (mg/L)")
plt.ylabel("Predicted DO (mg/L)")
plt.title("Random Forest: Actual vs. Predicted")

plt.tight_layout()
plt.show()
