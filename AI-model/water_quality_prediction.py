# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Print model coefficients for interpretation
print("\nModel Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²):", r2)

# Visualize actual vs. predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r')
plt.xlabel("Actual DO (mg/L)")
plt.ylabel("Predicted DO (mg/L)")
plt.title("Actual vs. Predicted DO (mg/L)")
plt.show()
