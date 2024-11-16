# Water Quality Prediction using Linear Regression and Random Forest

This project predicts **Dissolved Oxygen (DO)** levels in water samples using water quality parameters. It uses two machine learning models, **Linear Regression** and **Random Forest Regression**, and compares their performance. The repository includes data preprocessing, model training, evaluation, and visualization.

---

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Code Workflow](#code-workflow)
- [Output Example](#output-example)
- [Improvements and Extensions](#improvements-and-extensions)
- [License](#license)
- [Author](#author)

---

## Features
- **Data Exploration**: Summarizes the data and identifies missing values.
- **Linear Regression Model**: Basic regression model for predictions.
- **Random Forest Model**: An ensemble method for more robust predictions.
- **Performance Evaluation**: Compares models using **Mean Squared Error (MSE)** and **R-squared (R²)**.
- **Visualization**: Scatter plots for Actual vs. Predicted DO values for both models.

---

## Dataset
The project expects a CSV file named `water_quality_data.csv` with the following columns:
- `Temperature (°C)`
- `pH`
- `Turbidity (NTU)`
- `BOD (mg/L)`
- `Nitrate (mg/L)`
- `Phosphate (mg/L)`
- `DO (mg/L)` (target variable)

### Example Structure:
| Temperature (°C) | pH   | Turbidity (NTU) | BOD (mg/L) | Nitrate (mg/L) | Phosphate (mg/L) | DO (mg/L) |
|-------------------|-------|----------------|------------|----------------|------------------|-----------|
| 22.5             | 7.2   | 3.1            | 2.8        | 0.5            | 0.03             | 8.4       |
| 25.1             | 6.8   | 2.7            | 3.0        | 0.6            | 0.05             | 7.9       |

---

## Requirements
Install the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install them with:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
