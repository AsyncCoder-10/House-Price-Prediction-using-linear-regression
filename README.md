# House-Price-Prediction-using-linear-regression
Supervised learning, feature engineering, evaluation metrics, overfitting, regularization.


Project - 1
House Price Prediction using linear regression, train/test split, evaluation metrics MSE and R², feature scaling. Use California Housing dataset. Tools: Python, pandas, scikit-learn, matplotlib.

We'll cover:
Loading data
Exploratory Data Analysis (EDA) with matplotlib
Feature scaling (StandardScaler)
Train/test split
Linear regression model
Evaluation: MSE, R²
Possibly visualizations: actual vs predicted, residuals.

Let's structure the answer:
Introduction: goal, dataset.
Setup: import libraries.
Load dataset: fetch_california_housing, create DataFrame.
Exploratory Data Analysis: describe, histograms, correlations (optional).
Feature scaling: StandardScaler (fit on training, transform both train and test).
Split data: train_test_split (before or after scaling? Typically scaling after split to avoid data leakage. We'll scale after split.)
Train linear regression model.
Evaluate: predict on test, calculate MSE (Mean Squared Error) and R².
Visualization: scatter plot of actual vs predicted, residuals.
Conclusion.

