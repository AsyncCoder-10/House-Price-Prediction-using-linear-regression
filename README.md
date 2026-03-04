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

Project Setup:-
# Create project directory
mkdir house_price_prediction
cd house_price_prediction

# (Optional) Create and activate a virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
# Install packages
pip install -r requirements.txt

Output
<img width="1267" height="413" alt="image" src="https://github.com/user-attachments/assets/da369dd7-c26d-4502-a0d1-454e0f137418" />
<img width="1655" height="910" alt="image" src="https://github.com/user-attachments/assets/3c432174-5b65-4418-b070-4a4e196d138d" />
<img width="1663" height="932" alt="image" src="https://github.com/user-attachments/assets/58c7c16d-38f9-4e10-af71-f63a9728c6db" />


