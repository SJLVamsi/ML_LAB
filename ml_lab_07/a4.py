# A4_regressors.py
"""
Lab07 - A4
Compare regressors on regression problem

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Regressors
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings("ignore")

# ----------------- Load dataset -----------------
DATA_PATH = "dataset.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

# Assume last column is target (regression task)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ----------------- Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------- Define regressors -----------------
regressors = {
    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", SVR())
    ]),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, eval_metric="rmse"),
    "CatBoost": CatBoostRegressor(verbose=0, random_state=42),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("reg", MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=42))
    ])
}

# ----------------- Train & Evaluate -----------------
results = []

for name, reg in regressors.items():
    reg.fit(X_train, y_train)
    
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    
    metrics = {
        "Model": name,
        "Train_RMSE": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "Train_MAE": mean_absolute_error(y_train, y_train_pred),
        "Train_R2": r2_score(y_train, y_train_pred),
        "Test_RMSE": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "Test_MAE": mean_absolute_error(y_test, y_test_pred),
        "Test_R2": r2_score(y_test, y_test_pred),
    }
    
    results.append(metrics)

# ----------------- Tabulate results -----------------
results_df = pd.DataFrame(results)
print("\n=== Regressor Comparison ===\n")
print(results_df)

# Save to CSV
results_df.to_csv("A4_results.csv", index=False)
print("\nResults saved to A4_results.csv")