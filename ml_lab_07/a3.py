# A3_classifiers.py
"""
Lab07 - A3
Compare classifiers on classification problem

Run:
    python A3_classifiers.py

Requirements:
    pip install scikit-learn xgboost catboost
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Classifiers
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings("ignore")

# ----------------- Load dataset -----------------
DATA_PATH = "dataset.xlsx"
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

X = df.drop(columns=["label"])
y = df["label"]

# ----------------- Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------- Define classifiers -----------------
classifiers = {
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "NaiveBayes": GaussianNB(),
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42))
    ])
}

# ----------------- Train & Evaluate -----------------
results = []

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    metrics = {
        "Model": name,
        "Train_Accuracy": accuracy_score(y_train, y_train_pred),
        "Train_Precision": precision_score(y_train, y_train_pred),
        "Train_Recall": recall_score(y_train, y_train_pred),
        "Train_F1": f1_score(y_train, y_train_pred),
        "Test_Accuracy": accuracy_score(y_test, y_test_pred),
        "Test_Precision": precision_score(y_test, y_test_pred),
        "Test_Recall": recall_score(y_test, y_test_pred),
        "Test_F1": f1_score(y_test, y_test_pred),
    }
    
    results.append(metrics)

# ----------------- Tabulate results -----------------
results_df = pd.DataFrame(results)
print("\n=== Classifier Comparison ===\n")
print(results_df)

# Save to CSV
results_df.to_csv("A3_results.csv", index=False)
print("\nResults saved to A3_results.csv")
