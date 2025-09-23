# A2_randomized_search.py
"""
Lab07 - A2
Hyperparameter tuning using RandomizedSearchCV

Run:
    python A2_randomized_search.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore")

# ----------------- Load dataset -----------------
DATA_PATH = "dataset.xlsx"   # put your file in same folder
df = pd.read_excel(DATA_PATH, sheet_name="Sheet1")

X = df.drop(columns=["label"])
y = df["label"]

# ----------------- Split -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ----------------- Define models & param grids -----------------
models = {
    "RandomForest": (
        Pipeline([
            ("scaler", StandardScaler()), 
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        {
            "clf__n_estimators": [50, 100, 200],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2", None]
        }
    ),
    "SVC": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=42))
        ]),
        {
            "clf__C": np.logspace(-2, 2, 5),
            "clf__gamma": ["scale", "auto"],
            "clf__kernel": ["linear", "rbf"]
        }
    )
}

# ----------------- Run RandomizedSearchCV -----------------
for name, (pipeline, param_dist) in models.items():
    print(f"\n===== {name} =====")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=20,  # number of random combinations
        scoring="f1", 
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print("Best parameters:", search.best_params_)
    print("Best CV F1 Score:", search.best_score_)
    
    # Evaluate on train and test
    y_train_pred = search.predict(X_train)
    y_test_pred = search.predict(X_test)
    
    print("\nTrain Accuracy:", accuracy_score(y_train, y_train_pred))
    print("Train F1 Score:", f1_score(y_train, y_train_pred))
    print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
    print("Test F1 Score:", f1_score(y_test, y_test_pred))
    
    print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))
