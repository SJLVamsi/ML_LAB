# =======================
# LAB 10 - Feature Reduction & Explainable AI
# =======================

# --- Import Dependencies ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import shap
from lime.lime_tabular import LimeTabularExplainer
import os

# =======================
# A1. FEATURE CORRELATION ANALYSIS
# =======================

def feature_correlation_analysis(df):
    corr = df.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("A1_Correlation_Heatmap.png")
    plt.show()
    print("✅ Saved: A1_Correlation_Heatmap.png")

# =======================
# A2 & A3. PCA DIMENSIONALITY REDUCTION
# =======================

def pca_analysis(X, y, variance_ratio):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(variance_ratio)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Number of components to retain {variance_ratio*100:.0f}% variance: {pca.n_components_}")

    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {variance_ratio*100:.0f}% variance retained: {acc:.4f}")
    return model, X_train, X_test, y_train, y_test

# =======================
# A4. SEQUENTIAL FEATURE SELECTION
# =======================

def sequential_feature_selection(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=500)

    total_features = X.shape[1]
    num_features_to_select = max(1, total_features - 1)  # must be strictly less than total
    print(f"Selecting top {num_features_to_select} out of {total_features} features...")

    sfs = SequentialFeatureSelector(model, n_features_to_select=num_features_to_select, direction="forward")
    sfs.fit(X_scaled, y)

    selected_features = X.columns[sfs.get_support()]
    print("Selected Features:", list(selected_features))

    X_selected = X_scaled[:, sfs.get_support()]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy with Sequential Feature Selection: {acc:.4f}")
    return model, X_train, X_test, y_train, y_test, selected_features

# =======================
# A5. LIME & SHAP EXPLAINABILITY
# =======================

def explain_model_with_lime_shap(model, X_train, X_test, feature_names):
    # ---- LIME ----
    print("\nGenerating LIME Explanation...")
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=["Class 0", "Class 1"],
        mode="classification"
    )
    exp = explainer.explain_instance(X_test[0], model.predict_proba)
    exp.save_to_file("A5_LIME_Explanation.html")
    print("✅ Saved: A5_LIME_Explanation.html")

    # ---- SHAP ----
    print("\nGenerating SHAP Explanation...")
    #shap.initjs()
    explainer_shap = shap.Explainer(model, X_train)
    shap_values = explainer_shap(X_test)

    plt.title("SHAP Feature Importance Summary")
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("A5_SHAP_Summary.png")
    plt.close()
    print("✅ Saved: A5_SHAP_Summary.png")

# =======================
# MAIN PROGRAM
# =======================

if __name__ == "__main__":
    # Load dataset
    df = pd.read_excel("dataset.xlsx")
    print("Dataset Loaded Successfully!\n")
    print(df.head())

    # Identify features & target
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- A1: Correlation ---
    feature_correlation_analysis(X)

    # --- A2: PCA (99% variance) ---
    model_99, X_train_99, X_test_99, y_train_99, y_test_99 = pca_analysis(X, y, 0.99)

    # --- A3: PCA (95% variance) ---
    model_95, X_train_95, X_test_95, y_train_95, y_test_95 = pca_analysis(X, y, 0.95)

    # --- A4: Sequential Feature Selection ---
    model_sfs, X_train_sfs, X_test_sfs, y_train_sfs, y_test_sfs, selected_features = sequential_feature_selection(X, y)

    # --- A5: LIME & SHAP ---
    explain_model_with_lime_shap(model_sfs, X_train_sfs, X_test_sfs, selected_features)

    print("\n✅ All tasks (A1–A5) completed successfully.")
