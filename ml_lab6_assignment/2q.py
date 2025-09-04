import pandas as pd

# Load your dataset
df = pd.read_excel("dataset.xlsx")

# --- A2: Function to calculate Gini index ---
def calculate_gini(column):
    """Calculate Gini index of a pandas Series (categorical target)"""
    counts = column.value_counts(normalize=True)  # probabilities p_i
    gini = 1 - sum(p**2 for p in counts)
    return gini

# Calculate gini index of the target column
target_gini = calculate_gini(df['label'])
print("Gini Index of target (label):", target_gini)
