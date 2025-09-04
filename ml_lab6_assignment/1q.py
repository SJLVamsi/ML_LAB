import pandas as pd
import numpy as np

# Load your dataset
df = pd.read_excel("dataset.xlsx")

# --- A1: Function to calculate entropy ---
def calculate_entropy(column):
    """Calculate entropy of a pandas Series (categorical target)"""
    counts = column.value_counts(normalize=True)  # probabilities p_i
    entropy = -sum(p * np.log2(p) for p in counts if p > 0)
    return entropy

# Calculate entropy of the target column
target_entropy = calculate_entropy(df['label'])
print("Entropy of target (label):", target_entropy)

