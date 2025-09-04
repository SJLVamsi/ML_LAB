import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("dataset.xlsx")

# --- Helper: Entropy ---
def calculate_entropy(column):
    counts = column.value_counts(normalize=True)
    return -sum(p * np.log2(p) for p in counts if p > 0)

# --- Helper: Info Gain ---
def information_gain(data, feature, target="label", bins=4):
    """Calculate information gain of a feature with respect to target"""
    
    # If feature is continuous -> bin into categories
    binned = pd.cut(data[feature], bins=bins, labels=False)
    
    # Entropy of full dataset
    total_entropy = calculate_entropy(data[target])
    
    # Weighted entropy after split
    weighted_entropy = 0
    for val in binned.unique():
        subset = data[data[feature].apply(lambda x: pd.cut([x], bins=bins, labels=False)[0]) == val]
        if len(subset) > 0:
            weight = len(subset) / len(data)
            weighted_entropy += weight * calculate_entropy(subset[target])
    
    # Information gain
    ig = total_entropy - weighted_entropy
    return ig

# --- Find root node (best feature) ---
features = ['i', 'ii', 'v1', 'v5']
ig_scores = {f: information_gain(df, f) for f in features}

root_feature = max(ig_scores, key=ig_scores.get)

print("Information Gain Scores:", ig_scores)
print("Best root node feature:", root_feature)
"""
Information gain was computed for four features (i, ii, v1, v5). 
The results showed that features i, v1, and v5 had very high IG values (~0.99), 
indicating strong separation of classes.
Feature ii had an IG of 0.0, suggesting no contribution to classification. 
Based on this, the decision tree selected i as the root node. 
However, v1 or v5 could also serve as valid root nodes due to identical information gain.
"""