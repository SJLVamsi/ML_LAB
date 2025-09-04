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
    binned = pd.cut(data[feature], bins=bins, labels=False)
    total_entropy = calculate_entropy(data[target])
    
    weighted_entropy = 0
    for val in binned.unique():
        subset = data[binned == val]
        if len(subset) > 0:
            weight = len(subset) / len(data)
            weighted_entropy += weight * calculate_entropy(subset[target])
    return total_entropy - weighted_entropy

# --- Decision Tree Builder ---
def build_tree(data, features, target="label", depth=0, max_depth=3):
    # If all labels are same → leaf node
    if len(data[target].unique()) == 1:
        return data[target].iloc[0]
    
    # If no features left OR max depth reached → majority class
    if len(features) == 0 or depth == max_depth:
        return data[target].mode()[0]
    
    # Find best feature by IG
    ig_scores = {f: information_gain(data, f) for f in features}
    best_feature = max(ig_scores, key=ig_scores.get)
    
    tree = {best_feature: {}}
    
    # Bin best feature
    binned = pd.cut(data[best_feature], bins=4, labels=False)
    
    # Recurse for each bin
    for val in binned.unique():
        subset = data[binned == val]
        if subset.empty:
            tree[best_feature][val] = data[target].mode()[0]
        else:
            remaining_features = [f for f in features if f != best_feature]
            tree[best_feature][val] = build_tree(
                subset, remaining_features, target, depth+1, max_depth
            )
    
    return tree

# --- Build tree on your dataset ---
features = ['i', 'ii', 'v1', 'v5']
decision_tree = build_tree(df, features, target="label", max_depth=3)

print("Decision Tree:", decision_tree)
