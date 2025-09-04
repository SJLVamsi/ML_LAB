import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_excel("dataset.xlsx")

# Features and target
X = df[['i', 'ii', 'v1', 'v5']]
y = df['label']

# Train decision tree (use entropy for consistency with IG)
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X, y)

# --- Plot the tree ---
plt.figure(figsize=(12,8))
plot_tree(clf,
          feature_names=['i', 'ii', 'v1', 'v5'],
          class_names=['Healthy (0)', 'Diseased (1)'],
          filled=True,
          rounded=True,
          fontsize=10)
plt.show()
"""""
The decision tree was visualized using scikit-learn’s plot_tree().
The root node corresponded to feature i,
 splitting the dataset into Healthy and Diseased groups."""