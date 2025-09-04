import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_excel("dataset.xlsx")

# Use only 2 features for visualization
X = df[['i', 'v1']]
y = df['label']

# Train decision tree
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=42)
clf.fit(X, y)

# --- Create a meshgrid for plotting decision boundaries ---
x_min, x_max = X['i'].min() - 0.5, X['i'].max() + 0.5
y_min, y_max = X['v1'].min() - 0.5, X['v1'].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict class for each grid point
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# --- Plot decision boundary ---
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

# Plot actual data points
plt.scatter(X['i'], X['v1'], c=y, edgecolors='k', cmap=plt.cm.Paired, s=60)

plt.xlabel("i (Limb Lead)")
plt.ylabel("v1 (Chest Lead)")
plt.title("Decision Boundary of Decision Tree (i vs v1)")
plt.show()
"""
The decision boundary clearly separates most of the healthy and diseased cases"""