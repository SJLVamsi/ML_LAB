import pandas as pd

# Load dataset
df = pd.read_excel("dataset.xlsx")

# --- A4: Binning function ---
def bin_feature(series, bins=4, method="equal_width"):
    """
    Bin a continuous feature into categorical values.
    method = "equal_width" or "equal_frequency"
    """
    if method == "equal_width":
        binned = pd.cut(series, bins=bins, labels=False)
    elif method == "equal_frequency":
        binned = pd.qcut(series, q=bins, labels=False, duplicates='drop')
    else:
        raise ValueError("method must be 'equal_width' or 'equal_frequency'")
    return binned

# Example usage:
df['i_binned_width'] = bin_feature(df['i'], bins=4, method="equal_width")
df['i_binned_freq'] = bin_feature(df['i'], bins=4, method="equal_frequency")

print(df[['i', 'i_binned_width', 'i_binned_freq']].head(10))

"""
Continuous features were converted to categorical values using binning. 
For feature i, equal-width binning divided the range into four equal intervals,
while equal-frequency binning ensured that each bin contained ~25% of the rows. 
This process allows entropy and information gain to be computed on categorical splits, 
as required for decision tree construction."""