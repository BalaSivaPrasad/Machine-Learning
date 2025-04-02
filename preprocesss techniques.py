import pandas as pd
import numpy as np

# Sample dataset
data = {
    'item': ['A', 'B', 'C', 'D', np.nan, 'F', 'G', 'H'],
    'category': ['Electronics', 'Furniture', 'Clothing', 'Electronics', 'Clothing', np.nan, 'Furniture', 'Clothing'],
    'price': [200, 450, 50, 300, np.nan, 700, 120, 80],
    'quantity': [2, 3, 1, 4, 5, np.nan, 2, 3],
    'sales': [400, 1350, 50, 1200, np.nan, 1400, 240, 240],
    'discount': [10, 20, 5, 15, np.nan, 25, 10, 5]
}

# Create DataFrame
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# 1. Attribute Selection (keeping all for now but can drop unnecessary columns)

# 2. Handling Missing Values
# Numerical columns: mean imputation
for col in ['price', 'quantity', 'sales', 'discount']:
    df[col].fillna(df[col].mean(), inplace=True)

# Categorical columns: mode imputation
for col in ['item', 'category']:
    df[col].fillna('Unknown', inplace=True)

print("\nAfter Handling Missing Values:\n", df)

# 3. Discretization
# Binning 'price' into 4 categories: Low, Medium, High, Very High
price_bins = pd.qcut(df['price'], q=4, labels=['Low', 'Medium', 'High', 'Very High'])
df['price_bins'] = price_bins

# Binning 'discount' into 3 categories: Low, Medium, High
discount_bins = pd.cut(df['discount'], bins=[0, 10, 20, 30], labels=['Low', 'Medium', 'High'])
df['discount_bins'] = discount_bins

print("\nAfter Discretization:\n", df)

# 4. Elimination of Outliers using IQR method
for col in ['price', 'sales', 'discount']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

print("\nAfter Eliminating Outliers:\n", df)
