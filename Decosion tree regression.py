import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.tree import plot_tree

# Load dataset (replace 'house_prices.csv' with your actual file)
df = pd.read_csv('house prices.csv')

# Display first few rows
print(df.head())

# Drop missing values
df = df.dropna()

# Features and target variable
X = df[['sqft_living', 'bedrooms', 'bathrooms', 'floors', 'zipcode']]
y = df['prices']

# Convert categorical features (e.g., zipcode) using one-hot encoding
X = pd.get_dummies(X, columns=['zipcode'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
dt_regressor = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_regressor.fit(X_train, y_train)

# Make predictions
y_pred = dt_regressor.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display the results
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")

# Optional: Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_regressor, feature_names=X.columns, filled=True)
plt.title("Decision Tree Regressor")
plt.show()
