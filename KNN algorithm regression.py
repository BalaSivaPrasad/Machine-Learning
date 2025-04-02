import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample dataset: simple data for regression (house price prediction based on size)
data = {'Size': [600, 650, 700, 850, 900, 1200, 1500, 1850, 2000, 2500],
'Price': [150, 200, 250, 275, 300, 450, 500, 600, 650, 700]}

# Convert data to DataFrame
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# Step 1: Define features (X) and target (y)
X = df[['Size']].values # Feature: Size of the house
y = df['Price'].values # Target: Price of the house

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Normalize features using StandardScaler (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Initialize and train the KNN regressor (with k=3 neighbors)
k = 3
knn = KNeighborsRegressor(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

# Step 5: Make predictions
y_pred = knn.predict(X_test_scaled)

# Step 6: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Step 7: Visualize the predictions
plt.scatter(X_test, y_test, color='red', label='Actual Prices')
plt.scatter(X_test, y_pred, color='blue', label='Predicted Prices')
plt.title('KNN Regression: Actual vs Predicted Prices')
plt.xlabel('Size of House (sq ft)')
plt.ylabel('Price (in 1000s)')
plt.legend()
plt.show()
