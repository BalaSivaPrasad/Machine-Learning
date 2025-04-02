import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Sample dataset: simple data for classifying if a person will buy a product
# Features: Age, Income (1: Low, 2: Medium, 3: High)
data = {'Age': [25, 45, 35, 50, 23, 40, 60, 48, 33, 55],
'Income': [1, 2, 2, 3, 1, 2, 3, 3, 1, 3],
'Buy': ['No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']}

# Convert data to DataFrame
df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# Step 1: Define features (X) and target (y)
X = df[['Age', 'Income']].values # Features: Age and Income
y = df['Buy'].values # Target: Buy (Yes/No)

# Step 2: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Initialize and train the Decision Tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = clf.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=['Age', 'Income'], class_names=['No', 'Yes'], filled=True)
plt.title('Decision Tree for Classification')
plt.show()
